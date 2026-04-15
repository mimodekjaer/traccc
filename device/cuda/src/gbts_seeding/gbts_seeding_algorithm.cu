/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/GbtsGraphMakingKernels.cuh"
#include "./kernels/GbtsGraphProcessingKernels.cuh"
#include "./kernels/GbtsNodesMakingKernels.cuh"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/edm/container.hpp"


// C++ include(s)
#include <ranges>

namespace traccc::cuda {

struct gbts_ctx {
    // counters
    unsigned int nSp{};
    unsigned int nNodes{};
    unsigned int nUsedBinPairs{};
    unsigned int nMaxEdges{};

    unsigned int nEdges{};
    unsigned int nConnections{};
    unsigned int nConnectedEdges{};
    unsigned int nSeeds{};
    // nEdges, nConnections, nConnectedEdges, .., nSeeds
    collection_types<unsigned int>::buffer counters_buf;

    // device side graph building cuts
    gbts_graph_building_params d_graph_building_params;

    // node making and binning
    collection_types<int>::buffer layerCounts_buf;
    collection_types<short>::buffer spacepointsLayer_buf;
    // begin_idx + 1 for the surfaceToLayerMap or -layerBin if one to one
    collection_types<short>::buffer volumeToLayerMap_buf;
    collection_types<std::pair<unsigned int, unsigned int>>::buffer surfaceToLayerMap_buf;  // surface_index, layerBin
    collection_types<char>::buffer layerType_buf;
    // conversion to original sp from post layer binning index
    collection_types<int>::buffer original_sp_idx_buf;
    // conversion to orignal sp/node index from post binning index
    collection_types<int>::buffer node_index_buf;

    // x,y,z,cluster width in eta
    collection_types<float4>::buffer reducedSP_buf;
    // layer binned reducedSP
    collection_types<float4>::buffer sp_params_buf;

    collection_types<std::pair<int, int>>::buffer layer_info_buf;
    collection_types<std::pair<float, float>>::buffer layer_geo_buf; 

    collection_types<int>::buffer node_eta_index_buf;
    collection_types<int>::buffer node_phi_index_buf;

    collection_types<int>::buffer eta_phi_histo_buf;
    collection_types<int>::buffer phi_cusums_buf;
    collection_types<int>::buffer eta_node_counter_buf;

    collection_types<int>::buffer eta_bin_views_buf;
    // eta-bin views of the node_params array
    collection_types<int>::host eta_bin_views;

    collection_types<float>::buffer bin_rads_buf;
    collection_types<float>::host bin_rads;

    collection_types<int>::buffer bin_pair_views_buf;
    collection_types<int>::host bin_pair_views;

    collection_types<float>::buffer bin_pair_dphi_buf;
    collection_types<float>::host bin_pair_dphi;
    // node making output
    collection_types<float>::buffer node_params_buf;

    // GraphMaking
    collection_types<int2>::buffer edge_nodes_buf;
    collection_types<kernels::half4>::buffer edge_params_buf;
    collection_types<int>::buffer num_incoming_edges_buf;
    collection_types<int>::buffer edge_links_buf;
    collection_types<unsigned char>::buffer num_neighbours_buf;
    collection_types<int>::buffer reIndexer_buf;
    collection_types<int>::buffer neighbours_buf;
    // offload this for CPU-side seed extraction
    collection_types<int>::buffer output_graph_buf;

    // message-passing CCA
    // holds indices of the edges that need more CCA iterations
    collection_types<int>::buffer active_edges_buf;
    // d_levels[edge_idx] = the maxium length of seeds starting with this edge
    collection_types<char>::buffer levels_buf;
    // #paths, is terminus
    collection_types<short2>::buffer outgoing_paths_buf;

    // seed-extraction walkthrough

    // edge_idx and prev path_store idx forms a uniuqe path through the graph
    collection_types<int2>::buffer path_store_buf;
    collection_types<int2>::buffer seed_proposals_buf;  // int quality and final mini_state_idx
    // first 32 bits are seed quality second 32 bits are seed_proposals index
    collection_types<unsigned long long int>::buffer edge_bids_buf;
    // 0 as default/is real seed, 1 as maybe seed,
    //-1 as maybe fake seed, -2 as fake
    collection_types<char>::buffer seed_ambiguity_buf;
};

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(logger->clone()),
      m_config(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str) {}

gbts_seeding_algorithm::output_type gbts_seeding_algorithm::operator()(
    const traccc::edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements) const {

    gbts_ctx ctx;

    cudaStream_t stream = details::get_stream(m_stream);
    ctx.d_graph_building_params = m_config.graph_building_params;
    //cudaMalloc(&ctx.d_graph_building_params,
    //           sizeof(m_config.graph_building_params));
    //cudaMemcpyAsync(
    //    ctx.d_graph_building_params, &m_config.graph_building_params,
    //    sizeof(m_config.graph_building_params), cudaMemcpyHostToDevice);

    // 0. bin spacepoints by the maping supplied to config.m_surfaceToLayerMap
    ctx.nSp = m_copy.get().get_size(spacepoints);
    TRACCC_DEBUG("nSp " << ctx.nSp);
    if (ctx.nSp == 0) {
        return {0, m_mr.main};
    }

    unsigned int nThreads = 128;
    unsigned int nBlocks = 1 + (ctx.nSp - 1) / nThreads;

    ctx.layerCounts_buf = collection_types<int>::buffer(m_config.nLayers + 1, m_mr.main);
    m_copy.get().memset(ctx.layerCounts_buf, 0)->ignore();

    ctx.reducedSP_buf = collection_types<float4>::buffer(ctx.nSp, m_mr.main);
    m_copy.get().setup(ctx.reducedSP_buf)->ignore();

    ctx.spacepointsLayer_buf = collection_types<short>::buffer(ctx.nSp, m_mr.main);
    m_copy.get().setup(ctx.spacepointsLayer_buf)->ignore();

    ctx.volumeToLayerMap_buf = collection_types<short>::buffer(static_cast<uint>(m_config.volumeToLayerMap.size()), m_mr.main);
    m_copy.get().setup(ctx.volumeToLayerMap_buf)->ignore();
    m_copy.get()(vecmem::get_data(m_config.volumeToLayerMap), ctx.volumeToLayerMap_buf)->ignore();

    if (m_config.surfaceToLayerMap.size() != 0) {
        ctx.surfaceToLayerMap_buf = collection_types<std::pair<unsigned int, unsigned int>>::buffer(static_cast<uint>(m_config.surfaceToLayerMap.size()), m_mr.main);
        m_copy.get().setup(ctx.surfaceToLayerMap_buf)->ignore();
        m_copy.get()(vecmem::get_data(m_config.surfaceToLayerMap), ctx.surfaceToLayerMap_buf)->ignore();
    }  // may be zero and correct, volumeMapSize, nLayers are checked at config

    ctx.layerType_buf = collection_types<char>::buffer(m_config.nLayers, m_mr.main);
    m_copy.get().setup(ctx.layerType_buf)->ignore();
    m_copy.get()(vecmem::get_data(m_config.layerInfo.type), ctx.layerType_buf)->ignore();


    kernels::count_sp_by_layer<<<nBlocks, nThreads, 0, stream>>>(
        spacepoints, measurements, ctx.volumeToLayerMap_buf,
        ctx.surfaceToLayerMap_buf, ctx.layerType_buf, ctx.reducedSP_buf,
        ctx.layerCounts_buf, ctx.spacepointsLayer_buf,
        m_config.graph_building_params.type1_max_width, ctx.nSp,
        m_config.volumeToLayerMap.size(), m_config.surfaceToLayerMap.size());

    cudaStreamSynchronize(stream);

    // prefix sum layerCounts
    collection_types<int>::host layerCounts(m_config.nLayers + 1, m_mr.host);
    m_copy.get()(vecmem::get_data(ctx.layerCounts_buf), layerCounts)->ignore();
    
    //std::inclusive_scan(layerCounts.begin(), layerCounts.end(), layerCounts.begin()); // Mabye use a GPU version of a scan?
    for (size_t layer = 0; layer < m_config.nLayers; layer++) {
        layerCounts[layer + 1] += layerCounts[layer];
    }
    m_copy.get()(vecmem::get_data(layerCounts),
                 ctx.layerCounts_buf)->wait();

    ctx.nNodes = static_cast<unsigned int>(layerCounts[m_config.nLayers]); // The number of spacepoints in the configured layers
    //ctx.nNodes = ctx.nSp; // TODO: This is a temporary fix to avoid the issue with the layerCounts buffer
    TRACCC_DEBUG("nNodes " << ctx.nNodes);
    if (ctx.nNodes == 0)
        return {0, m_mr.main};
    //layerCounts.reset();

    ctx.sp_params_buf = collection_types<float4>::buffer(ctx.nSp, m_mr.main);
    m_copy.get().setup(ctx.sp_params_buf)->ignore();
    ctx.original_sp_idx_buf = collection_types<int>::buffer(ctx.nSp, m_mr.main);
    m_copy.get().setup(ctx.original_sp_idx_buf)->ignore();

    kernels::bin_sp_by_layer<<<nBlocks, nThreads, 0, stream>>>(
        ctx.sp_params_buf, ctx.reducedSP_buf, ctx.layerCounts_buf,
        ctx.spacepointsLayer_buf, ctx.original_sp_idx_buf, ctx.nSp);

    cudaStreamSynchronize(stream);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // 1. histogram spacepoints by layer->eta->phi and convert to nodes
    // do this in config setup?
    ctx.layer_info_buf = collection_types<std::pair<int, int>>::buffer(m_config.nLayers, m_mr.main);
    m_copy.get().setup(ctx.layer_info_buf)->ignore();
    m_copy.get()(vecmem::get_data(m_config.layerInfo.info), ctx.layer_info_buf)->ignore();

    ctx.layer_geo_buf = collection_types<std::pair<float, float>>::buffer(m_config.nLayers, m_mr.main);
    m_copy.get().setup(ctx.layer_geo_buf)->ignore();
    m_copy.get()(vecmem::get_data(m_config.layerInfo.geo), ctx.layer_geo_buf)->ignore();

    ctx.node_phi_index_buf = collection_types<int>::buffer(ctx.nNodes, m_mr.main);
    m_copy.get().setup(ctx.node_phi_index_buf)->ignore();

    nThreads = 256;
    unsigned int nNodesPerBlock = nThreads * 64;

    nBlocks = 1 + (ctx.nNodes - 1) / nNodesPerBlock;

    kernels::node_phi_binning_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.sp_params_buf, ctx.node_phi_index_buf, nNodesPerBlock, ctx.nNodes,
        m_config.n_phi_bins);

    cudaStreamSynchronize(stream);

    ctx.node_eta_index_buf = collection_types<int>::buffer(ctx.nNodes, m_mr.main);
    m_copy.get().setup(ctx.node_eta_index_buf)->ignore();

    nBlocks = m_config.nLayers;

    kernels::node_eta_binning_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.sp_params_buf, ctx.layer_info_buf, ctx.layer_geo_buf,
        ctx.node_eta_index_buf, ctx.layerCounts_buf, m_config.nLayers);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    unsigned int hist_size = m_config.n_eta_bins * m_config.n_phi_bins;
    ctx.eta_phi_histo_buf = collection_types<int>::buffer(hist_size, m_mr.main);
    m_copy.get().setup(ctx.eta_phi_histo_buf)->ignore();
    m_copy.get().memset(ctx.eta_phi_histo_buf, 0)->ignore();
    ctx.phi_cusums_buf = collection_types<int>::buffer(hist_size, m_mr.main);
    m_copy.get().setup(ctx.phi_cusums_buf)->ignore();

    nBlocks = 1 + (ctx.nNodes - 1) / nNodesPerBlock;

    kernels::eta_phi_histo_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.node_phi_index_buf, ctx.node_eta_index_buf, ctx.eta_phi_histo_buf,
        nNodesPerBlock, ctx.nNodes, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    ctx.eta_node_counter_buf = collection_types<int>::buffer(m_config.n_eta_bins, m_mr.main);
    m_copy.get().setup(ctx.eta_node_counter_buf)->ignore();

    unsigned int nBinsPerBlock = 128;

    nThreads = nBinsPerBlock;

    nBlocks = 1 + (m_config.n_eta_bins - 1) / nBinsPerBlock;

    kernels::eta_phi_counting_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.eta_phi_histo_buf, ctx.eta_node_counter_buf, ctx.phi_cusums_buf,
        nBinsPerBlock, m_config.n_eta_bins, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    collection_types<int>::host eta_sums(m_config.n_eta_bins, m_mr.host);
    m_copy.get()(vecmem::get_data(ctx.eta_node_counter_buf), eta_sums)->wait();

    for (unsigned int k = 0; k < m_config.n_eta_bins; k++) { // TODO: Maybe use a GPU version of a scan? or std version?
        eta_sums[k + 1] += eta_sums[k];
    }
    // send back
    m_copy.get()(vecmem::get_data(eta_sums), ctx.eta_node_counter_buf)->wait();

    ctx.eta_bin_views = collection_types<int>::host(2 * m_config.n_eta_bins, m_mr.host);

    for (unsigned view_idx = 0; view_idx < m_config.n_eta_bins; view_idx++) {
        unsigned int pos = 2 * view_idx;
        ctx.eta_bin_views[pos] = (view_idx == 0) ? 0 : eta_sums[view_idx - 1];
        ctx.eta_bin_views[pos + 1] = eta_sums[view_idx];
    }

    cudaStreamSynchronize(stream);

    kernels::eta_phi_prefix_sum_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.eta_node_counter_buf, ctx.phi_cusums_buf, nBinsPerBlock,
        m_config.n_eta_bins, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    ctx.node_params_buf = collection_types<float>::buffer(5 * ctx.nNodes, m_mr.main);
    m_copy.get().setup(ctx.node_params_buf)->ignore();
    ctx.node_index_buf = collection_types<int>::buffer(ctx.nNodes, m_mr.main);
    m_copy.get().setup(ctx.node_index_buf)->ignore();

    nThreads = 256;
    nNodesPerBlock = nThreads * 64;

    nBlocks = 1 + (ctx.nNodes - 1) / nNodesPerBlock;

    kernels::node_sorting_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.sp_params_buf, ctx.node_eta_index_buf, ctx.node_phi_index_buf,
        ctx.phi_cusums_buf, ctx.node_params_buf, ctx.node_index_buf,
        ctx.original_sp_idx_buf, ctx.d_graph_building_params, nNodesPerBlock,
        ctx.nNodes, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);


    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    ctx.eta_bin_views_buf = collection_types<int>::buffer(2 * m_config.n_eta_bins, m_mr.main); // Here the type changed as a quick fix to avoid the issue with the eta_bin_views buffer
    m_copy.get().setup(ctx.eta_bin_views_buf)->ignore();
    m_copy.get()(vecmem::get_data(ctx.eta_bin_views), ctx.eta_bin_views_buf)->wait();

    ctx.bin_rads_buf = collection_types<float>::buffer(2*m_config.n_eta_bins, m_mr.main);
    m_copy.get().setup(ctx.bin_rads_buf)->ignore();

    cudaStreamSynchronize(stream);

    nBinsPerBlock = 128;

    nThreads = nBinsPerBlock;

    nBlocks = 1 + (m_config.n_eta_bins - 1) / nBinsPerBlock;

    kernels::minmax_rad_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.eta_bin_views_buf, ctx.node_params_buf, ctx.bin_rads_buf, nBinsPerBlock,
        m_config.n_eta_bins);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    ctx.bin_rads = collection_types<float>::host(2 * m_config.n_eta_bins, m_mr.host);
    m_copy.get()(vecmem::get_data(ctx.bin_rads_buf), ctx.bin_rads)->wait();

    cudaStreamSynchronize(stream);

    // 2. prepare input for the graph making part of the code:

    int int_nBinPairs = 0;  // the number of eta bin pairs

    for (std::pair<unsigned int, unsigned int> binPair : m_config.binTables) {
        // loop over bin pairs defined by the layer
        // connection table and geometry settings

        int bin1_begin = ctx.eta_bin_views[2 * binPair.first];
        int bin1_end = ctx.eta_bin_views[2 * binPair.first + 1];
        // large bins will be split into smaller sub-views

        int nNodesInBin1 = bin1_end - bin1_begin;
        if (bin1_begin > bin1_end) {
            nNodesInBin1 = bin1_begin - bin1_end;
        }
        int_nBinPairs +=
            1 + (nNodesInBin1 - 1) /
                    traccc::device::gbts_consts::node_buffer_length;
    }
    unsigned int nBinPairs = static_cast<unsigned int>(int_nBinPairs);

    ctx.bin_pair_views = collection_types<int>::host(4 * nBinPairs, m_mr.host);
    ctx.bin_pair_dphi = collection_types<float>::host(nBinPairs, m_mr.host);

    unsigned int pairIdx = 0;
    for (std::pair<unsigned int, unsigned int> binPair : m_config.binTables) {
        float rb1 = ctx.bin_rads[2 * binPair.first];  // min radius

        int begin_bin1 = ctx.eta_bin_views[2 * binPair.first];
        int end_bin1 = ctx.eta_bin_views[2 * binPair.first + 1];
        // skip empty pairs
        if (begin_bin1 == end_bin1)
            continue;
        if (ctx.eta_bin_views[2 * binPair.second] ==
            ctx.eta_bin_views[2 * binPair.second + 1])
            continue;

        float rb2 = ctx.bin_rads[2 * binPair.second + 1];  // max radius

        // max radius of bin2 - min radius of bin1
        float maxDeltaR = std::fabs(rb2 - rb1);

        float deltaPhi = m_config.graph_building_params.min_delta_phi +
                         m_config.graph_building_params.dphi_coeff * maxDeltaR;

        if (maxDeltaR < 60) {
            deltaPhi =
                m_config.graph_building_params.min_delta_phi_low_dr +
                m_config.graph_building_params.dphi_coeff_low_dr * maxDeltaR;
        }
        // splitting large bins into more consistent sizes

        int currBegin_bin1 = begin_bin1;

        int currEnd_bin1 =
            end_bin1 < traccc::device::gbts_consts::node_buffer_length
                ? end_bin1
                : begin_bin1 + traccc::device::gbts_consts::node_buffer_length;

        for (; currEnd_bin1 < end_bin1;
             currEnd_bin1 += traccc::device::gbts_consts::node_buffer_length,
             pairIdx++) {
            unsigned int offset = 4 * pairIdx;

            ctx.bin_pair_views[offset] = currBegin_bin1;
            ctx.bin_pair_views[1 + offset] = currEnd_bin1;
            ctx.bin_pair_views[2 + offset] =
                ctx.eta_bin_views[2 * binPair.second];
            ctx.bin_pair_views[3 + offset] =
                ctx.eta_bin_views[2 * binPair.second + 1];
            ctx.bin_pair_dphi[pairIdx] = deltaPhi;

            currBegin_bin1 = currEnd_bin1;
        }
        currEnd_bin1 = end_bin1;

        unsigned int offset = 4 * pairIdx;

        ctx.bin_pair_views[offset] = currBegin_bin1;
        ctx.bin_pair_views[1 + offset] = currEnd_bin1;
        ctx.bin_pair_views[2 + offset] =
            ctx.eta_bin_views[2 * binPair.second];
        ctx.bin_pair_views[3 + offset] =
            ctx.eta_bin_views[2 * binPair.second + 1];
        ctx.bin_pair_dphi[pairIdx] = deltaPhi;
        pairIdx++;
    }
    ctx.nUsedBinPairs = pairIdx;
    TRACCC_DEBUG("nUsedBinPairs " << ctx.nUsedBinPairs);
    if (ctx.nUsedBinPairs == 0)
        return {0, m_mr.main};
    //ctx.eta_bin_views.reset();
    // allocate memory and copy bin pair views and phi cuts to GPU

    ctx.bin_pair_views_buf = collection_types<int>::buffer(4*ctx.nUsedBinPairs, m_mr.main);
    m_copy.get().setup(ctx.bin_pair_views_buf)->ignore();
    m_copy.get()(vecmem::get_data(ctx.bin_pair_views), ctx.bin_pair_views_buf)->ignore();

    ctx.bin_pair_dphi_buf = collection_types<float>::buffer(ctx.nUsedBinPairs, m_mr.main);
    m_copy.get().setup(ctx.bin_pair_dphi_buf)->ignore();
    m_copy.get()(vecmem::get_data(ctx.bin_pair_dphi), ctx.bin_pair_dphi_buf)->ignore();

    ctx.counters_buf = collection_types<unsigned int>::buffer(12, m_mr.main);
    m_copy.get().setup(ctx.counters_buf)->ignore();
    m_copy.get().memset(ctx.counters_buf, 0)->ignore();

    cudaStreamSynchronize(stream);

    // 2. Find edges between spacepoint pairs
    ctx.nMaxEdges = m_config.max_edges_factor * ctx.nNodes;
    ctx.edge_params_buf = collection_types<kernels::half4>::buffer(ctx.nMaxEdges, m_mr.main);
    m_copy.get().setup(ctx.edge_params_buf)->ignore();
    ctx.edge_nodes_buf = collection_types<int2>::buffer(ctx.nMaxEdges, m_mr.main);
    m_copy.get().setup(ctx.edge_nodes_buf)->ignore();
    ctx.num_incoming_edges_buf = collection_types<int>::buffer(ctx.nNodes + 1, m_mr.main);
    m_copy.get().setup(ctx.num_incoming_edges_buf)->ignore();
    m_copy.get().memset(ctx.num_incoming_edges_buf, 0)->ignore();

    nBlocks = ctx.nUsedBinPairs;
    nThreads = 128;

    kernels::graphEdgeMakingKernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.bin_pair_views_buf, ctx.bin_pair_dphi_buf, ctx.node_params_buf,
        ctx.d_graph_building_params, ctx.counters_buf, ctx.edge_nodes_buf,
        ctx.edge_params_buf, ctx.num_incoming_edges_buf, ctx.nMaxEdges,
        m_config.n_phi_bins);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    
    collection_types<unsigned int>::host h_counters(12, m_mr.host);
    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->wait();
    ctx.nEdges = h_counters[0];
    

    TRACCC_DEBUG("Created " << ctx.nEdges << " edges with a cap of "
                            << ctx.nMaxEdges);

    if (ctx.nEdges > ctx.nMaxEdges) {
        TRACCC_ERROR("Number of edges exceeds the maximum allowed, Removing "
                     << ctx.nEdges - ctx.nMaxEdges << " edges");
        ctx.nEdges = ctx.nMaxEdges;
    } else if (ctx.nEdges == 0) {
        return {0, m_mr.main};
    }

    collection_types<int>::host cusum(ctx.nNodes + 1, m_mr.host);

    m_copy.get()(vecmem::get_data(ctx.num_incoming_edges_buf), cusum)->ignore();

    cudaStreamSynchronize(stream);

    for (unsigned int k = 0; k < ctx.nNodes; k++){ // TODO: Maybe use a GPU version of a scan? or std version?
        cusum[k + 1] += cusum[k];
    }

    m_copy.get()(vecmem::get_data(cusum), ctx.num_incoming_edges_buf)->ignore();

    cudaStreamSynchronize(stream);


    // 3. link edges and nodes

    ctx.edge_links_buf = collection_types<int>::buffer(ctx.nEdges, m_mr.main);
    m_copy.get().setup(ctx.edge_links_buf)->ignore();

    nThreads = 256;
    nBlocks = 1 + (ctx.nEdges - 1) / nThreads;

    kernels::graphEdgeLinkingKernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.edge_nodes_buf, ctx.edge_links_buf, ctx.num_incoming_edges_buf,
        ctx.nEdges);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());


    // 4. edge matching to create edge-to-edge connections

    ctx.num_neighbours_buf = collection_types<unsigned char>::buffer(ctx.nEdges, m_mr.main);
    m_copy.get().setup(ctx.num_neighbours_buf)->ignore();
    m_copy.get().memset(ctx.num_neighbours_buf, 0)->ignore();

    ctx.reIndexer_buf = collection_types<int>::buffer(ctx.nEdges, m_mr.main);
    m_copy.get().setup(ctx.reIndexer_buf)->ignore();
    m_copy.get().memset(ctx.reIndexer_buf, 0xFF)->ignore();

    ctx.neighbours_buf = collection_types<int>::buffer(m_config.max_num_neighbours * ctx.nEdges, m_mr.main);
    m_copy.get().setup(ctx.neighbours_buf)->ignore();
    m_copy.get().memset(ctx.neighbours_buf, 0)->ignore();

    kernels::graphEdgeMatchingKernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.d_graph_building_params, ctx.edge_params_buf, ctx.edge_nodes_buf,
        ctx.num_incoming_edges_buf, ctx.edge_links_buf, ctx.num_neighbours_buf,
        ctx.neighbours_buf, ctx.reIndexer_buf, ctx.counters_buf, ctx.nEdges,
        m_config.max_num_neighbours);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // 5. Edge re-indexing to keep only edges involved in any connection

    kernels::edgeReIndexingKernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.reIndexer_buf, ctx.counters_buf, ctx.nEdges);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    collection_types<unsigned int>::host nStats(3, m_mr.host);

    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->ignore();
    nStats[0] = h_counters[0];
    nStats[1] = h_counters[1];
    nStats[2] = h_counters[2];

    ctx.nConnections = nStats[1];
    ctx.nConnectedEdges = nStats[2];

    TRACCC_DEBUG("created " << ctx.nConnections << " edge links, found "
                            << ctx.nConnectedEdges
                            << " connected edges for seed extraction");
    if (ctx.nConnectedEdges == 0)
        return {0, m_mr.main};

    unsigned int nIntsPerEdge = 2 + 1 + m_config.max_num_neighbours;

    ctx.output_graph_buf = collection_types<int>::buffer(ctx.nConnectedEdges * nIntsPerEdge, m_mr.main);
    m_copy.get().setup(ctx.output_graph_buf)->ignore();

    nThreads = 256;
    unsigned int nEdgesPerBlock = nThreads * 64;

    nBlocks = 1 + (ctx.nEdges - 1) / nEdgesPerBlock;

    kernels::graphCompressionKernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.node_index_buf, ctx.edge_nodes_buf, ctx.num_neighbours_buf,
        ctx.neighbours_buf, ctx.reIndexer_buf, ctx.output_graph_buf, nEdgesPerBlock,
        ctx.nEdges, m_config.max_num_neighbours);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // 6. Find longest segments with CCA

    ctx.active_edges_buf = collection_types<int>::buffer(ctx.nConnectedEdges, m_mr.main);
    m_copy.get().setup(ctx.active_edges_buf)->ignore();
    m_copy.get().memset(ctx.active_edges_buf, 0xFF)->ignore();

    ctx.levels_buf = collection_types<char>::buffer(2 * ctx.nConnectedEdges, m_mr.main);
    m_copy.get().setup(ctx.levels_buf)->ignore();
    m_copy.get().memset(ctx.levels_buf, 0x1)->ignore();

    ctx.outgoing_paths_buf = collection_types<short2>::buffer(ctx.nConnectedEdges, m_mr.main);
    m_copy.get().setup(ctx.outgoing_paths_buf)->ignore();

    unsigned int nEdgesLeft = ctx.nConnectedEdges;
    h_counters[3] = nEdgesLeft;
    m_copy.get()(vecmem::get_data(h_counters), ctx.counters_buf)->ignore(); // Only counters 3 is used to the device
    if (nEdgesLeft == 0)
        return {0, m_mr.main};

    cudaStreamSynchronize(stream);

    nThreads = 128;
    nBlocks = 1 + (nEdgesLeft - 1) / nThreads;
    for (int iter = 0; iter < traccc::device::gbts_consts::max_cca_iter;
         iter++) {
        kernels::CCA_IterationKernel<<<nBlocks, nThreads, 0, stream>>>(
            ctx.output_graph_buf, ctx.levels_buf, ctx.active_edges_buf,
            ctx.outgoing_paths_buf, ctx.counters_buf, iter, ctx.nConnectedEdges,
            m_config.max_num_neighbours, m_config.minLevel);

        cudaStreamSynchronize(stream);
    }

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());


    nThreads = 128;
    nBlocks = 1 + (ctx.nConnectedEdges - 1) / nThreads;

    cudaStreamSynchronize(stream);

    kernels::count_terminus_edges<<<nBlocks, nThreads, 0, stream>>>(
        ctx.outgoing_paths_buf, ctx.counters_buf,
        ctx.nConnectedEdges);

    cudaStreamSynchronize(stream);

    // nPaths to terminus, nTerminusEdges
    unsigned int path_sizes[2];
    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->wait();
    path_sizes[0] = h_counters[6];
    path_sizes[1] = h_counters[7];

    unsigned int pathsPerTerminus = 1 + (path_sizes[0] - 1) / path_sizes[1];

    TRACCC_DEBUG(path_sizes[0] << "size of path store | nTerminusEdges "
                               << path_sizes[1]);

    ctx.path_store_buf = collection_types<int2>::buffer(path_sizes[0] + path_sizes[1], m_mr.main);
    m_copy.get().setup(ctx.path_store_buf)->ignore();
    ctx.seed_proposals_buf = collection_types<int2>::buffer(path_sizes[0], m_mr.main);
    m_copy.get().setup(ctx.seed_proposals_buf)->ignore();
    ctx.seed_ambiguity_buf = collection_types<char>::buffer(path_sizes[0], m_mr.main);
    m_copy.get().setup(ctx.seed_ambiguity_buf)->ignore();

    ctx.edge_bids_buf = collection_types<unsigned long long int>::buffer(ctx.nConnectedEdges, m_mr.main);
    m_copy.get().setup(ctx.edge_bids_buf)->ignore();
    m_copy.get().memset(ctx.edge_bids_buf, 0)->ignore();

    nThreads = 128;
    kernels::add_terminus_to_path_store<<<nBlocks, nThreads, 0, stream>>>(
        ctx.path_store_buf, ctx.outgoing_paths_buf,
        ctx.nConnectedEdges);

    unsigned int terminusPerBlock = std::min(
        nThreads, 1 + (traccc::device::gbts_consts::live_path_buffer - 1) /
                          pathsPerTerminus);
    nBlocks = 1 + (path_sizes[1] - 1) / terminusPerBlock;
    
    kernels::fill_path_store<<<nBlocks, nThreads, 0, stream>>>(
        ctx.path_store_buf, ctx.output_graph_buf, ctx.levels_buf, ctx.counters_buf,
        path_sizes[1], terminusPerBlock, m_config.max_num_neighbours,
        path_sizes[0] + path_sizes[1]);

    nThreads = 128;
    nBlocks = 1 + (path_sizes[0] + path_sizes[1] - 1) / nThreads;

    kernels::fit_segments<<<nBlocks, nThreads, 0, stream>>>(
        ctx.reducedSP_buf, ctx.output_graph_buf, ctx.path_store_buf,
        ctx.seed_proposals_buf, ctx.edge_bids_buf, ctx.seed_ambiguity_buf,
        ctx.counters_buf, path_sizes[1], m_config.minLevel,
        m_config.max_num_neighbours, m_config.seed_extraction_params);

    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->wait(); // Only counters 8 is used
    unsigned int nProps = h_counters[8];

    TRACCC_DEBUG("nProps " << nProps);

    cudaStreamSynchronize(stream);

    if (nProps == 0) {
        return {0, m_mr.main};
    }

    nThreads = 128;
    nBlocks = 1 + (nProps - 1) / nThreads;

    kernels::reset_edge_bids<<<nBlocks, nThreads, 0, stream>>>(
        ctx.path_store_buf, ctx.seed_proposals_buf, ctx.edge_bids_buf,
        ctx.seed_ambiguity_buf, ctx.counters_buf, -1);

    for (int round = 0; round < 5; ++round) {

        m_copy.get().memset(ctx.edge_bids_buf, 0)->ignore();

        kernels::seeds_rebid_for_edges<<<nBlocks, nThreads, 0, stream>>>(
            ctx.path_store_buf, ctx.seed_proposals_buf, ctx.edge_bids_buf,
            ctx.seed_ambiguity_buf, nProps);

        kernels::reset_edge_bids<<<nBlocks, nThreads, 0, stream>>>(
            ctx.path_store_buf, ctx.seed_proposals_buf, ctx.edge_bids_buf,
            ctx.seed_ambiguity_buf, ctx.counters_buf, round);
    }
    
    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->ignore(); // Only counters 9 is used
    unsigned int nRejectedProps = h_counters[9];
    ctx.nSeeds = nProps - nRejectedProps;

    TRACCC_DEBUG("Rejecetd " << nRejectedProps << " out of " << nProps
                             << " seed proposals");


    // 8. convert to 3sp seeds and make output buffer

    edm::seed_collection::buffer output_seeds(
        ctx.nSeeds, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.get().setup(output_seeds)->ignore();

    kernels::gbts_seed_conversion_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.seed_proposals_buf, ctx.seed_ambiguity_buf, ctx.path_store_buf,
        ctx.output_graph_buf, output_seeds, nProps, m_config.max_num_neighbours);

    cudaStreamSynchronize(stream);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    TRACCC_DEBUG("GBTS found " << ctx.nSeeds << " seeds");
    return output_seeds;
}

}  // namespace traccc::cuda
