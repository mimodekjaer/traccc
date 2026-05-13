/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/add_terminus_to_path_store.hpp"
#include "traccc/gbts_seeding/device/bin_sp_by_layer.hpp"
#include "traccc/gbts_seeding/device/cca_iteration.hpp"
#include "traccc/gbts_seeding/device/count_sp_by_layer.hpp"
#include "traccc/gbts_seeding/device/count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/edge_re_indexing.hpp"
#include "traccc/gbts_seeding/device/eta_phi_counting.hpp"
#include "traccc/gbts_seeding/device/eta_phi_histo.hpp"
#include "traccc/gbts_seeding/device/eta_phi_prefix_sum.hpp"
#include "traccc/gbts_seeding/device/fill_path_store.hpp"
#include "traccc/gbts_seeding/device/fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_conversion.hpp"
#include "traccc/gbts_seeding/device/graph_compression.hpp"
#include "traccc/gbts_seeding/device/graph_edge_linking.hpp"
#include "traccc/gbts_seeding/device/graph_edge_making.hpp"
#include "traccc/gbts_seeding/device/graph_edge_matching.hpp"
#include "traccc/gbts_seeding/device/minmax_rad.hpp"
#include "traccc/gbts_seeding/device/node_eta_binning.hpp"
#include "traccc/gbts_seeding/device/node_sorting.hpp"
#include "traccc/gbts_seeding/device/reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/seeds_bid_for_hits.hpp"
#include "traccc/gbts_seeding/device/seeds_rebid_for_edges.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::cuda {

using float4 = traccc::float4;
using float2 = traccc::float2;
using int2 = traccc::int2;
using short2 = traccc::short2;

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
    collection_types<std::pair<unsigned int, unsigned int>>::buffer
        surfaceToLayerMap_buf;  // surface_index, layerBin
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
    collection_types<int2>::buffer
        seed_proposals_buf;  // int quality and final mini_state_idx
    // first 32 bits are seed quality second 32 bits are seed_proposals index
    collection_types<unsigned long long int>::buffer edge_bids_buf;
    collection_types<unsigned long long int>::buffer hit_bids_buf;
    // 0 as default/is real seed, 1 as maybe seed,
    //-1 as maybe fake seed, -2 as fake
    collection_types<char>::buffer seed_ambiguity_buf;
};

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    vecmem::copy& copy, cuda::stream& str,
    std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      cuda::algorithm_base{str} {}

void gbts_seeding_algorithm::sync() const {
    stream().synchronize();
}

void gbts_seeding_algorithm::count_sp_by_layer_kernel(
    const count_sp_by_layer_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::count_sp_by_layer<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.spacepoints, payload.measurements, payload.volumeToLayerMap,
        payload.surfaceToLayerMap, payload.layerType, payload.reducedSP,
        payload.layerCounts, payload.spacepointsLayer, payload.type1_max_width,
        payload.nSp, payload.volumeMapSize, payload.surfaceMapSize,
        payload.doTauCut);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::bin_sp_by_layer_kernel(
    const bin_sp_by_layer_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::bin_sp_by_layer<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(
        payload.sp_params, payload.reducedSP, payload.layerCounts,
        payload.spacepointsLayer, payload.original_sp_idx, payload.nSp);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::node_eta_binning_kernel(
    const node_eta_binning_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = payload.nLayers;
    kernels::node_eta_binning<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(
        payload.sp_params, payload.layer_info, payload.layer_geo,
        payload.node_eta_index, payload.layerCounts, payload.nLayers);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::eta_phi_histo_kernel(
    const eta_phi_histo_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nNodes - 1) / n_threads;
    kernels::eta_phi_histo<<<n_blocks, n_threads, 0,
                             details::get_stream(stream())>>>(
        payload.node_phi_index, payload.node_eta_index, payload.eta_phi_histo,
        payload.sp_params, payload.nNodes, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::eta_phi_counting_kernel(
    const eta_phi_counting_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::eta_phi_counting<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(
        payload.eta_phi_histo, payload.eta_node_counter, payload.phi_cusums,
        payload.nEtaBins, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::eta_phi_prefix_sum_kernel(
    const eta_phi_prefix_sum_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::eta_phi_prefix_sum<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(
        payload.eta_node_counter, payload.phi_cusums, payload.nEtaBins,
        payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::node_sorting_kernel(
    const node_sorting_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nNodes - 1) / n_threads;
    kernels::node_sorting<<<n_blocks, n_threads, 0,
                            details::get_stream(stream())>>>(
        payload.sp_params, payload.node_eta_index, payload.node_phi_index,
        payload.phi_cusums, payload.node_params, payload.node_index,
        payload.original_sp_idx, payload.graph_building_params, payload.nNodes,
        payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::minmax_rad_kernel(
    const minmax_rad_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::minmax_rad<<<n_blocks, n_threads, 0,
                          details::get_stream(stream())>>>(
        payload.eta_bin_views, payload.node_params, payload.bin_rads,
        payload.nEtaBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::graph_edge_making_kernel(
    const graph_edge_making_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = payload.nUsedBinPairs;
    kernels::graph_edge_making<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.bin_pair_views, payload.bin_pair_dphi, payload.node_params,
        payload.graph_building_params, &payload.nEdgesCounter,
        payload.edge_nodes, payload.edge_params, payload.num_outgoing_edges,
        payload.nMaxEdges, payload.nPhiBins);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::graph_edge_linking_kernel(
    const graph_edge_linking_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_edge_linking<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(
        payload.edge_nodes, payload.edge_links, payload.num_outgoing_edges,
        payload.nEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::graph_edge_matching_kernel(
    const graph_edge_matching_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_edge_matching<<<n_blocks, n_threads, 0,
                                   details::get_stream(stream())>>>(
        payload.graph_building_params, payload.edge_params, payload.edge_nodes,
        payload.num_outgoing_edges, payload.edge_links, payload.num_neighbours,
        payload.neighbours, payload.reIndexer, &payload.nConnectionsCounter,
        payload.nEdges, payload.nMaxNei);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::edge_re_indexing_kernel(
    const edge_re_indexing_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::edge_re_indexing<<<n_blocks, n_threads, 0,
                                details::get_stream(stream())>>>(
        payload.reIndexer, &payload.nConnectedEdgesCounter, payload.nEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::graph_compression_kernel(
    const graph_compression_kernel_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::graph_compression<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(
        payload.orig_node_index, payload.edge_nodes, payload.num_neighbours,
        payload.neighbours, payload.reIndexer, payload.output_graph,
        payload.nEdges, payload.nMaxNei);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::cca_iteration_kernel(
    const cca_iteration_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::cca_iteration<<<n_blocks, n_threads, 0,
                             details::get_stream(stream())>>>(
        payload.output_graph, payload.levels, payload.active_edges,
        payload.outgoing_paths, &payload.nActiveEdgesA, &payload.nActiveEdgesB,
        &payload.nCCABlockCounter, payload.iter, payload.nConnectedEdges,
        payload.max_num_neighbours, payload.minLevel);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::count_terminus_edges_kernel(
    const count_terminus_edges_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::count_terminus_edges<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(
        payload.outgoing_paths, &payload.nPathsCounter,
        &payload.nPathStoreSizeCounter, payload.nConnectedEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::add_terminus_to_path_store_kernel(
    const add_terminus_to_path_store_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::add_terminus_to_path_store<<<n_blocks, n_threads, 0,
                                          details::get_stream(stream())>>>(
        payload.path_store, payload.outgoing_paths, payload.nConnectedEdges);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::fill_path_store_kernel(
    const fill_path_store_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks =
        1 + (payload.nTerminusEdges - 1) / payload.nTerminusPerBlock;
    kernels::fill_path_store<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(
        payload.path_store, payload.output_graph, payload.levels,
        &payload.nPathStoreSizeCounter, payload.nTerminusEdges,
        payload.nTerminusPerBlock, payload.max_num_neighbours, payload.nPaths);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::fit_segments_kernel(
    const fit_segments_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nPaths - 1) / n_threads;
    kernels::fit_segments<<<n_blocks, n_threads, 0,
                            details::get_stream(stream())>>>(
        payload.reducedSP, payload.output_graph, payload.path_store,
        payload.seed_proposals, payload.edge_bids, payload.seed_ambiguity,
        &payload.nPathStoreSize, &payload.nPropsCounter, payload.nTerminusEdges,
        payload.minLevel, payload.max_num_neighbours,
        payload.seed_extraction_params);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::reset_edge_bids_kernel(
    const reset_edge_bids_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::reset_edge_bids<<<n_blocks, n_threads, 0,
                               details::get_stream(stream())>>>(
        payload.path_store, payload.seed_proposals, payload.edge_bids,
        payload.seed_ambiguity, payload.nProps, &payload.nRejectedPropsCounter,
        payload.round);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::seeds_rebid_for_edges_kernel(
    const seeds_rebid_for_edges_kernel_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::seeds_rebid_for_edges<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(
        payload.path_store, payload.seed_proposals, payload.edge_bids,
        payload.seed_ambiguity, payload.nProps);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    nThreads = 128;
    nBlocks = 1 + (ctx.nConnectedEdges - 1) / nThreads;

    m_stream.get().synchronize();

    kernels::count_terminus_edges<<<nBlocks, nThreads, 0, stream>>>(
        ctx.outgoing_paths_buf, ctx.counters_buf, ctx.nConnectedEdges);

    m_stream.get().synchronize();

    // nPaths to terminus, nTerminusEdges
    unsigned int path_sizes[2];
    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)->wait();
    path_sizes[0] = h_counters[6];
    path_sizes[1] = h_counters[7];

    unsigned int pathsPerTerminus = 1 + (path_sizes[0] - 1) / path_sizes[1];

    TRACCC_DEBUG(path_sizes[0] << "size of path store | nTerminusEdges "
                               << path_sizes[1]);

    ctx.path_store_buf = collection_types<int2>::buffer(
        path_sizes[0] + path_sizes[1], m_mr.main);
    m_copy.get().setup(ctx.path_store_buf)->ignore();
    ctx.seed_proposals_buf =
        collection_types<int2>::buffer(path_sizes[0], m_mr.main);
    m_copy.get().setup(ctx.seed_proposals_buf)->ignore();
    ctx.seed_ambiguity_buf =
        collection_types<char>::buffer(path_sizes[0], m_mr.main);
    m_copy.get().setup(ctx.seed_ambiguity_buf)->ignore();

    ctx.edge_bids_buf = collection_types<unsigned long long int>::buffer(
        ctx.nConnectedEdges, m_mr.main);
    m_copy.get().setup(ctx.edge_bids_buf)->ignore();
    m_copy.get().memset(ctx.edge_bids_buf, 0)->ignore();

    nThreads = 128;
    kernels::add_terminus_to_path_store<<<nBlocks, nThreads, 0, stream>>>(
        ctx.path_store_buf, ctx.outgoing_paths_buf, ctx.nConnectedEdges);

    unsigned int terminusPerBlock = std::min(
        nThreads, 1 + (traccc::device::gbts_consts::live_path_buffer - 1) /
                          pathsPerTerminus);
    nBlocks = 1 + (path_sizes[1] - 1) / terminusPerBlock;

    kernels::fill_path_store<<<nBlocks, nThreads, 0, stream>>>(
        ctx.path_store_buf, ctx.output_graph_buf, ctx.levels_buf,
        ctx.counters_buf, path_sizes[1], terminusPerBlock,
        m_config.max_num_neighbours, path_sizes[0] + path_sizes[1]);

    nThreads = 128;
    nBlocks = 1 + (path_sizes[0] + path_sizes[1] - 1) / nThreads;

    kernels::fit_segments<<<nBlocks, nThreads, 0, stream>>>(
        ctx.reducedSP_buf, ctx.output_graph_buf, ctx.path_store_buf,
        ctx.seed_proposals_buf, ctx.edge_bids_buf, ctx.seed_ambiguity_buf,
        ctx.counters_buf, path_sizes[1], m_config.minLevel,
        m_config.max_num_neighbours, m_config.seed_extraction_params);

    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)
        ->wait();  // Only counters 8 is used
    unsigned int nProps = h_counters[8];

    TRACCC_DEBUG("nProps " << nProps);

    m_stream.get().synchronize();

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

    m_copy.get()(vecmem::get_data(ctx.counters_buf), h_counters)
        ->ignore();  // Only counters 9 is used
    unsigned int nRejectedProps = h_counters[9];
    ctx.nSeeds = nProps - nRejectedProps;

    TRACCC_DEBUG("Rejected " << nRejectedProps << " out of " << nProps
                             << " seed proposals");

    // 8. convert to 3sp seeds and make output buffer
    // allocate extra seed space for hit permutation
    edm::seed_collection::buffer output_seeds(
        2 * ctx.nSeeds, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.get().setup(output_seeds)->wait();

    ctx.hit_bids_buf =
        collection_types<unsigned long long int>::buffer(ctx.nSp, m_mr.main);
    m_copy.get().setup(ctx.hit_bids_buf)->ignore();
    m_copy.get().memset(ctx.hit_bids_buf, 0)->wait();

    nThreads = 128;
    nBlocks = 1 + (ctx.nSeeds - 1) / nThreads;

    kernels::seeds_bid_for_hits<<<nBlocks, nThreads, 0, stream>>>(
        ctx.output_graph_buf, ctx.seed_proposals_buf, ctx.path_store_buf,
        ctx.seed_ambiguity_buf, ctx.hit_bids_buf, nProps,
        1 + 2 + m_config.max_num_neighbours);

    kernels::gbts_seed_conversion_kernel<<<nBlocks, nThreads, 0, stream>>>(
        ctx.seed_proposals_buf, ctx.seed_ambiguity_buf, ctx.path_store_buf,
        ctx.output_graph_buf, ctx.reducedSP_buf, output_seeds, ctx.hit_bids_buf,
        nProps, m_config.max_num_neighbours,
        m_config.seed_ambi_params.dropout_dcurv_m,
        m_config.seed_ambi_params.force_dropout_max_curv_m,
        m_config.seed_ambi_params.best_hit_frac,
        m_config.seed_ambi_params.tight_bid_cot_threshold,
        m_config.seed_ambi_params.use_dropout);

    m_stream.get().synchronize();

    ctx.nSeeds = m_copy.get().get_size(output_seeds);

    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
