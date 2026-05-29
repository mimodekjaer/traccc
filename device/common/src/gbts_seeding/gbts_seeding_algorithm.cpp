/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"

#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace traccc::device {

struct gbts_seeding_algorithm::data {
    gbts_seedfinder_config m_config;
};  // struct gbts_seeding_algorithm::data

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      algorithm_base{mr, copy},
      m_data{std::make_unique<data>(cfg)} {}

gbts_seeding_algorithm::~gbts_seeding_algorithm() = default;

auto gbts_seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements) const
    -> output_type {

    assert(m_data);

    const gbts_seedfinder_config& cfg = m_data->m_config;

    // 0. Bin spacepoints by the mapping supplied to config.surfaceToLayerMap.
    const unsigned int nSp = copy().get_size(spacepoints);
    TRACCC_DEBUG("nSp " << nSp);
    if (nSp == 0) {
        TRACCC_WARNING("No spacepoints were found in the event");
        return {0, mr().main};
    }

    collection_types<unsigned int>::buffer layerCounts_buf(cfg.nLayers + 1, mr().main);
    copy().memset(layerCounts_buf, 0)->ignore();

    collection_types<float4>::buffer reducedSP_buf(nSp, mr().main);
    copy().setup(reducedSP_buf)->ignore();

    collection_types<short>::buffer spacepointsLayer_buf(nSp, mr().main);
    copy().setup(spacepointsLayer_buf)->ignore();

    collection_types<short>::buffer volumeToLayerMap_buf(
        static_cast<unsigned int>(cfg.volumeToLayerMap.size()), mr().main);
    copy().setup(volumeToLayerMap_buf)->ignore();
    copy()(vecmem::get_data(cfg.volumeToLayerMap), volumeToLayerMap_buf)
        ->ignore();

    collection_types<std::pair<unsigned int, unsigned int>>::buffer
        surfaceToLayerMap_buf;
    if (!cfg.surfaceToLayerMap.empty()) {
        surfaceToLayerMap_buf =
            collection_types<std::pair<unsigned int, unsigned int>>::buffer(
                static_cast<unsigned int>(cfg.surfaceToLayerMap.size()),
                mr().main);
        copy().setup(surfaceToLayerMap_buf)->ignore();
        copy()(vecmem::get_data(cfg.surfaceToLayerMap), surfaceToLayerMap_buf)
            ->ignore();
    }

    collection_types<char>::buffer layerType_buf(cfg.nLayers, mr().main);
    copy().setup(layerType_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.type), layerType_buf)->ignore();
    count_sp_by_layer_kernel({nSp, spacepoints, measurements,
                              volumeToLayerMap_buf, surfaceToLayerMap_buf,
                              layerType_buf, reducedSP_buf, layerCounts_buf,
                              spacepointsLayer_buf,
                              cfg.volumeToLayerMap.size(), cfg.surfaceToLayerMap.size(),
                              cfg.sp_counting_params});
    sync();

    // Prefix-sum layerCounts on host.
    collection_types<unsigned int>::host layerCounts(cfg.nLayers + 1, mr().host);
    copy()(vecmem::get_data(layerCounts_buf), layerCounts)->wait();
    for (unsigned int layer = 0; layer < cfg.nLayers; layer++) {
        layerCounts[layer + 1] += layerCounts[layer];
    }
    copy()(vecmem::get_data(layerCounts), layerCounts_buf)->wait();

    const unsigned int nNodes =
        static_cast<unsigned int>(layerCounts[cfg.nLayers]);
    TRACCC_DEBUG("nNodes " << nNodes);
    if (nNodes == 0) {
        TRACCC_WARNING("No nodes were found after spacepoint counting");
        return {0, mr().main};
    }

    collection_types<float4>::buffer sp_params_buf(nSp, mr().main);
    copy().setup(sp_params_buf)->ignore();
    collection_types<unsigned int>::buffer original_sp_idx_buf(nSp, mr().main);
    copy().setup(original_sp_idx_buf)->ignore();

    bin_sp_by_layer_kernel({nSp, sp_params_buf, reducedSP_buf, layerCounts_buf,
                            spacepointsLayer_buf, original_sp_idx_buf});
    sync();

    // 1. Histogram spacepoints by layer->eta->phi and convert to nodes.
    collection_types<std::pair<unsigned int, unsigned int>>::buffer layer_info_buf(cfg.nLayers,
                                                                 mr().main);
    copy().setup(layer_info_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.info), layer_info_buf)->ignore();

    collection_types<std::pair<float, float>>::buffer layer_geo_buf(
        cfg.nLayers, mr().main);
    copy().setup(layer_geo_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.geo), layer_geo_buf)->ignore();

    collection_types<unsigned int>::buffer node_phi_index_buf(nNodes, mr().main);
    copy().setup(node_phi_index_buf)->ignore();

    collection_types<unsigned int>::buffer node_eta_index_buf(nNodes, mr().main);
    copy().setup(node_eta_index_buf)->ignore();

    node_eta_binning_kernel({cfg.nLayers, sp_params_buf, layer_info_buf,
                             layer_geo_buf, node_eta_index_buf,
                             layerCounts_buf});
    sync();

    const unsigned int hist_size = cfg.n_eta_bins * cfg.n_phi_bins;
    collection_types<unsigned int>::buffer eta_phi_histo_buf(hist_size, mr().main);
    copy().setup(eta_phi_histo_buf)->ignore();
    copy().memset(eta_phi_histo_buf, 0)->ignore();
    collection_types<unsigned int>::buffer phi_cusums_buf(hist_size, mr().main);
    copy().setup(phi_cusums_buf)->ignore();

    eta_phi_histo_kernel({nNodes, cfg.n_phi_bins, node_phi_index_buf,
                          node_eta_index_buf, eta_phi_histo_buf,
                          sp_params_buf});
    sync();

    collection_types<unsigned int>::buffer eta_node_counter_buf(cfg.n_eta_bins,
                                                               mr().main);
    copy().setup(eta_node_counter_buf)->ignore();

    eta_phi_counting_kernel({cfg.n_eta_bins, cfg.n_phi_bins, eta_phi_histo_buf,
                             eta_node_counter_buf, phi_cusums_buf});
    sync();

    collection_types<unsigned int>::host eta_sums(cfg.n_eta_bins, mr().host);
    copy()(vecmem::get_data(eta_node_counter_buf), eta_sums)->wait();
    for (unsigned int k = 0; k < cfg.n_eta_bins; k++) {
        eta_sums[k + 1] += eta_sums[k];
    }
    copy()(vecmem::get_data(eta_sums), eta_node_counter_buf)->wait();

    collection_types<unsigned int>::host eta_bin_views(2 * cfg.n_eta_bins, mr().host);
    for (unsigned int view_idx = 0; view_idx < cfg.n_eta_bins; view_idx++) {
        const unsigned int pos = 2 * view_idx;
        eta_bin_views[pos] = (view_idx == 0) ? 0 : eta_sums[view_idx - 1];
        eta_bin_views[pos + 1] = eta_sums[view_idx];
    }

    eta_phi_prefix_sum_kernel({cfg.n_eta_bins, cfg.n_phi_bins,
                               eta_node_counter_buf, phi_cusums_buf});
    sync();

    collection_types<float>::buffer node_params_buf(5 * nNodes, mr().main);
    copy().setup(node_params_buf)->ignore();
    collection_types<unsigned int>::buffer node_index_buf(nNodes, mr().main);
    copy().setup(node_index_buf)->ignore();

    node_sorting_kernel({nNodes, cfg.n_phi_bins, sp_params_buf,
                         node_eta_index_buf, node_phi_index_buf, phi_cusums_buf,
                         node_params_buf, node_index_buf, original_sp_idx_buf,
                         cfg.node_sorting});
    sync();

    collection_types<unsigned int>::buffer eta_bin_views_buf(2 * cfg.n_eta_bins,
                                                    mr().main);
    copy().setup(eta_bin_views_buf)->ignore();
    copy()(vecmem::get_data(eta_bin_views), eta_bin_views_buf)->wait();

    collection_types<float>::buffer bin_rads_buf(2 * cfg.n_eta_bins, mr().main);
    copy().setup(bin_rads_buf)->ignore();

    minmax_rad_kernel({cfg.n_eta_bins, eta_bin_views_buf, node_params_buf,
                       bin_rads_buf});
    sync();

    collection_types<float>::host bin_rads(2 * cfg.n_eta_bins, mr().host);
    copy()(vecmem::get_data(bin_rads_buf), bin_rads)->wait();

    // 2. Prepare input for the graph-making part of the code.
    int int_nBinPairs = 0;
    for (const std::pair<unsigned int, unsigned int>& binPair : cfg.binTables) {
        const int bin1_begin = eta_bin_views[2 * binPair.first];
        const int bin1_end = eta_bin_views[2 * binPair.first + 1];
        int nNodesInBin1 = bin1_end - bin1_begin;
        if (bin1_begin > bin1_end) {
            nNodesInBin1 = bin1_begin - bin1_end;
        }
        int_nBinPairs +=
            1 + (nNodesInBin1 - 1) / gbts_consts::node_buffer_length;
    }
    const unsigned int nBinPairs = static_cast<unsigned int>(int_nBinPairs);

    collection_types<unsigned int>::host bin_pair_views(4 * nBinPairs, mr().host);
    collection_types<float>::host bin_pair_dphi(nBinPairs, mr().host);

    unsigned int pairIdx = 0;
    for (const std::pair<unsigned int, unsigned int>& binPair : cfg.binTables) {
        const float rb1 = bin_rads[2 * binPair.first];

        const unsigned int begin_bin1 = eta_bin_views[2 * binPair.first];
        const unsigned int end_bin1 = eta_bin_views[2 * binPair.first + 1];
        if (begin_bin1 == end_bin1) {
            continue;
        }
        if (eta_bin_views[2 * binPair.second] ==
            eta_bin_views[2 * binPair.second + 1]) {
            continue;
        }

        const float rb2 = bin_rads[2 * binPair.second + 1];
        const float maxDeltaR = std::fabs(rb2 - rb1);

        float deltaPhi = cfg.dphi_window.min_delta_phi +
                         cfg.dphi_window.dphi_coeff * maxDeltaR;
        if (maxDeltaR < 60) {
            deltaPhi =
                cfg.dphi_window.min_delta_phi_low_dr +
                cfg.dphi_window.dphi_coeff_low_dr *
                    maxDeltaR;
        }

        unsigned int currBegin_bin1 = begin_bin1;
        unsigned int currEnd_bin1 =
            end_bin1 < gbts_consts::node_buffer_length
                ? end_bin1
                : begin_bin1 + gbts_consts::node_buffer_length;

        for (; currEnd_bin1 < end_bin1;
             currEnd_bin1 += gbts_consts::node_buffer_length, pairIdx++) {
            const unsigned int offset = 4 * pairIdx;
            bin_pair_views[offset] = currBegin_bin1;
            bin_pair_views[1 + offset] = currEnd_bin1;
            bin_pair_views[2 + offset] = eta_bin_views[2 * binPair.second];
            bin_pair_views[3 + offset] = eta_bin_views[2 * binPair.second + 1];
            bin_pair_dphi[pairIdx] = deltaPhi;
            currBegin_bin1 = currEnd_bin1;
        }
        currEnd_bin1 = end_bin1;

        const unsigned int offset = 4 * pairIdx;
        bin_pair_views[offset] = currBegin_bin1;
        bin_pair_views[1 + offset] = currEnd_bin1;
        bin_pair_views[2 + offset] = eta_bin_views[2 * binPair.second];
        bin_pair_views[3 + offset] = eta_bin_views[2 * binPair.second + 1];
        bin_pair_dphi[pairIdx] = deltaPhi;
        pairIdx++;
    }
    const unsigned int nUsedBinPairs = pairIdx;
    TRACCC_DEBUG("nUsedBinPairs " << nUsedBinPairs);
    if (nUsedBinPairs == 0) {
        TRACCC_WARNING("No bin pairs were used for edge finding");
        return {0, mr().main};
    }

    collection_types<unsigned int>::buffer bin_pair_views_buf(4 * nUsedBinPairs,
                                                     mr().main);
    copy().setup(bin_pair_views_buf)->ignore();
    copy()(vecmem::get_data(bin_pair_views), bin_pair_views_buf)->ignore();

    collection_types<float>::buffer bin_pair_dphi_buf(nUsedBinPairs, mr().main);
    copy().setup(bin_pair_dphi_buf)->ignore();
    copy()(vecmem::get_data(bin_pair_dphi), bin_pair_dphi_buf)->ignore();

    // Named indices into the flat counter buffer.  Each counter is documented
    // at the point of declaration so that it is clear which kernel writes/reads
    // which slot.
    enum gbts_counter_idx : unsigned int {
        // Number of edges produced by device::graph_edge_making.
        k_nEdges = 0,
        // Number of edge-to-edge connections accumulated by
        // device::graph_edge_matching.
        k_nConnections,
        // Number of edges that survive the re-indexing step in
        // device::edge_re_indexing (i.e. edges with at least one connection).
        k_nConnectedEdges,
        // Total number of paths reachable from any terminus edge, accumulated
        // by device::count_terminus_edges.
        k_nPaths,
        // Running size of the path store.  Initialised to the number of
        // terminus edges by device::count_terminus_edges and grown further by
        // device::fill_path_store; device::fit_segments reads it as the upper
        // bound on path indices.
        k_nPathStoreSize,
        // Number of seed proposals produced by device::fit_segments.
        k_nProps,
        // Number of seed proposals rejected by device::reset_edge_bids.
        k_nRejectedProps,
        // Total counter slots.
        k_nCounters
    };

    // 10 named counters held in a single device buffer; one memset zeros all.
    collection_types<unsigned int>::buffer counters_buf(k_nCounters, mr().main);
    copy().setup(counters_buf)->ignore();
    copy().memset(counters_buf, 0)->ignore();
    // Raw device pointer used to bind references to individual counter slots.
    unsigned int* d_counters = counters_buf.ptr();
    // Host-side mirror, pulled in one transfer at each checkpoint.
    vecmem::vector<unsigned int> h_counters(
        k_nCounters, mr().host ? mr().host : &(mr().main));

    // 2. Find edges between spacepoint pairs.
    const unsigned int nMaxEdges = cfg.max_edges_factor * nNodes;
    // SoA edge-parameter buffers: each per-edge field lives in its own float
    // buffer so the matcher can early-exit a cut without paying for the
    // unread fields. Same total bytes as the prior packed float4 buffer.
    collection_types<float>::buffer edge_exp_eta_buf(nMaxEdges, mr().main);
    copy().setup(edge_exp_eta_buf)->ignore();
    collection_types<float>::buffer edge_curv_buf(nMaxEdges, mr().main);
    copy().setup(edge_curv_buf)->ignore();
    collection_types<float>::buffer edge_phi_z_buf(nMaxEdges, mr().main);
    copy().setup(edge_phi_z_buf)->ignore();
    collection_types<float>::buffer edge_phi_w_buf(nMaxEdges, mr().main);
    copy().setup(edge_phi_w_buf)->ignore();
    collection_types<uint2>::buffer edge_nodes_buf(nMaxEdges, mr().main);
    copy().setup(edge_nodes_buf)->ignore();
    collection_types<unsigned int>::buffer num_incoming_edges_buf(nNodes + 1, mr().main);
    copy().setup(num_incoming_edges_buf)->ignore();
    copy().memset(num_incoming_edges_buf, 0)->ignore();

    graph_edge_making_kernel({.nUsedBinPairs = nUsedBinPairs,
                              .nMaxEdges = nMaxEdges,
                              .nPhiBins = cfg.n_phi_bins,
                              .bin_pair_views = bin_pair_views_buf,
                              .bin_pair_dphi = bin_pair_dphi_buf,
                              .node_params = node_params_buf,
                              .edge_making_params = cfg.edge_making,
                              .nEdgesCounter = d_counters + k_nEdges,
                              .edge_nodes = edge_nodes_buf,
                              .edge_exp_eta = edge_exp_eta_buf,
                              .edge_curv = edge_curv_buf,
                              .edge_phi_z = edge_phi_z_buf,
                              .edge_phi_w = edge_phi_w_buf,
                              .num_outgoing_edges = num_incoming_edges_buf});
    sync();

    // Read back the number of edges produced (and the rest of the counters in
    // the same transfer).
    copy()(counters_buf, h_counters)->wait();

    unsigned int nEdges = h_counters[k_nEdges];
    TRACCC_DEBUG("Created " << nEdges << " edges with a cap of " << nMaxEdges);
    if (nEdges > nMaxEdges) {
        TRACCC_ERROR("Number of edges exceeds the maximum allowed, Removing "
                     << nEdges - nMaxEdges << " edges");
        nEdges = nMaxEdges;
    } else if (nEdges == 0) {
        return {0, mr().main};
    }

    collection_types<unsigned int>::host cusum(nNodes + 1, mr().host);
    copy()(vecmem::get_data(num_incoming_edges_buf), cusum)->wait();
    for (unsigned int k = 0; k < nNodes; k++) {
        cusum[k + 1] += cusum[k];
    }
    copy()(vecmem::get_data(cusum), num_incoming_edges_buf)->wait();

    // 3. Link edges and nodes.
    collection_types<unsigned int>::buffer edge_links_buf(nEdges, mr().main);
    copy().setup(edge_links_buf)->ignore();

    graph_edge_linking_kernel({nEdges, edge_nodes_buf, edge_links_buf,
                               num_incoming_edges_buf});
    sync();

    // 4. Edge matching to create edge-to-edge connections.
    collection_types<unsigned char>::buffer num_neighbours_buf(nEdges,
                                                               mr().main);
    copy().setup(num_neighbours_buf)->ignore();
    copy().memset(num_neighbours_buf, 0)->ignore();

    collection_types<unsigned int>::buffer reIndexer_buf(nEdges, mr().main);
    copy().setup(reIndexer_buf)->ignore();
    copy().memset(reIndexer_buf, UINT8_MAX)->ignore();

    collection_types<unsigned int>::buffer neighbours_buf(
        cfg.max_num_neighbours * nEdges, mr().main);
    copy().setup(neighbours_buf)->ignore();
    copy().memset(neighbours_buf, 0)->ignore();

    graph_edge_matching_kernel({.nEdges = nEdges,
                                .nMaxNei = cfg.max_num_neighbours,
                                .edge_matching_params = cfg.edge_matching,
                                .edge_exp_eta = edge_exp_eta_buf,
                                .edge_curv = edge_curv_buf,
                                .edge_phi_z = edge_phi_z_buf,
                                .edge_phi_w = edge_phi_w_buf,
                                .edge_nodes = edge_nodes_buf,
                                .num_outgoing_edges = num_incoming_edges_buf,
                                .edge_links = edge_links_buf,
                                .num_neighbours = num_neighbours_buf,
                                .neighbours = neighbours_buf,
                                .reIndexer = reIndexer_buf,
                                .nConnectionsCounter =
                                    d_counters + k_nConnections});
    sync();

    // 5. Edge re-indexing to keep only edges involved in any connection.
    edge_re_indexing_kernel(
        {nEdges, reIndexer_buf, d_counters + k_nConnectedEdges});
    sync();

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nConnections = h_counters[k_nConnections];
    const unsigned int nConnectedEdges = h_counters[k_nConnectedEdges];
    TRACCC_DEBUG("created " << nConnections << " edge links, found "
                            << nConnectedEdges
                            << " connected edges for seed extraction");
    if (nConnectedEdges == 0) {
        TRACCC_WARNING("No connected edges were found");
        return {0, mr().main};
    }

    const unsigned int nIntsPerEdge = 2 + 1 + cfg.max_num_neighbours;
    collection_types<unsigned int>::buffer output_graph_buf(
        nConnectedEdges * nIntsPerEdge, mr().main);
    copy().setup(output_graph_buf)->ignore();

    graph_compression_kernel({nEdges, cfg.max_num_neighbours, nConnectedEdges,
                              node_index_buf, edge_nodes_buf,
                              num_neighbours_buf, neighbours_buf, reIndexer_buf,
                              output_graph_buf});
    sync();

    // 6. Find longest segments with CCA.
    // active_edges is the per-edge "next iter index" flag: holds `iter` while
    // the edge is active in iteration `iter`, and 0xFF once it settles. A
    // single byte is sufficient because max_cca_iter <= 15.
    collection_types<unsigned char>::buffer active_edges_buf(nConnectedEdges,
                                                             mr().main);
    copy().setup(active_edges_buf)->ignore();
    copy().memset(active_edges_buf, 0xFF)->ignore();

    collection_types<unsigned char>::buffer levels_buf(2 * nConnectedEdges, mr().main);
    copy().setup(levels_buf)->ignore();
    copy().memset(levels_buf, 0x1)->ignore();

    collection_types<short2>::buffer outgoing_paths_buf(nConnectedEdges,
                                                        mr().main);
    copy().setup(outgoing_paths_buf)->wait();

    // Seed the CCA "load" counter (A) with the initial number of active edges.
    // Only this one slot is written; the rest of the device counters keep the
    // values they already had.
    for (char unsigned iter = 0;
         iter < traccc::device::gbts_consts::max_cca_iter; ++iter) {
        cca_iteration_kernel({nConnectedEdges, cfg.max_num_neighbours,
                                cfg.minLevel, output_graph_buf, levels_buf,
                                active_edges_buf, outgoing_paths_buf, iter});
        sync();
    }

    count_terminus_edges_kernel({nConnectedEdges, outgoing_paths_buf,
                                 d_counters + k_nPaths,
                                 d_counters + k_nPathStoreSize});
    sync();

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nPaths = h_counters[k_nPaths];
    const unsigned int nTerminusEdges = h_counters[k_nPathStoreSize];
    if (nTerminusEdges == 0) {
        TRACCC_WARNING("No terminus edges were found");
        return {0, mr().main};
    }
    const unsigned int pathsPerTerminus = 1 + (nPaths - 1) / nTerminusEdges;

    TRACCC_DEBUG(nPaths << " size of path store | nTerminusEdges "
                        << nTerminusEdges);

    collection_types<int2>::buffer path_store_buf(
        nPaths + nTerminusEdges, mr().main);
    copy().setup(path_store_buf)->ignore();
    collection_types<int2>::buffer seed_proposals_buf(nPaths, mr().main);
    copy().setup(seed_proposals_buf)->ignore();
    collection_types<char>::buffer seed_ambiguity_buf(nPaths, mr().main);
    copy().setup(seed_ambiguity_buf)->ignore();

    collection_types<unsigned long long int>::buffer edge_bids_buf(
        nConnectedEdges, mr().main);
    copy().setup(edge_bids_buf)->ignore();
    copy().memset(edge_bids_buf, 0)->ignore();

    add_terminus_to_path_store_kernel(
        {nConnectedEdges, path_store_buf, outgoing_paths_buf});

    constexpr unsigned int fill_block_threads = 128;
    const unsigned int terminusPerBlock = std::min(
        fill_block_threads,
        1 + (gbts_consts::live_path_buffer - 1) / pathsPerTerminus);

    fill_path_store_kernel({nTerminusEdges, terminusPerBlock,
                            cfg.max_num_neighbours,
                            nPaths + nTerminusEdges, nConnectedEdges,
                            path_store_buf, output_graph_buf, levels_buf,
                            d_counters + k_nPathStoreSize});

    fit_segments_kernel({nPaths, nTerminusEdges, cfg.max_num_neighbours,
                         nConnectedEdges, cfg.minLevel, reducedSP_buf,
                         output_graph_buf, path_store_buf, seed_proposals_buf,
                         edge_bids_buf, seed_ambiguity_buf,
                         d_counters + k_nPathStoreSize,
                         d_counters + k_nProps,
                         cfg.seed_extraction_params});

    copy()(counters_buf, h_counters)->wait();

    const unsigned int nProps = h_counters[k_nProps];
    TRACCC_DEBUG("nProps " << nProps);
    if (nProps == 0) {
        TRACCC_WARNING("No seed proposals were found");
        return {0, mr().main};
    }

    reset_edge_bids_kernel(
        {nProps, -1, path_store_buf, seed_proposals_buf, edge_bids_buf,
         seed_ambiguity_buf, d_counters + k_nRejectedProps});

    for (int round = 0; round < 5; ++round) {
        copy().memset(edge_bids_buf, 0)->ignore();

        seeds_rebid_for_edges_kernel({nProps, path_store_buf,
                                      seed_proposals_buf, edge_bids_buf,
                                      seed_ambiguity_buf});

        reset_edge_bids_kernel(
            {nProps, round, path_store_buf, seed_proposals_buf, edge_bids_buf,
             seed_ambiguity_buf, d_counters + k_nRejectedProps});
    }

    copy()(counters_buf, h_counters)->wait();
    const unsigned int nRejectedProps = h_counters[k_nRejectedProps];
    const unsigned int nSeeds = (nRejectedProps >= nProps) ? 0u : nProps - nRejectedProps;
    
    TRACCC_DEBUG("Rejected " << nRejectedProps << " out of " << nProps
                        << " seed proposals");
    if (nSeeds == 0) {
        TRACCC_WARNING("All seed proposals were rejected");
        return {0, mr().main};
    }

    // 8. Convert to 3sp seeds and make output buffer.
    edm::seed_collection::buffer output_seeds(
        2 * nSeeds, mr().main, vecmem::data::buffer_type::resizable);
    copy().setup(output_seeds)->wait();

    collection_types<unsigned long long int>::buffer hit_bids_buf(nSp,
                                                                       mr().main);
    copy().setup(hit_bids_buf)->ignore();
    copy().memset(hit_bids_buf, 0)->wait();

    const unsigned int edge_size = 1u + 2u + cfg.max_num_neighbours;
    seeds_bid_for_hits_kernel({nProps, nSeeds, edge_size, nConnectedEdges,
                               output_graph_buf, seed_proposals_buf,
                               path_store_buf, seed_ambiguity_buf,
                               hit_bids_buf});

    gbts_seed_conversion_kernel({nProps, nSeeds, cfg.max_num_neighbours,
                                 nConnectedEdges, seed_proposals_buf,
                                 seed_ambiguity_buf, path_store_buf,
                                 output_graph_buf, reducedSP_buf, output_seeds,
                                 hit_bids_buf, cfg.seed_ambi_params});
    sync();

    const unsigned int outputSeeds = copy().get_size(output_seeds);
    TRACCC_DEBUG("GBTS found " << outputSeeds << " seeds");
    return output_seeds;
}

}  // namespace traccc::device
