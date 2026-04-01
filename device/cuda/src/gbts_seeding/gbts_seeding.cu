/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/edm/container.hpp"

#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"


// Kernel includes
#include "traccc/gbts_seeding/device/count_sp_by_layer.hpp"
#include "traccc/gbts_seeding/device/bin_sp_by_layer.hpp"
#include "traccc/gbts_seeding/device/node_phi_binning.hpp"
#include "traccc/gbts_seeding/device/node_eta_binning.hpp"
#include "traccc/gbts_seeding/device/eta_phi_histo.hpp"
#include "traccc/gbts_seeding/device/eta_phi_counting.hpp"
#include "traccc/gbts_seeding/device/eta_phi_prefix_sum.hpp"
#include "traccc/gbts_seeding/device/node_sorting.hpp"
#include "traccc/gbts_seeding/device/minmax_rad.hpp"
#include "traccc/gbts_seeding/device/cca_iteration.hpp"
#include "traccc/gbts_seeding/device/count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/add_terminus_to_path_store.hpp"
#include "traccc/gbts_seeding/device/fill_path_store.hpp"
#include "traccc/gbts_seeding/device/fit_segments.hpp"
#include "traccc/gbts_seeding/device/reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/seeds_rebid_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_seed_conversion.hpp"
#include "traccc/gbts_seeding/device/graph_edge_making.hpp"
#include "traccc/gbts_seeding/device/graph_edge_linking.hpp"
#include "traccc/gbts_seeding/device/graph_edge_matching.hpp"
#include "traccc/gbts_seeding/device/edge_reindexing.hpp"
#include "traccc/gbts_seeding/device/graph_compression.hpp"



namespace traccc::cuda {
namespace kernels {

__global__ void count_sp_by_layer_kernel(const traccc::edm::spacepoint_collection::const_view spacepoints_view,
    const edm::measurement_collection::const_view measurements_view,
    const collection_types<short>::const_view volumeToLayerMap_view, const collection_types<std::pair<unsigned int, unsigned int>>::const_view surfaceToLayerMap_view,
    const collection_types<char>::const_view layerType_view, const collection_types<float4>::view reducedSP_view, const collection_types<int>::view layerCounts_view,
    const collection_types<short>::view spacepointsLayer_view, const float type1_max_width,
    const unsigned int nSp, const long unsigned int volumeMapSize,
    const long unsigned int surfaceMapSize, bool doTauCut = true) {
        
        device::count_sp_by_layer(details::global_index1(), spacepoints_view, measurements_view, volumeToLayerMap_view, surfaceToLayerMap_view, layerType_view, reducedSP_view, layerCounts_view, spacepointsLayer_view, type1_max_width, nSp, volumeMapSize, surfaceMapSize, doTauCut);
    }

__global__ void bin_sp_by_layer_kernel(const collection_types<float4>::view sp_params_view, const collection_types<float4>::const_view reducedSP_view,
    const collection_types<int>::view layerCounts_view,
    const collection_types<short>::const_view spacepointsLayer_view,
    const collection_types<int>::view original_sp_idx_view, const unsigned int nSp) {

        device::bin_sp_by_layer(details::global_index1(), sp_params_view, reducedSP_view, layerCounts_view, spacepointsLayer_view, original_sp_idx_view, nSp);
    }

__global__ void node_phi_binning_kernel(const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<int>::view d_node_phi_index_view,
    const unsigned int nNodesPerBlock,
    const unsigned int nNodes,
    const unsigned int nPhiBins) {

        int begin_node = blockIdx.x * nNodesPerBlock;
        for (int idx = threadIdx.x + begin_node; idx < begin_node + nNodesPerBlock; idx += blockDim.x) {
            device::node_phi_binning_kernel(idx, d_sp_params_view, d_node_phi_index_view, nNodesPerBlock, nNodes, nPhiBins);
        }
    }

__global__ void node_eta_binning_kernel(const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<std::pair<int, int>>::const_view d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view d_layer_geo_view,
    const collection_types<int>::view d_node_eta_index_view,
    const collection_types<int>::const_view layerCounts_view,
    const unsigned int nLayers) {

        usigned int layerIdx = blockIdx.x;

        if (layerIdx >= nLayers){
            return;
        }
        const collection_types<int>::const_device layerCounts(layerCounts_view);
 
        const int layer_begin = layerCounts[layerIdx];
        const int layer_end = layerCounts[layerIdx + 1];

        for (unsigned int idx = threadIdx.x + layer_begin; idx < layer_end;
            idx += blockDim.x) {
            device::node_eta_binning_kernel(layerIdx, idx, d_sp_params_view, d_layer_info_view, d_layer_geo_view, d_node_eta_index_view, layerCounts_view, nLayers);
        }


    }

__global__ void eta_phi_histo_kernel(const collection_types<int>::const_view d_node_phi_index_view,
    const collection_types<int>::const_view d_node_eta_index_view,
    const collection_types<int>::view d_eta_phi_histo_view,
    const unsigned int nNodesPerBlock,
    const unsigned int nNodes,
    const unsigned int nPhiBins) {

        int begin_node = blockIdx.x * nNodesPerBlock;
        for (int idx = threadIdx.x + begin_node; idx < begin_node + nNodesPerBlock; idx += blockDim.x) {
            device::eta_phi_histo_kernel(idx, d_node_phi_index_view, d_node_eta_index_view, d_eta_phi_histo_view, nNodes, nPhiBins);
        }
}

__global__ void eta_phi_counting_kernel(const collection_types<int>::const_view d_histo_view,
    const collection_types<int>::view d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned nBinsPerBlock,
    const unsigned maxEtaBin,
    const unsigned nPhiBins) {
        
        int eta_bin_start = nBinsPerBlock * blockIdx.x;

        int eta_bin_idx = eta_bin_start + threadIdx.x;

        device::eta_phi_counting_kernel(eta_bin_idx, d_histo_view, d_eta_node_counter_view, d_phi_cusums_view, maxEtaBin, nPhiBins);
    }

__global__ void eta_phi_prefix_sum_kernel(const collection_types<int>::const_view d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned int nBinsPerBlock,
    const unsigned int maxEtaBin,
    const unsigned int nPhiBins) {
        int eta_bin_start = nBinsPerBlock * blockIdx.x;

    int eta_bin_idx = eta_bin_start + threadIdx.x;

    device::eta_phi_prefix_sum_kernel(eta_bin_idx, d_eta_node_counter_view, d_phi_cusums_view, nPhiBins, maxEtaBin);
}


__global__ void node_sorting_kernel(
    const collection_types<float4>::const_view d_sp_params_view, 
    const collection_types<int>::const_view d_node_eta_index_view,
    const collection_types<int>::const_view d_node_phi_index_view, 
    const collection_types<int>::view d_phi_cusums_view, 
    const collection_types<float>::view d_node_params_view,
    const collection_types<int>::view d_node_index_view, 
    const collection_types<int>::const_view d_original_sp_idx_view, 
    const gbts_graph_building_params ap,
    const unsigned int nNodesPerBlock, 
    const unsigned int nNodes,
    const unsigned int nPhiBins) {
        int begin_node = blockIdx.x * nNodesPerBlock;

    for (int idx = threadIdx.x + begin_node; idx < begin_node + nNodesPerBlock;
         idx += blockDim.x) {
            device::node_sorting_kernel(idx, d_sp_params_view, d_node_eta_index_view, d_node_phi_index_view, d_phi_cusums_view, d_node_params_view, d_node_index_view, d_original_sp_idx_view, ap, nNodesPerBlock, nNodes, nPhiBins);
         }
    }


__global__ void minmax_rad_kernel(const collection_types<int>::const_view d_eta_bin_views_view, // Here the type changed as a quick fix to avoid the issue with the eta_bin_views buffer
    const collection_types<float>::const_view d_node_params_view,
    const collection_types<float>::view d_bin_rads_view, // Here the type changed as a quick fix to avoid the issue with the bin_rads buffer
    const unsigned int nBinsPerBlock,
    const unsigned int maxEtaBin) {
        int eta_bin_start = nBinsPerBlock * blockIdx.x;
        int eta_bin_idx = eta_bin_start + threadIdx.x;

        device::minmax_rad_kernel(eta_bin_idx, d_eta_bin_views_view, d_node_params_view, d_bin_rads_view, nBinsPerBlock, maxEtaBin);
    }


__global__ static void CCA_IterationKernel(
    const collection_types<int>::const_view d_output_graph_view, 
    const collection_types<char>::view d_levels_view, 
    const collection_types<int>::view d_active_edges_view,
    const collection_types<short2>::view d_outgoing_paths_view, 
    const collection_types<unsigned int>::view d_counters_view, 
    const int iter,
    const unsigned int nEdges, 
    const unsigned int max_num_neighbours, 
    const int minLevel) {
        device::cca_iteration_kernel(details::global_index1(), d_output_graph_view, d_levels_view, d_active_edges_view, d_outgoing_paths_view, d_counters_view, iter, nEdges, max_num_neighbours, minLevel);
    }

__global__ void count_terminus_edges_kernel(const collection_types<short2>::view d_outgoing_paths_view,
    const collection_types<unsigned int>::view d_counters_view, unsigned int nEdges) {
        device::count_terminus_edges_kernel(details::global_index1(), d_outgoing_paths_view, d_counters_view, nEdges);
    }

__global__ void add_terminus_to_path_store_kernel(const collection_types<int2>::view d_path_store_view,
    const collection_types<short2>::const_view d_outgoing_paths_view, unsigned int nEdges) {
        device::add_terminus_to_path_store_kernel(details::global_index1(), d_path_store_view, d_outgoing_paths_view, nEdges);
    }

void __global__ fill_path_store(const collection_types<int2>::view d_path_store_view,
    const collection_types<int>::const_view d_output_graph_view,
    const collection_types<char>::const_view d_levels_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nTerminus,
    const unsigned int nTerminusPerBlock,
    const unsigned int max_num_neighbours,
    const unsigned int nPaths) {
        // TODO: ref to common implementation
    }



void __global__ fit_segments(
    const collection_types<float4>::const_view d_sp_reduced_view, 
    const collection_types<int>::const_view d_output_graph_view, 
    const collection_types<int2>::const_view d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view, 
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view, 
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nTerminusEdges, 
    const int minLevel, 
    const unsigned int max_num_neighbours,
    const gbts_seed_extraction_params seed_extraction_params) {
        device::fit_segments_kernel(details::global_index1(), d_sp_reduced_view, d_output_graph_view, d_path_store_view, d_seed_proposals_view, d_edge_bids_view, d_seed_ambiguity_view, d_counters_view, nTerminusEdges, minLevel, max_num_neighbours, seed_extraction_params);
    }



void __global__ reset_edge_bids(const collection_types<int2>::const_view d_path_store_view, 
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const collection_types<unsigned int>::view d_counters_view,
    const int round) {
        device::reset_edge_bids_kernel(details::global_index1(), d_path_store_view, d_seed_proposals_view, d_edge_bids_view, d_seed_ambiguity_view, d_counters_view, round);
    }

void __global__ seeds_rebid_for_edges(const collection_types<int2>::const_view d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const unsigned int nProps) {
        device::seeds_rebid_for_edges_kernel(details::global_index1(), d_path_store_view, d_seed_proposals_view, d_edge_bids_view, d_seed_ambiguity_view, nProps);
    }

    void __global__ gbts_seed_conversion_kernel(
        const collection_types<int2>::view d_seed_proposals_view, 
        const collection_types<char>::view d_seed_ambiguity_view, 
        const collection_types<int2>::const_view d_path_store_view,
        const collection_types<int>::const_view d_output_graph_view, 
        const edm::seed_collection::view output_seeds,
        const unsigned int nProps, 
        const unsigned int max_num_neighbours) {
        device::gbts_seed_conversion_kernel(details::global_index1(), d_seed_proposals_view, d_seed_ambiguity_view, d_path_store_view, d_output_graph_view, output_seeds, nProps, max_num_neighbours);
    }


    __global__ static void graphEdgeMakingKernel(
        const collection_types<int>::const_view d_bin_pair_views_view, const collection_types<float>::const_view d_bin_pair_dphi_view, // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
        const collection_types<float>::const_view d_node_params_view,
        const gbts_graph_building_params d_graph_building_params,
        const collection_types<unsigned int>::view d_counters_view, const collection_types<int2>::view d_edge_nodes_view, const collection_types<half4>::view d_edge_params_view,
        const collection_types<int>::view d_num_outgoing_edges_view, const unsigned int nMaxEdges,
        const unsigned int nPhiBins) {
        // TODO: ref to common implementation
    }

    __global__ static void graphEdgeLinkingKernel(const collection_types<int2>::const_view d_edge_nodes_view,
        const collection_types<int>::view d_edge_links_view,
        const collection_types<int>::view d_num_outgoing_edges_view,
        const unsigned int nEdges) {
        device::graph_edge_linking_kernel(details::global_index1(), d_edge_nodes_view, d_edge_links_view, d_num_outgoing_edges_view, nEdges);
    }

    __global__ static void graphEdgeMatchingKernel(
        const gbts_graph_building_params d_graph_building_params,
        const collection_types<float4>::const_view d_edge_params_view, const collection_types<int2>::const_view d_edge_nodes_view,
        const collection_types<int>::const_view d_num_outgoing_edges_view, const collection_types<int>::const_view d_edge_links_view,
        const collection_types<unsigned char>::view d_num_neighbours_view, const collection_types<int>::view d_neighbours_view, const collection_types<int>::view d_reIndexer_view,
        const collection_types<unsigned int>::view d_counters_view, const unsigned int nEdges,
        const unsigned int nMaxNei) {

        device::graph_edge_matching_kernel(details::global_index1(), d_graph_building_params, d_edge_params_view, d_edge_nodes_view, d_num_outgoing_edges_view, d_edge_links_view, d_num_neighbours_view, d_neighbours_view, d_reIndexer_view, d_counters_view, nEdges, nMaxNei);
    }

    __global__ static void edgeReIndexingKernel(const collection_types<int>::view d_reIndexer_view, const collection_types<unsigned int>::view d_counters_view,
        const unsigned int nEdges) {
        device::edge_re_indexing_kernel(details::global_index1(), d_reIndexer_view, d_counters_view, nEdges);
    }

    __global__ static void graphCompressionKernel(
        const collection_types<int>::const_view d_orig_node_index_view, const collection_types<int2>::const_view d_edge_nodes_view,
        const collection_types<unsigned char>::const_view d_num_neighbours_view, const collection_types<int>::const_view d_neighbours_view,
        const collection_types<int>::const_view d_reIndexer_view, const collection_types<int>::view d_output_graph_view,
        const unsigned int nEdgesPerBlock, const unsigned int nEdges,
        const unsigned int nMaxNei) {
        
        int begin_edge = blockIdx.x * nEdgesPerBlock;
        for (int idx = threadIdx.x + begin_edge; idx < begin_edge + nEdgesPerBlock; idx += blockDim.x) {
            device::graph_compression_kernel(idx, d_orig_node_index_view, d_edge_nodes_view, d_num_neighbours_view, d_neighbours_view, d_reIndexer_view, d_output_graph_view, nMaxNei, nEdges);
        }
    }



} // namespace kernels
} // namespace traccc::cuda