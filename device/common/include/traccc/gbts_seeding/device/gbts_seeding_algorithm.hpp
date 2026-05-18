/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/algorithm_base.hpp"

// Project include(s).
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// System include(s).
#include <memory>
#include <utility>

namespace traccc::device {

/// Main algorithm for performing GBTS seeding on a device (backend-agnostic).
///
/// The algorithm orchestrates the sequence of kernel launches and host-side
/// synchronisations.  Backend-specific subclasses are responsible for
/// implementing the individual kernel launchers.
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class gbts_seeding_algorithm
    : public algorithm<edm::seed_collection::buffer(
          const edm::spacepoint_collection::const_view&,
          const edm::measurement_collection::const_view&)>,
      public messaging,
      public algorithm_base {

    public:
    /// Constructor for the GBTS seed finding algorithm
    ///
    /// @param cfg The GBTS seed finding configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param logger The logger instance to use
    ///
    gbts_seeding_algorithm(
        const gbts_seedfinder_config& cfg, const memory_resource& mr,
        vecmem::copy& copy,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Destructor
    virtual ~gbts_seeding_algorithm();

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints is a view of all spacepoints in the event
    /// @param measurements is a view of all measurements in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(
        const edm::spacepoint_collection::const_view& spacepoints,
        const edm::measurement_collection::const_view& measurements)
        const override;

    protected:
    /// @name Payloads & kernel launchers (to be implemented by backends)
    /// @{

    /// Payload for the @c count_sp_by_layer_kernel function
    struct count_sp_by_layer_kernel_payload {
        unsigned int nSp;
        const edm::spacepoint_collection::const_view& spacepoints;
        const edm::measurement_collection::const_view& measurements;
        const collection_types<short>::const_view& volumeToLayerMap;
        const collection_types<std::pair<unsigned int, unsigned int>>::
            const_view& surfaceToLayerMap;
        const collection_types<char>::const_view& layerType;
        collection_types<float4>::view& reducedSP;
        collection_types<int>::view& layerCounts;
        collection_types<short>::view& spacepointsLayer;
        float type1_max_width;
        unsigned long volumeMapSize;
        unsigned long surfaceMapSize;
        bool doTauCut;
    };
    virtual void count_sp_by_layer_kernel(
        const count_sp_by_layer_kernel_payload& payload) const = 0;

    /// Payload for the @c bin_sp_by_layer_kernel function
    struct bin_sp_by_layer_kernel_payload {
        unsigned int nSp;
        collection_types<float4>::view& sp_params;
        const collection_types<float4>::const_view& reducedSP;
        collection_types<int>::view& layerCounts;
        const collection_types<short>::const_view& spacepointsLayer;
        collection_types<int>::view& original_sp_idx;
    };
    virtual void bin_sp_by_layer_kernel(
        const bin_sp_by_layer_kernel_payload& payload) const = 0;

    /// Payload for the @c node_eta_binning_kernel function
    struct node_eta_binning_kernel_payload {
        unsigned int nLayers;
        const collection_types<float4>::const_view& sp_params;
        const collection_types<std::pair<int, int>>::const_view& layer_info;
        const collection_types<std::pair<float, float>>::const_view& layer_geo;
        collection_types<int>::view& node_eta_index;
        collection_types<int>::view& layerCounts;
    };
    virtual void node_eta_binning_kernel(
        const node_eta_binning_kernel_payload& payload) const = 0;

    /// Payload for the @c eta_phi_histo_kernel function
    struct eta_phi_histo_kernel_payload {
        unsigned int nNodes;
        unsigned int nPhiBins;
        collection_types<int>::view& node_phi_index;
        const collection_types<int>::const_view& node_eta_index;
        collection_types<int>::view& eta_phi_histo;
        const collection_types<float4>::const_view& sp_params;
    };
    virtual void eta_phi_histo_kernel(
        const eta_phi_histo_kernel_payload& payload) const = 0;

    /// Payload for the @c eta_phi_counting_kernel function
    struct eta_phi_counting_kernel_payload {
        unsigned int nEtaBins;
        unsigned int nPhiBins;
        const collection_types<int>::const_view& eta_phi_histo;
        collection_types<int>::view& eta_node_counter;
        collection_types<int>::view& phi_cusums;
    };
    virtual void eta_phi_counting_kernel(
        const eta_phi_counting_kernel_payload& payload) const = 0;

    /// Payload for the @c eta_phi_prefix_sum_kernel function
    struct eta_phi_prefix_sum_kernel_payload {
        unsigned int nEtaBins;
        unsigned int nPhiBins;
        const collection_types<int>::const_view& eta_node_counter;
        collection_types<int>::view& phi_cusums;
    };
    virtual void eta_phi_prefix_sum_kernel(
        const eta_phi_prefix_sum_kernel_payload& payload) const = 0;

    /// Payload for the @c node_sorting_kernel function
    struct node_sorting_kernel_payload {
        unsigned int nNodes;
        unsigned int nPhiBins;
        const collection_types<float4>::const_view& sp_params;
        const collection_types<int>::const_view& node_eta_index;
        const collection_types<int>::const_view& node_phi_index;
        collection_types<int>::view& phi_cusums;
        collection_types<float>::view& node_params;
        collection_types<int>::view& node_index;
        const collection_types<int>::const_view& original_sp_idx;
        gbts_graph_building_params graph_building_params;
    };
    virtual void node_sorting_kernel(
        const node_sorting_kernel_payload& payload) const = 0;

    /// Payload for the @c minmax_rad_kernel function
    struct minmax_rad_kernel_payload {
        unsigned int nEtaBins;
        const collection_types<int>::const_view& eta_bin_views;
        const collection_types<float>::const_view& node_params;
        collection_types<float>::view& bin_rads;
    };
    virtual void minmax_rad_kernel(
        const minmax_rad_kernel_payload& payload) const = 0;

    /// Payload for the @c graph_edge_making_kernel function
    struct graph_edge_making_kernel_payload {
        unsigned int nUsedBinPairs;
        unsigned int nMaxEdges;
        unsigned int nPhiBins;
        const collection_types<int>::const_view& bin_pair_views;
        const collection_types<float>::const_view& bin_pair_dphi;
        const collection_types<float>::const_view& node_params;
        gbts_graph_building_params graph_building_params;
        unsigned int& nEdgesCounter;
        collection_types<int2>::view& edge_nodes;
        collection_types<half4>::view& edge_params;
        collection_types<int>::view& num_outgoing_edges;
    };
    virtual void graph_edge_making_kernel(
        const graph_edge_making_kernel_payload& payload) const = 0;

    /// Payload for the @c graph_edge_linking_kernel function
    struct graph_edge_linking_kernel_payload {
        unsigned int nEdges;
        const collection_types<int2>::const_view& edge_nodes;
        collection_types<int>::view& edge_links;
        collection_types<int>::view& num_outgoing_edges;
    };
    virtual void graph_edge_linking_kernel(
        const graph_edge_linking_kernel_payload& payload) const = 0;

    /// Payload for the @c graph_edge_matching_kernel function
    struct graph_edge_matching_kernel_payload {
        unsigned int nEdges;
        unsigned int nMaxNei;
        gbts_graph_building_params graph_building_params;
        const collection_types<half4>::const_view& edge_params;
        const collection_types<int2>::const_view& edge_nodes;
        const collection_types<int>::const_view& num_outgoing_edges;
        const collection_types<int>::const_view& edge_links;
        collection_types<unsigned char>::view& num_neighbours;
        collection_types<int>::view& neighbours;
        collection_types<int>::view& reIndexer;
        unsigned int& nConnectionsCounter;
    };
    virtual void graph_edge_matching_kernel(
        const graph_edge_matching_kernel_payload& payload) const = 0;

    /// Payload for the @c edge_re_indexing_kernel function
    struct edge_re_indexing_kernel_payload {
        unsigned int nEdges;
        collection_types<int>::view& reIndexer;
        unsigned int& nConnectedEdgesCounter;
    };
    virtual void edge_re_indexing_kernel(
        const edge_re_indexing_kernel_payload& payload) const = 0;

    /// Payload for the @c graph_compression_kernel function
    struct graph_compression_kernel_payload {
        unsigned int nEdges;
        unsigned int nMaxNei;
        const collection_types<int>::const_view& orig_node_index;
        const collection_types<int2>::const_view& edge_nodes;
        const collection_types<unsigned char>::const_view& num_neighbours;
        const collection_types<int>::const_view& neighbours;
        const collection_types<int>::const_view& reIndexer;
        collection_types<int>::view& output_graph;
    };
    virtual void graph_compression_kernel(
        const graph_compression_kernel_payload& payload) const = 0;

    /// Payload for the @c cca_iteration_kernel function
    struct cca_iteration_kernel_payload {
        unsigned int nConnectedEdges;
        unsigned int max_num_neighbours;
        int iter;
        int minLevel;
        const collection_types<int>::const_view& output_graph;
        collection_types<char>::view& levels;
        collection_types<int>::view& active_edges;
        collection_types<short2>::view& outgoing_paths;
        unsigned int& nActiveEdgesA;
        unsigned int& nActiveEdgesB;
        unsigned int& nCCABlockCounter;
    };
    virtual void cca_iteration_kernel(
        const cca_iteration_kernel_payload& payload) const = 0;

    /// Payload for the @c count_terminus_edges_kernel function
    struct count_terminus_edges_kernel_payload {
        unsigned int nConnectedEdges;
        collection_types<short2>::view& outgoing_paths;
        unsigned int& nPathsCounter;
        unsigned int& nPathStoreSizeCounter;
    };
    virtual void count_terminus_edges_kernel(
        const count_terminus_edges_kernel_payload& payload) const = 0;

    /// Payload for the @c add_terminus_to_path_store_kernel function
    struct add_terminus_to_path_store_kernel_payload {
        unsigned int nConnectedEdges;
        collection_types<int2>::view& path_store;
        const collection_types<short2>::const_view& outgoing_paths;
    };
    virtual void add_terminus_to_path_store_kernel(
        const add_terminus_to_path_store_kernel_payload& payload) const = 0;

    /// Payload for the @c fill_path_store_kernel function
    struct fill_path_store_kernel_payload {
        unsigned int nTerminusEdges;
        unsigned int nTerminusPerBlock;
        unsigned int max_num_neighbours;
        unsigned int nPaths;
        collection_types<int2>::view& path_store;
        const collection_types<int>::const_view& output_graph;
        const collection_types<char>::const_view& levels;
        unsigned int& nPathStoreSizeCounter;
    };
    virtual void fill_path_store_kernel(
        const fill_path_store_kernel_payload& payload) const = 0;

    /// Payload for the @c fit_segments_kernel function
    struct fit_segments_kernel_payload {
        unsigned int nPaths;
        unsigned int nTerminusEdges;
        unsigned int max_num_neighbours;
        int minLevel;
        const collection_types<float4>::const_view& reducedSP;
        const collection_types<int>::const_view& output_graph;
        const collection_types<int2>::const_view& path_store;
        collection_types<int2>::view& seed_proposals;
        collection_types<unsigned long long int>::view& edge_bids;
        collection_types<char>::view& seed_ambiguity;
        unsigned int& nPathStoreSize;
        unsigned int& nPropsCounter;
        gbts_seed_extraction_params seed_extraction_params;
    };
    virtual void fit_segments_kernel(
        const fit_segments_kernel_payload& payload) const = 0;

    /// Payload for the @c reset_edge_bids_kernel function
    struct reset_edge_bids_kernel_payload {
        unsigned int nProps;
        int round;
        const collection_types<int2>::const_view& path_store;
        collection_types<int2>::view& seed_proposals;
        collection_types<unsigned long long int>::view& edge_bids;
        collection_types<char>::view& seed_ambiguity;
        unsigned int& nRejectedPropsCounter;
    };
    virtual void reset_edge_bids_kernel(
        const reset_edge_bids_kernel_payload& payload) const = 0;

    /// Payload for the @c seeds_rebid_for_edges_kernel function
    struct seeds_rebid_for_edges_kernel_payload {
        unsigned int nProps;
        const collection_types<int2>::const_view& path_store;
        collection_types<int2>::view& seed_proposals;
        collection_types<unsigned long long int>::view& edge_bids;
        collection_types<char>::view& seed_ambiguity;
    };
    virtual void seeds_rebid_for_edges_kernel(
        const seeds_rebid_for_edges_kernel_payload& payload) const = 0;

    /// Payload for the @c seeds_bid_for_hits_kernel function
    struct seeds_bid_for_hits_kernel_payload {
        unsigned int nProps;
        unsigned int nSeeds;
        int edge_size;
        const collection_types<int>::const_view& output_graph;
        const collection_types<int2>::const_view& seed_proposals;
        const collection_types<int2>::const_view& path_store;
        const collection_types<char>::const_view& seed_ambiguity;
        collection_types<unsigned long long int>::view& hit_bids;
    };
    virtual void seeds_bid_for_hits_kernel(
        const seeds_bid_for_hits_kernel_payload& payload) const = 0;

    /// Payload for the @c gbts_seed_conversion_kernel function
    struct gbts_seed_conversion_kernel_payload {
        unsigned int nProps;
        unsigned int nSeeds;
        unsigned int max_num_neighbours;
        const collection_types<int2>::const_view& seed_proposals;
        const collection_types<char>::const_view& seed_ambiguity;
        const collection_types<int2>::const_view& path_store;
        const collection_types<int>::const_view& output_graph;
        const collection_types<float4>::const_view& reducedSP;
        edm::seed_collection::view& output_seeds;
        collection_types<unsigned long long int>::view& hit_bids;
        gbts_seed_ambi_params seed_ambi_params;
    };
    virtual void gbts_seed_conversion_kernel(
        const gbts_seed_conversion_kernel_payload& payload) const = 0;

    /// @}

    /// Hook for backend-specific stream synchronisation (used between kernels
    /// when the host needs to read back counter values).
    virtual void sync() const = 0;

    private:
    /// Internal data type
    struct data;
    /// Pointer to internal data
    std::unique_ptr<data> m_data;

};  // class gbts_seeding_algorithm

}  // namespace traccc::device
