/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

 #pragma once

 // Local include(s).
 #include "traccc/cuda/utils/algorithm_base.hpp"
 
 // Project include(s).
 #include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"
 
 namespace traccc::cuda {
 
 /// GBTS seeding algorithm running on an NVIDIA GPU.
 ///
 /// This algorithm returns a buffer which is not necessarily filled yet. A
 /// synchronisation statement is required before destroying this buffer.
 ///
 class gbts_seeding_algorithm : public device::gbts_seeding_algorithm,
                                public cuda::algorithm_base {
 
     public:
     /// Constructor
     ///
     /// @param cfg    GBTS seeding configuration
     /// @param mr     Memory resource(s)
     /// @param copy   Copy object
     /// @param str    CUDA stream
     /// @param logger Logger instance
     ///
     gbts_seeding_algorithm(
         const gbts_seedfinder_config& cfg, const traccc::memory_resource& mr,
         vecmem::copy& copy, stream& str,
         std::unique_ptr<const Logger> logger = getDummyLogger().clone());
 
     private:
     /// @name Virtual kernel launcher implementations
     /// @{
 
     void count_sp_by_layer_kernel(
         const count_sp_by_layer_payload&) const override;
     void bin_sp_by_layer_kernel(
         const bin_sp_by_layer_payload&) const override;
     void node_phi_binning_kernel(
         const node_phi_binning_payload&) const override;
     void node_eta_binning_kernel(
         const node_eta_binning_payload&) const override;
     void eta_phi_histo_kernel(
         const eta_phi_histo_payload&) const override;
     void eta_phi_counting_kernel(
         const eta_phi_counting_payload&) const override;
     void eta_phi_prefix_sum_kernel(
         const eta_phi_prefix_sum_payload&) const override;
     void node_sorting_kernel(
         const node_sorting_payload&) const override;
     void minmax_rad_kernel(
         const minmax_rad_payload&) const override;
     void graph_edge_making_kernel(
         const graph_edge_making_payload&) const override;
     void graph_edge_linking_kernel(
         const graph_edge_linking_payload&) const override;
     void graph_edge_matching_kernel(
         const graph_edge_matching_payload&) const override;
     void edge_re_indexing_kernel(
         const edge_re_indexing_payload&) const override;
     void graph_compression_kernel(
         const graph_compression_payload&) const override;
     void cca_iteration_kernel(
         const cca_iteration_payload&) const override;
     void count_terminus_edges_kernel(
         const count_terminus_edges_payload&) const override;
     void add_terminus_to_path_store_kernel(
         const add_terminus_to_path_store_payload&) const override;
     void fill_path_store_kernel(
         const fill_path_store_payload&) const override;
     void fit_segments_kernel(
         const fit_segments_payload&) const override;
     void reset_edge_bids_kernel(
         const reset_edge_bids_payload&) const override;
     void seeds_rebid_for_edges_kernel(
         const seeds_rebid_for_edges_payload&) const override;
     void gbts_seed_conversion_kernel(
         const gbts_seed_conversion_payload&) const override; 
     /// @}
 };
 
 }  // namespace traccc::cuda
 