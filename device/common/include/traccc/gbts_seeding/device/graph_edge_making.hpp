/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// Create candidate edges between node pairs in compatible (eta, phi) bins.
///
/// One CUDA block handles one bin-pair task.  Threads stage a chunk of bin-1
/// nodes into shared-memory caches (phi as a separate float array, and
/// (r, z, tau_min, tau_max) packed into a float4), then for every bin-2 node
/// test the cached chunk against geometric and kinematic cuts, atomically
/// reserving a slot in the output via nEdgesCounter.  Per-bin-1 outgoing
/// counts are accumulated in a third shared-mem buffer and flushed to global
/// once at the end of the block.
///
/// @param[in]  blockIndex                    CUDA block index
/// @param[in]  threadIndex                   CUDA thread index in block
/// @param[in]  blockSize                     CUDA block size
/// @param[in]  barrier                       Block-wide barrier
/// @param[in,out] phi                        Shared-mem cache: phi / node
/// @param[in,out] node_pack                  Shared-mem cache: (r, z,
///                                           tau_min, tau_max) / node
/// @param[in,out] s_outgoing                 Shared-mem per-bin-1 outgoing
///                                           edge counter (flushed at end)
/// @param[in]  d_bin_pair_views_view         Per-task bin1/bin2 ranges
/// @param[in]  d_bin_pair_dphi_view          Per-task dphi window
/// @param[in]  d_node_params_view            Sorted node geometry tuples
/// @param[in]  d_graph_building_params       Geometric / kinematic cuts
/// @param[in,out] nEdgesCounter              Global edge-slot atomic counter
/// @param[out] d_edge_nodes_view             (src, dst) per produced edge
/// @param[out] d_edge_exp_eta_view           Per-edge exp(-eta) ( = sqrt(1+tau^2)-tau )
/// @param[out] d_edge_curv_view              Per-edge curvature (dphi / dr)
/// @param[out] d_edge_phi_z_view             Per-edge predicted phi at layer-2 radius
/// @param[out] d_edge_phi_w_view             Per-edge predicted phi at layer-1 radius
/// @param[out] d_num_outgoing_edges_view     Per-dst-node incoming-edge count
/// @param[in]  nMaxEdges                     Upper bound on edges (cap)
/// @param[in]  nPhiBins                      Number of phi bins per eta slice
///
template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_making(
    const unsigned int blockIndex,
    const unsigned int threadIndex,
    const unsigned int blockSize,
    const barrier_t& barrier, float* phi, float4* node_pack,
    unsigned int* s_outgoing,
    const collection_types<unsigned int>::const_view& d_bin_pair_views_view,
    const collection_types<float>::const_view& d_bin_pair_dphi_view,
    const collection_types<float>::const_view& d_node_params_view,
    const gbts_edge_making_params& d_gbts_edge_making_params,
    unsigned int& nEdgesCounter,
    const collection_types<uint2>::view d_edge_nodes_view,
    const collection_types<float>::view d_edge_exp_eta_view,
    const collection_types<float>::view d_edge_curv_view,
    const collection_types<float>::view d_edge_phi_z_view,
    const collection_types<float>::view d_edge_phi_w_view,
    const collection_types<unsigned int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_making.ipp"
