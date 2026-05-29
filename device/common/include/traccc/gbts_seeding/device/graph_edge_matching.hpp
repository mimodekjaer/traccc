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

/// For each edge, find compatible neighbour edges sharing its destination node.
///
/// One CUDA block processes a chunk of edges.  Threads pair-test each edge
/// against all incoming edges at its destination node using the cached edge
/// parameters, recording up to nMaxNei accepted neighbours, marking the
/// edge as "kept", and atomically incrementing nConnectionsCounter.
///
/// The kernel accumulates per-block connection counts in shared memory and
/// commits a single global atomic per block, so the wrapping `__global__`
/// kernel must allocate one `unsigned int` of shared memory and pass it in
/// via @p sm_block_connections, alongside a block-level barrier.
///
/// @param[in]  globalIndex                  Global thread index (one thread per edge)
/// @param[in]  threadIndex                  Block-local thread index
/// @param[in]  barrier                      Block-level barrier
/// @param[in]  d_graph_building_params      Geometric / kinematic cuts
/// @param[in]  d_edge_exp_eta_view          Per-edge exp(-eta) ( = sqrt(1+tau^2)-tau )
/// @param[in]  d_edge_curv_view             Per-edge curvature (dphi / dr)
/// @param[in]  d_edge_phi_z_view            Per-edge predicted phi at layer-2 radius
/// @param[in]  d_edge_phi_w_view            Per-edge predicted phi at layer-1 radius
/// @param[in]  d_edge_nodes_view            (src, dst) per edge
/// @param[in]  d_num_outgoing_edges_view    Per-node prefix sum (now offsets)
/// @param[in]  d_edge_links_view            Per-edge slot in dst's incoming
/// @param[out] d_num_neighbours_view        Accepted neighbour count per edge
/// @param[out] d_neighbours_view            Neighbour edge indices per edge
/// @param[out] d_reIndexer_view             Per-edge "kept" flag
/// @param[in,out] nConnectionsCounter       Global connection-count atomic
/// @param[in]  nEdges                       Number of edges
/// @tparam NMaxNei                          Compile-time max neighbours per edge
///                                          (use the runtime overload when
///                                          this isn't known statically)
///
template <unsigned int NMaxNei, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex, const unsigned int threadIndex,
    const barrier_t& barrier,
    const gbts_edge_matching_params& d_graph_building_params,
    const collection_types<float>::const_view& d_edge_exp_eta_view,
    const collection_types<float>::const_view& d_edge_curv_view,
    const collection_types<float>::const_view& d_edge_phi_z_view,
    const collection_types<float>::const_view& d_edge_phi_w_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<unsigned int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nEdges);

/// Runtime-`nMaxNei` overload for callers that can't pick a compile-time bound
/// (e.g. CLI-driven configurations). Same semantics as the templated form.
template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex, const unsigned int threadIndex,
    const barrier_t& barrier,
    const gbts_edge_matching_params& d_graph_building_params,
    const collection_types<float>::const_view& d_edge_exp_eta_view,
    const collection_types<float>::const_view& d_edge_curv_view,
    const collection_types<float>::const_view& d_edge_phi_z_view,
    const collection_types<float>::const_view& d_edge_phi_w_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<unsigned int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nEdges,
    const unsigned int nMaxNei);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_matching.ipp"
