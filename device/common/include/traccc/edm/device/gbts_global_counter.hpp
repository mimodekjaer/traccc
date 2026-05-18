/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

/// Collection of named device-side counters used by the GBTS seeding kernels.
///
/// Replaces the flat array of 10 unsigned integers that earlier versions of
/// the algorithm passed between kernels.  Each counter is documented at the
/// point of declaration so that it is clear which kernel writes/reads which
/// field.
struct gbts_global_counter {

    /// Number of edges produced by @c device::graph_edge_making.
    unsigned int m_nEdges;

    /// Number of edge-to-edge connections accumulated by
    /// @c device::graph_edge_matching.
    unsigned int m_nConnections;

    /// Number of edges that survive the re-indexing step in
    /// @c device::edge_re_indexing (i.e. edges with at least one connection).
    unsigned int m_nConnectedEdges;

    /// Active-edge counter for even CCA iterations (ping-pong buffer).
    unsigned int m_nActiveEdgesA;

    /// Active-edge counter for odd CCA iterations (ping-pong buffer).
    unsigned int m_nActiveEdgesB;

    /// Per-iteration block-completion counter used by CCA.
    unsigned int m_nCCABlockCounter;

    /// Total number of paths reachable from any terminus edge, accumulated
    /// by @c device::count_terminus_edges.
    unsigned int m_nPaths;

    /// Running size of the path store.  Initialised to the number of
    /// terminus edges by @c device::count_terminus_edges and grown further
    /// by @c device::fill_path_store; @c device::fit_segments reads it as
    /// the upper bound on path indices.
    unsigned int m_nPathStoreSize;

    /// Number of seed proposals produced by @c device::fit_segments.
    unsigned int m_nProps;

    /// Number of seed proposals rejected by @c device::reset_edge_bids.
    unsigned int m_nRejectedProps;

};  // struct gbts_global_counter

}  // namespace traccc::device
