/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_run_cca_iteration function
struct gbts_run_cca_iteration_payload {
    /// Number of edges in the compacted graph
    unsigned int nConnectedEdges;
    /// Maximum number of neighbours retained per edge
    unsigned int max_num_neighbours;
    /// Minimum path length required for an edge to be considered active
    unsigned char minLevel;
    /// Compacted graph from gbts_compress_graph
    vecmem::data::vector_view<const unsigned int> output_graph;
    /// In/out: per-edge level ping-pong buffer (2 * nConnectedEdges bytes)
    vecmem::data::vector_view<unsigned char> levels;
    /// In/out: per-edge active-flag (holds the next iter index, or -1
    /// once the edge is no longer active).
    vecmem::data::vector_view<char> active_edges;
    /// Output: per-edge (reachable-path count, terminus flag); must be zeroed
    /// before iteration 0
    vecmem::data::vector_view<short2> outgoing_paths;
    /// Iteration index (0-based): the CCA iteration in the level passes, the
    /// level-1 being processed in the count passes
    unsigned char iter;
    /// False: CCA level-relaxation pass. True: post-CCA path-count /
    /// terminus-flag pass (levels must be final)
    bool count_pass;
};

/// @brief One iteration of the cellular-automaton "longest path" relaxation,
/// or (count_pass) one level of the post-CCA path-count sweep.
///
/// Level pass: threads process the current active-edge list, propagate levels
/// along the compact graph into the opposite ping-pong half (selected by iter
/// parity), and update the per-edge active flag -- own-slot writes only.
///
/// Count pass: runs after the levels are final. Pass k processes exactly the
/// edges at level k+1: each sums its reachable paths from its level-k
/// children (counted in pass k-1, so the read crosses a kernel boundary) and
/// flags all its neighbours as non-terminus. The only cross-thread writes
/// store the single value -1 (idempotent), so the result is a pure function
/// of the graph -- no dependence on thread ordering.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id,
    const gbts_run_cca_iteration_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_run_cca_iteration.ipp"
