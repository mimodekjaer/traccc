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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_sort_nodes
/// function
struct gbts_sort_nodes_payload {
    /// Total number of GBTS nodes
    unsigned int nNodes;
    /// Number of phi bins per eta slice
    unsigned int nPhiBins;
    /// Layer-ordered (x, y, z, r) per spacepoint
    vecmem::data::vector_view<const float4> sp_params;
    /// Eta-bin index per node
    vecmem::data::vector_view<const unsigned int> node_eta_index;
    /// Phi-bin index per node
    vecmem::data::vector_view<const unsigned int> node_phi_index;
    /// In/out: per-(eta, phi) write cursor (atomically advanced)
    vecmem::data::vector_view<unsigned int> phi_cusums;
    /// Output: per-node (tau_min, tau_max, r, z), written in sorted order
    vecmem::data::vector_view<float4> node_params;
    /// Output: per-node phi, written in sorted order
    vecmem::data::vector_view<float> node_phi;
    /// Output: per-sorted-slot original spacepoint (collection) index
    vecmem::data::vector_view<unsigned int> node_index;
    /// Output: raw spacepoint (x, y, z, w), written in sorted order; the
    /// downstream stages address spacepoints by sorted-node slot
    vecmem::data::vector_view<float4> node_xyzw;
    /// Map from layer-ordered SP index to the original SP slot
    vecmem::data::vector_view<const unsigned int> original_sp_idx;
    /// Optional tau lookup table (used iff gbts_sort_nodes_params.useTauLUT)
    vecmem::data::vector_view<const float> tau_lut;
    /// Tau-prediction cuts read by @c device::gbts_sort_nodes
    traccc::gbts_sort_nodes_params gbts_sort_nodes_params;
};

/// Per-node geometry / kinematic tuple computed from the raw spacepoint. Used
/// both by the scatter and by the deterministic gather when (re)building the
/// sorted node arrays.
struct gbts_node_geometry {
    /// (min_tau, max_tau, r, z)
    traccc::float4 params;
    /// Azimuthal angle of the node
    float phi;
    /// Original SP slot this node maps back to (= original_sp_idx[globalIndex])
    unsigned int node_index;
    /// Raw spacepoint (x, y, z, w) of the node
    traccc::float4 sp;
};

/// @brief Flattened (eta, phi) histogram bin index for a single node.
///
/// @param[in] globalIndex Layer-ordered node index
/// @param[in] payload     The global memory payload
/// @return histo_bin = node_phi_index + nPhiBins * node_eta_index
///
TRACCC_HOST_DEVICE inline unsigned int gbts_compute_histo_bin(
    unsigned int globalIndex, const gbts_sort_nodes_payload& payload);

/// @brief Compute the geometry / kinematic tuple for a single node.
///
/// @param[in] globalIndex Layer-ordered node index
/// @param[in] payload     The global memory payload
/// @return the (params, phi, node_index) tuple for the node
///
TRACCC_HOST_DEVICE inline gbts_node_geometry gbts_compute_node_geometry(
    unsigned int globalIndex, const gbts_sort_nodes_payload& payload);

/// @brief Fill the composite sort key and identity permutation for the
/// deterministic node sort.
///
/// One thread per node writes @c keys[i] = (histo_bin << 32) | orderable(phi)
/// and @c perm[i] = i. A stable sort of @p perm_view by @p keys_view then
/// orders the nodes by (eta, phi) bin independently of the race-ordered
/// spacepoint collection upstream, which also makes the make_graph_edges phi
/// sliding window monotone (hence its edge set reproducible).
///
/// @param[in]  thread_id Thread identifier for the kernel launch
/// @param[in]  payload   The global memory payload
/// @param[out] keys_view Per-node composite (histo_bin, orderable phi) key
/// @param[out] perm_view Identity permutation to be sorted by @p keys_view
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_fill_sort_keys(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload,
    vecmem::data::vector_view<std::uint64_t> keys_view,
    vecmem::data::vector_view<unsigned int> perm_view);

/// @brief Gather the per-node geometry tuple into the (eta, phi)-sorted slots.
///
/// One thread per sorted slot reads its source node from the sorted permutation,
/// recomputes the geometry tuple (cheaper than moving the 24-byte tuple), and
/// writes the (params, phi, node_index, xyzw) into the slot. The within-bin
/// order is the canonical ascending phi, so the output is identical run to run.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
/// @param[in] perm_view Permutation sorted by @c gbts_fill_sort_keys' keys
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_gather_sorted_nodes(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload,
    vecmem::data::vector_view<unsigned int> perm_view);

/// @brief Scatter nodes into (eta, phi)-sorted slots and pack their geometry
/// tuple.
///
/// Each thread picks one node, atomically increments its (eta, phi) write
/// cursor, packs the geometry / kinematic tuple into the node parameters at
/// that slot, and stores the original SP index in the node index map.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_sort_nodes(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_sort_nodes.ipp"
