/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/utils/trigonometric_helpers.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

// Compile-time upper bound on nMaxNei (= cfg.max_num_neighbours, default 10),
// used to size the thread-local top-K neighbour-selection buffers below. Must
// be >= any configured max_num_neighbours.
constexpr unsigned int gbts_match_max_nei = 16u;

// For each edge, find up to nMaxNei compatible neighbour edges sharing its
// outer node, recording them and marking both edges as "kept". The edge
// parameters are read from the packed float4 buffer ([exp_eta, curv, phi_z,
// phi_w]) and the cuts are evaluated in float.
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_match_graph_edges(
    const thread_id_t& thread_id,
    const gbts_match_graph_edges_payload& payload) {

    const vecmem::device_vector<const float4> d_edge_params(
        payload.edge_params);
    const vecmem::device_vector<const uint2> d_edge_nodes(payload.edge_nodes);
    const vecmem::device_vector<const unsigned int> d_num_outgoing_edges(
        payload.num_outgoing_edges);
    const vecmem::device_vector<const unsigned int> d_edge_links(
        payload.edge_links);
    vecmem::device_vector<unsigned char> d_num_neighbours(
        payload.num_neighbours);
    vecmem::device_vector<unsigned int> d_neighbours(payload.neighbours);
    vecmem::device_vector<unsigned int> d_inclusive_kept(
        payload.inclusive_kept);

    const float cut_dphi_max =
        payload.gbts_match_graph_edges_params.cut_dphi_max;
    const float cut_dcurv_max =
        payload.gbts_match_graph_edges_params.cut_dcurv_max;
    const float cut_tau_ratio_max =
        payload.gbts_match_graph_edges_params.cut_tau_ratio_max;

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int globalIndex = globalIdx; globalIndex < payload.nEdges;
         globalIndex += blockDimX * gridDimX) {

        const unsigned int sharedNode = d_edge_nodes[globalIndex].x;

        const unsigned int link_begin = d_num_outgoing_edges[sharedNode];
        // the number of edges leaving the sharedNode
        const unsigned int nLinks =
            d_num_outgoing_edges[sharedNode + 1u] - link_begin;
        if (nLinks == 0u) {
            // num_neighbours has no host-side zero-init; every edge's slot is
            // defined here or via num_nei below.
            d_num_neighbours[globalIndex] = 0;
            continue;
        }

        const float4 params1 = d_edge_params[globalIndex];  // [exp_eta, curv,
                                                            //  phi_z, phi_w]
        const float uat_2 = 1.0f / params1.x;
        const float Phi2 = params1.z;
        const float curv2 = params1.y;

        const unsigned int nei_pos = payload.nMaxNei * globalIndex;

        // Deterministic neighbour selection: keep the K candidates with the
        // smallest neighbour outer-node index (edge_nodes[edge2_idx].x) -- a
        // unique, run-to-run-stable key per candidate. This replaces the old
        // "first nMaxNei in (race-ordered) edge_links order" early break, so the
        // kept neighbour SET (and hence the kept-edge set and the seeds) is
        // reproducible. Downstream consumers use only the set, not its order.
        const unsigned int K = (payload.nMaxNei < gbts_match_max_nei)
                                   ? payload.nMaxNei
                                   : gbts_match_max_nei;
        unsigned int best_val[gbts_match_max_nei];  // kept edge2_idx
        unsigned int best_key[gbts_match_max_nei];  // its outer-node key
        unsigned char num_nei = 0;
        unsigned int max_key = 0u;  // largest kept key (valid once num_nei == K)
        unsigned int max_pos = 0u;  // slot holding max_key

        for (unsigned int k = 0u; k < nLinks; k++) {

            const unsigned int edge2_idx = d_edge_links[link_begin + k];

            if (num_nei == K) {
                // Set full: a candidate can only enter if its key beats the
                // current largest -- check the cheap key first and skip the
                // param load + cuts otherwise.
                const unsigned int key = d_edge_nodes[edge2_idx].x;
                if (key >= max_key) {
                    continue;
                }
                const float4 params2 = d_edge_params[edge2_idx];
                const float tau_ratio = params2.x * uat_2 - 1.0f;
                if (math::fabs(tau_ratio) > cut_tau_ratio_max) {
                    continue;
                }
                const float dPhi = traccc::detail::wrap_phi(Phi2 - params2.w);
                if (math::fabs(dPhi) > cut_dphi_max) {
                    continue;
                }
                const float dcurv = curv2 - params2.y;
                if (math::fabs(dcurv) > cut_dcurv_max) {
                    continue;
                }
                // Evict the current largest, then recompute it over the set.
                best_val[max_pos] = edge2_idx;
                best_key[max_pos] = key;
                max_key = best_key[0];
                max_pos = 0u;
                for (unsigned int j = 1u; j < K; j++) {
                    if (best_key[j] > max_key) {
                        max_key = best_key[j];
                        max_pos = j;
                    }
                }
            } else {
                const float4 params2 = d_edge_params[edge2_idx];
                const float tau_ratio = params2.x * uat_2 - 1.0f;
                if (math::fabs(tau_ratio) > cut_tau_ratio_max) {  // bad match
                    continue;
                }
                const float dPhi = traccc::detail::wrap_phi(Phi2 - params2.w);
                if (math::fabs(dPhi) > cut_dphi_max) {
                    continue;
                }
                const float dcurv = curv2 - params2.y;
                if (math::fabs(dcurv) > cut_dcurv_max) {
                    continue;
                }
                best_val[num_nei] = edge2_idx;
                best_key[num_nei] = d_edge_nodes[edge2_idx].x;
                ++num_nei;
                if (num_nei == K) {  // set just filled: seal the max
                    max_key = best_key[0];
                    max_pos = 0u;
                    for (unsigned int j = 1u; j < K; j++) {
                        if (best_key[j] > max_key) {
                            max_key = best_key[j];
                            max_pos = j;
                        }
                    }
                }
            }
        }

        // Mark only the final survivors as kept (eviction means an
        // inserted-then-evicted candidate must not stay marked).
        for (unsigned char i = 0; i < num_nei; i++) {
            d_neighbours[nei_pos + i] = best_val[i];
            d_inclusive_kept[best_val[i]] = 1u;
        }

        d_num_neighbours[globalIndex] = num_nei;

        if (num_nei != 0) {
            d_inclusive_kept[globalIndex] = 1u;
        }
    }
}

}  // namespace traccc::device
