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

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cmath>

namespace traccc::device {

namespace detail {

// Single source of truth for the matcher body. Both the compile-time-bound
// template entry point and the runtime-bound fallback forward here; the
// `n_max_nei` argument is either a literal (constant-folded by the compiler
// after inlining) or a runtime value depending on which caller invoked us.
//
// SoA layout: each per-edge field lives in its own `float` buffer, so the
// inner loop can issue per-cut 4-byte loads and bail on the first failing
// cut without paying for fields it never reads. This is the portable
// replacement for the original packed-half4 layout.
template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_matching_body(
    const global_index_t globalIndex, const unsigned int /*threadIndex*/,
    const barrier_t& /*barrier*/,
    const gbts_edge_matching_params& d_graph_building_params,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_exp_eta_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_curv_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_z_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_w_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<unsigned int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nEdges,
    const unsigned int n_max_nei) {

    const collection_types<gbts_edge_real_t>::const_device d_edge_exp_eta(
        d_edge_exp_eta_view);
    const collection_types<gbts_edge_real_t>::const_device d_edge_curv(
        d_edge_curv_view);
    const collection_types<gbts_edge_real_t>::const_device d_edge_phi_z(
        d_edge_phi_z_view);
    const collection_types<gbts_edge_real_t>::const_device d_edge_phi_w(
        d_edge_phi_w_view);
    const collection_types<uint2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<unsigned int>::const_device d_num_outgoing_edges(
        d_num_outgoing_edges_view);
    const collection_types<unsigned int>::const_device d_edge_links(
        d_edge_links_view);
    collection_types<unsigned char>::device d_num_neighbours(
        d_num_neighbours_view);
    collection_types<unsigned int>::device d_neighbours(d_neighbours_view);
    collection_types<unsigned int>::device d_reIndexer(d_reIndexer_view);

    const float cut_dphi_max = d_graph_building_params.cut_dphi_max;
    const float cut_dcurv_max = d_graph_building_params.cut_dcurv_max;
    const float cut_tau_ratio_max = d_graph_building_params.cut_tau_ratio_max;

    if (globalIndex >= nEdges) {
        return;
    }

    unsigned char num_nei = 0;

    const unsigned int sharedNode = d_edge_nodes[globalIndex].x;
    const unsigned int link_begin = d_num_outgoing_edges[sharedNode];
    const unsigned int nLinks =
        d_num_outgoing_edges[sharedNode + 1u] - link_begin;

    if (nLinks != 0u) {
        // Self-edge fields. The candidate side never reads `.w` for the self
        // edge and the self side never reads `.w` either, so only three of
        // the four self fields are loaded.
        const float uat_2 = 1.0f / gbts_to_float(d_edge_exp_eta[globalIndex]);
        const float curv2 = gbts_to_float(d_edge_curv[globalIndex]);
        const float Phi2  = gbts_to_float(d_edge_phi_z[globalIndex]);

        const unsigned int nei_pos = n_max_nei * globalIndex;

        // Software-pipelined prefetch of the first field touched by the cuts.
        // The remaining per-candidate fields are loaded lazily after each
        // short-circuit early-out, so a rejected candidate only ever pays
        // for the one exp_eta load.
        unsigned int next_edge2_idx = d_edge_links[link_begin];
        gbts_edge_real_t next_exp_eta = d_edge_exp_eta[next_edge2_idx];

        for (unsigned int k = 0u; k < nLinks; ++k) {
            if (num_nei >= n_max_nei) {
                break;
            }

            const unsigned int edge2_idx = next_edge2_idx;
            const float exp_eta_2 = gbts_to_float(next_exp_eta);

            if (k + 1u < nLinks) {
                next_edge2_idx = d_edge_links[link_begin + k + 1u];
                next_exp_eta   = d_edge_exp_eta[next_edge2_idx];
            }

            const float tau_ratio = exp_eta_2 * uat_2 - 1.0f;
            if (fabsf(tau_ratio) > cut_tau_ratio_max) {
                continue;
            }

            const float curv_2 = gbts_to_float(d_edge_curv[edge2_idx]);
            const float dcurv  = curv_2 - curv2;
            if (fabsf(dcurv) > cut_dcurv_max) {
                continue;
            }

            const float phi_w_2 = gbts_to_float(d_edge_phi_w[edge2_idx]);
            float dPhi = Phi2 - phi_w_2;
            dPhi -= traccc::device::gbts_two_pi_f *
                    static_cast<float>(dPhi > traccc::device::gbts_pi_f);
            dPhi += traccc::device::gbts_two_pi_f *
                    static_cast<float>(dPhi < -traccc::device::gbts_pi_f);
            if (fabsf(dPhi) > cut_dphi_max) {
                continue;
            }

            d_neighbours[nei_pos + num_nei] = edge2_idx;
            d_reIndexer[edge2_idx] = 1u;
            ++num_nei;
        }
    }

    d_num_neighbours[globalIndex] = num_nei;
    if (num_nei != 0) {
        d_reIndexer[globalIndex] = 1u;
        vecmem::device_atomic_ref<unsigned int>(nConnectionsCounter)
            .fetch_add(static_cast<unsigned int>(num_nei));
    }
}

}  // namespace detail

// Compile-time bound on neighbours: callers pass NMaxNei as a non-type
// template parameter, and the body's `num_nei >= n_max_nei` check plus
// the `nei_pos = n_max_nei * globalIndex` multiply constant-fold.
template <unsigned int NMaxNei, concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex, const unsigned int threadIndex,
    const barrier_t& barrier,
    const gbts_edge_matching_params& d_graph_building_params,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_exp_eta_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_curv_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_z_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_w_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<unsigned int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nEdges) {

    detail::graph_edge_matching_body(
        globalIndex, threadIndex, barrier, d_graph_building_params,
        d_edge_exp_eta_view, d_edge_curv_view, d_edge_phi_z_view,
        d_edge_phi_w_view, d_edge_nodes_view, d_num_outgoing_edges_view,
        d_edge_links_view, d_num_neighbours_view, d_neighbours_view,
        d_reIndexer_view, nConnectionsCounter, nEdges, NMaxNei);
}

// Runtime-bound fallback used when the CLI passes a non-default
// max_num_neighbours. Same body, but `n_max_nei` is a runtime parameter so
// the compiler can't fold the bound.
template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const global_index_t globalIndex, const unsigned int threadIndex,
    const barrier_t& barrier,
    const gbts_edge_matching_params& d_graph_building_params,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_exp_eta_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_curv_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_z_view,
    const collection_types<gbts_edge_real_t>::const_view& d_edge_phi_w_view,
    const collection_types<uint2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned int>::const_view& d_num_outgoing_edges_view,
    const collection_types<unsigned int>::const_view& d_edge_links_view,
    const collection_types<unsigned char>::view& d_num_neighbours_view,
    const collection_types<unsigned int>::view& d_neighbours_view,
    const collection_types<unsigned int>::view& d_reIndexer_view,
    unsigned int& nConnectionsCounter, const unsigned int nEdges,
    const unsigned int nMaxNei) {

    detail::graph_edge_matching_body(
        globalIndex, threadIndex, barrier, d_graph_building_params,
        d_edge_exp_eta_view, d_edge_curv_view, d_edge_phi_z_view,
        d_edge_phi_w_view, d_edge_nodes_view, d_num_outgoing_edges_view,
        d_edge_links_view, d_num_neighbours_view, d_neighbours_view,
        d_reIndexer_view, nConnectionsCounter, nEdges, nMaxNei);
}

}  // namespace traccc::device
