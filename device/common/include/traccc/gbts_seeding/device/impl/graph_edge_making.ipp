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

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_making(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize, const barrier_t& barrier, float* phi,
    float4* node_pack,
    const collection_types<unsigned int>::const_view& d_bin_pair_views_view,
    const collection_types<float>::const_view& d_bin_pair_dphi_view,
    const collection_types<float>::const_view& d_node_params_view,
    const gbts_edge_making_params& d_gbts_edge_making_params,
    unsigned int& nEdgesCounter,
    const collection_types<uint2>::view d_edge_nodes_view,
    const collection_types<gbts_edge_real_t>::view d_edge_exp_eta_view,
    const collection_types<gbts_edge_real_t>::view d_edge_curv_view,
    const collection_types<gbts_edge_real_t>::view d_edge_phi_z_view,
    const collection_types<gbts_edge_real_t>::view d_edge_phi_w_view,
    const collection_types<unsigned int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins) {

    const collection_types<unsigned int>::const_device d_bin_pair_views(
        d_bin_pair_views_view);
    const collection_types<float>::const_device d_bin_pair_dphi(
        d_bin_pair_dphi_view);
    const collection_types<float>::const_device d_node_params(
        d_node_params_view);
    collection_types<uint2>::device d_edge_nodes(d_edge_nodes_view);
    collection_types<gbts_edge_real_t>::device d_edge_exp_eta(
        d_edge_exp_eta_view);
    collection_types<gbts_edge_real_t>::device d_edge_curv(d_edge_curv_view);
    collection_types<gbts_edge_real_t>::device d_edge_phi_z(d_edge_phi_z_view);
    collection_types<gbts_edge_real_t>::device d_edge_phi_w(d_edge_phi_w_view);
    collection_types<unsigned int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    const float deltaPhi = d_bin_pair_dphi[blockIndex];

    const unsigned int begin_bin1 = d_bin_pair_views[4u * blockIndex];
    const unsigned int begin_bin2 = d_bin_pair_views[4u * blockIndex + 2u];
    const unsigned int num_nodes1 = d_bin_pair_views[4u * blockIndex + 1u] -
        begin_bin1;
    const unsigned int num_nodes2 = d_bin_pair_views[4u * blockIndex + 3u] -
        begin_bin2;

    const float minDeltaRad = d_gbts_edge_making_params.minDeltaRadius;
    const float min_z0 = d_gbts_edge_making_params.min_z0;
    const float max_z0 = d_gbts_edge_making_params.max_z0;
    const float maxOuterRad = d_gbts_edge_making_params.maxOuterRadius;
    const float min_zU = d_gbts_edge_making_params.cut_zMinU;
    const float max_zU = d_gbts_edge_making_params.cut_zMaxU;
    const float max_kappa = d_gbts_edge_making_params.max_Kappa;
    const float low_Kappa_d0 = d_gbts_edge_making_params.low_Kappa_d0;
    const float high_Kappa_d0 = d_gbts_edge_making_params.high_Kappa_d0;

    for (unsigned int idx = threadIndex; idx < num_nodes1; idx += blockSize) {
        const unsigned int offset = 5u * (idx + begin_bin1);
        const float tau_min_v = d_node_params[offset];
        const float tau_max_v = d_node_params[offset + 1u];
        phi[idx] = d_node_params[offset + 2u];
        const float r_v = d_node_params[offset + 3u];
        const float z_v = d_node_params[offset + 4u];
        node_pack[idx] = make_float4(r_v, z_v, tau_min_v, tau_max_v);
    }

    barrier.blockBarrier();

    // Hoist loop-invariants out of the n2 loop.
    const float phi_front = phi[0];
    const float phi_back = phi[num_nodes1 - 1u];
    const float phi_bin_width =
        2.0f * traccc::device::gbts_pi_f / static_cast<float>(nPhiBins);
    const float break_threshold =
        deltaPhi + phi_bin_width - traccc::device::gbts_pi_f;

    unsigned int last_n1 = 0u;

    for (unsigned int n2Idx = threadIndex; n2Idx < num_nodes2;
         n2Idx += blockSize) {
        unsigned int n1Idx = last_n1;
        const unsigned int globalIdx2 = begin_bin2 + n2Idx;
        const unsigned int o2 = 5u * globalIdx2;
        const float phi2 = d_node_params[o2 + 2u];

        float min_phi1 = phi2 - deltaPhi;
        float max_phi1 = phi2 + deltaPhi;

        if (min_phi1 < -traccc::device::gbts_pi_f) {
            min_phi1 += 2.0f * traccc::device::gbts_pi_f;
        }
        if (max_phi1 > traccc::device::gbts_pi_f) {
            max_phi1 -= 2.0f * traccc::device::gbts_pi_f;
        }
        const bool boundary = max_phi1 < min_phi1;

        max_phi1 += phi_bin_width;
        min_phi1 -= phi_bin_width;

        if (!boundary) {
            if (phi_front > max_phi1) {
                continue;
            }
            if (phi_back < min_phi1) {
                if (phi_front > break_threshold) {
                    break;
                }
                continue;
            }
        } else {
            if (phi_front < max_phi1) {
                n1Idx = 0;
            } else if (phi_back < min_phi1) {
                continue;
            }
        }

        const float tau_min2 = d_node_params[o2];
        const float tau_max2 = d_node_params[o2 + 1u];
        const float r2 = d_node_params[o2 + 3u];
        const float z2 = d_node_params[o2 + 4u];

        // boundary is loop-invariant for the inner loop; hoisting saves a
        // per-iteration branch and lets the compiler specialise the dphi
        // wrap-around.
        if (!boundary) {
            for (; n1Idx < num_nodes1; n1Idx++) {
                const float phi1 = phi[n1Idx];
                if (phi1 > max_phi1) {
                    break;
                }
                if (phi1 < min_phi1) {
                    continue;
                }
                last_n1 = n1Idx;

                const float4 p = node_pack[n1Idx];
                const float r1 = p.x;
                const float dr = r2 - r1;
                if (dr < minDeltaRad) {
                    continue;
                }
                const float z1 = p.y;
                const float dz = z2 - z1;
                const float inv_dr = 1.0f / dr;
                const float tau = dz * inv_dr;
                const float ftau = fabsf(tau);

                if (ftau > 36.0f) { // ~4.3 eta detector acceptance
                    continue;
                }
                if ((ftau < tau_min2) || (ftau > tau_max2)) {
                    continue;
                }
                if ((ftau < p.z) || (ftau > p.w)) {
                    continue;
                }
                const float z0 = z1 - r1 * tau;
                if ((z0 < min_z0) || (z0 > max_z0)) {
                    continue;
                }
                const float zouter = z0 + maxOuterRad * tau;
                if (zouter < min_zU || zouter > max_zU) {
                    continue;
                }
                const float dphi = phi2 - phi1;
                if (fabsf(dphi) > deltaPhi) {
                    continue;
                }
                const float curv = dphi * inv_dr;
                const float d0_for_max_curv =
                    r1 * r2 * (fabsf(curv) - max_kappa);
                const float d0_max =
                    (ftau < 4.0f) ? low_Kappa_d0 : high_Kappa_d0;
                if (d0_for_max_curv > d0_max) {
                    continue;
                }
                const unsigned int nEdges =
                    vecmem::device_atomic_ref<unsigned int>(nEdgesCounter).fetch_add(1u);
                if (nEdges < nMaxEdges) {
                    const float exp_eta = sqrtf(1.0f + tau * tau) - tau;
                    vecmem::device_atomic_ref<unsigned int>(
                        d_num_outgoing_edges[begin_bin1 + n1Idx])
                        .fetch_add(1u);
                    d_edge_nodes[nEdges] =
                        make_uint2(globalIdx2, begin_bin1 + n1Idx);
                    d_edge_exp_eta[nEdges] = gbts_to_real(exp_eta);
                    d_edge_curv[nEdges]    = gbts_to_real(curv);
                    d_edge_phi_z[nEdges]   = gbts_to_real(phi2 + curv * r2);
                    d_edge_phi_w[nEdges]   = gbts_to_real(phi1 + curv * r1);
                }
            }
        } else {
            for (; n1Idx < num_nodes1; n1Idx++) {
                const float phi1 = phi[n1Idx];
                if (phi1 > max_phi1 && phi1 < min_phi1) {
                    if (n1Idx < last_n1) {
                        n1Idx = last_n1 - 1;
                    }
                    continue;
                }

                const float4 p = node_pack[n1Idx];
                const float r1 = p.x;
                const float dr = r2 - r1;
                if (dr < minDeltaRad) {
                    continue;
                }
                const float z1 = p.y;
                const float dz = z2 - z1;
                const float inv_dr = 1.0f / dr;
                const float tau = dz * inv_dr;
                const float ftau = fabsf(tau);

                if (ftau > 36.0f) {
                    continue;
                }
                if ((ftau < tau_min2) || (ftau > tau_max2)) {
                    continue;
                }
                if ((ftau < p.z) || (ftau > p.w)) {
                    continue;
                }
                const float z0 = z1 - r1 * tau;
                if ((z0 < min_z0) || (z0 > max_z0)) {
                    continue;
                }
                const float zouter = z0 + maxOuterRad * tau;
                if (zouter < min_zU || zouter > max_zU) {
                    continue;
                }
                float dphi = phi2 - phi1;
                if (dphi < -traccc::device::gbts_pi_f) {
                    dphi += 2.0f * traccc::device::gbts_pi_f;
                } else if (dphi > traccc::device::gbts_pi_f) {
                    dphi -= 2.0f * traccc::device::gbts_pi_f;
                }
                if (fabsf(dphi) > deltaPhi) {
                    continue;
                }
                const float curv = dphi * inv_dr;
                const float d0_for_max_curv =
                    r1 * r2 * (fabsf(curv) - max_kappa);
                const float d0_max =
                    (ftau < 4.0f) ? low_Kappa_d0 : high_Kappa_d0;
                if (d0_for_max_curv > d0_max) {
                    continue;
                }
                const unsigned int nEdges = 
                    vecmem::device_atomic_ref<unsigned int>(nEdgesCounter).fetch_add(1u);
                if (nEdges < nMaxEdges) {
                    const float exp_eta = sqrtf(1.0f + tau * tau) - tau;
                    vecmem::device_atomic_ref<unsigned int>(
                        d_num_outgoing_edges[begin_bin1 + n1Idx])
                        .fetch_add(1u);
                    d_edge_nodes[nEdges] =
                        make_uint2(globalIdx2, begin_bin1 + n1Idx);
                    d_edge_exp_eta[nEdges] = gbts_to_real(exp_eta);
                    d_edge_curv[nEdges]    = gbts_to_real(curv);
                    d_edge_phi_z[nEdges]   = gbts_to_real(phi2 + curv * r2);
                    d_edge_phi_w[nEdges]   = gbts_to_real(phi1 + curv * r1);
                }
            }
        }
    }
}

}  // namespace traccc::device
