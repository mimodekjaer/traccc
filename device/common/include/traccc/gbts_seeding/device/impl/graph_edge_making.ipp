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
    const unsigned int blockSize, const barrier_t& barrier, float* tau_min,
    float* tau_max, float* phi, float* r, float* z,
    const collection_types<int>::const_view& d_bin_pair_views_view,
    const collection_types<float>::const_view& d_bin_pair_dphi_view,
    const collection_types<float>::const_view& d_node_params_view,
    const gbts_graph_building_params& d_graph_building_params,
    unsigned int& nEdgesCounter,
    collection_types<int2>::view d_edge_nodes_view,
    collection_types<half4>::view d_edge_params_view,
    collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins) {

    const collection_types<int>::const_device d_bin_pair_views(
        d_bin_pair_views_view);
    const collection_types<float>::const_device d_bin_pair_dphi(
        d_bin_pair_dphi_view);
    const collection_types<float>::const_device d_node_params(
        d_node_params_view);
    collection_types<int2>::device d_edge_nodes(d_edge_nodes_view);
    collection_types<half4>::device d_edge_params(d_edge_params_view);
    collection_types<int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    const float deltaPhi = d_bin_pair_dphi[blockIndex];

    const unsigned int begin_bin1 = static_cast<unsigned int>(
        d_bin_pair_views[4u * blockIndex]);
    const unsigned int begin_bin2 = static_cast<unsigned int>(
        d_bin_pair_views[4u * blockIndex + 2u]);
    const unsigned int num_nodes1 =
        static_cast<unsigned int>(d_bin_pair_views[4u * blockIndex + 1u]) -
        begin_bin1;
    const unsigned int num_nodes2 =
        static_cast<unsigned int>(d_bin_pair_views[4u * blockIndex + 3u]) -
        begin_bin2;

    const float minDeltaRad = d_graph_building_params.minDeltaRadius;
    const float min_z0 = d_graph_building_params.min_z0;
    const float max_z0 = d_graph_building_params.max_z0;
    const float maxOuterRad = d_graph_building_params.maxOuterRadius;
    const float min_zU = d_graph_building_params.cut_zMinU;
    const float max_zU = d_graph_building_params.cut_zMaxU;
    const float max_kappa = d_graph_building_params.max_Kappa;
    const float low_Kappa_d0 = d_graph_building_params.low_Kappa_d0;
    const float high_Kappa_d0 = d_graph_building_params.high_Kappa_d0;

    for (unsigned int idx = threadIndex; idx < num_nodes1; idx += blockSize) {
        const unsigned int offset = 5u * (idx + begin_bin1);
        tau_min[idx] = d_node_params[offset];
        tau_max[idx] = d_node_params[offset + 1u];
        phi[idx] = d_node_params[offset + 2u];
        r[idx] = d_node_params[offset + 3u];
        z[idx] = d_node_params[offset + 4u];
    }

    barrier.blockBarrier();

    int last_n1 = 0;

    const float phi_bin_width =
        2.0f * traccc::device::gbts_pi_f / static_cast<float>(nPhiBins);

    for (unsigned int n2Idx = threadIndex; n2Idx < num_nodes2;
         n2Idx += blockSize) {
        int n1Idx = last_n1;
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
            if (phi[0] > max_phi1) {
                continue;
            }
            if (phi[num_nodes1 - 1u] < min_phi1) {
                if (phi[0] >
                    deltaPhi + phi_bin_width - traccc::device::gbts_pi_f) {
                    break;
                }
                continue;
            }
        } else {
            if (phi[0] < max_phi1) {
                n1Idx = 0;
            } else if (phi[num_nodes1 - 1u] < min_phi1) {
                continue;
            }
        }

        const float tau_min2 = d_node_params[o2];
        const float tau_max2 = d_node_params[o2 + 1u];
        const float r2 = d_node_params[o2 + 3u];
        const float z2 = d_node_params[o2 + 4u];

        for (; n1Idx < static_cast<int>(num_nodes1); n1Idx++) {
            const float phi1 = phi[n1Idx];

            if (!boundary) {
                if (phi1 > max_phi1) {
                    break;
                }
                if (phi1 < min_phi1) {
                    continue;
                }
                last_n1 = n1Idx;
            } else {
                if (phi1 > max_phi1 && phi1 < min_phi1) {
                    if (n1Idx < last_n1) {
                        n1Idx = last_n1 - 1;
                    }
                    continue;
                }
            }

            const float r1 = r[n1Idx];
            const float dr = r2 - r1;
            if (dr < minDeltaRad) {
                continue;
            }
            const float z1 = z[n1Idx];
            const float dz = z2 - z1;
            const float tau = dz / dr;
            const float ftau = fabsf(tau);

            if (ftau > 36.0f) {
                continue;
            }
            if ((ftau < tau_min2) || (ftau > tau_max2)) {
                continue;
            }
            if ((ftau < tau_min[n1Idx]) || (ftau > tau_max[n1Idx])) {
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
            if (boundary) {
                if (dphi < -traccc::device::gbts_pi_f) {
                    dphi += 2.0f * traccc::device::gbts_pi_f;
                } else if (dphi > traccc::device::gbts_pi_f) {
                    dphi -= 2.0f * traccc::device::gbts_pi_f;
                }
            }
            if (fabsf(dphi) > deltaPhi) {
                continue;
            }
            const float curv = dphi / dr;
            const float d0_for_max_curv =
                r1 * r2 * (fabsf(curv) - max_kappa);
            const float d0_max =
                (ftau < 4.0f) ? low_Kappa_d0 : high_Kappa_d0;
            if (d0_for_max_curv > d0_max) {
                continue;
            }
            const unsigned int nEdges =
                vecmem::device_atomic_ref<unsigned int>(nEdgesCounter)
                    .fetch_add(1u);
            if (nEdges < nMaxEdges) {
                const float exp_eta = sqrtf(1.0f + tau * tau) - tau;
                vecmem::device_atomic_ref<int>(
                    d_num_outgoing_edges[begin_bin1 +
                                         static_cast<unsigned int>(n1Idx)])
                    .fetch_add(1);
                d_edge_nodes[nEdges] =
                    make_int2(static_cast<int>(globalIdx2),
                              static_cast<int>(begin_bin1) + n1Idx);
                d_edge_params[nEdges] = make_half4(
                    exp_eta, curv, phi2 + curv * r2, phi1 + curv * r1);
            }
        }
    }
}

}  // namespace traccc::device
