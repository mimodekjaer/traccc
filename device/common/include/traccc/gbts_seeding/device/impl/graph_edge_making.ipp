/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
 #pragma once

 // Local include(s).
 #include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/edm/container.hpp"


namespace traccc::device {

// I have to check how to change this

TRACCC_HOST_DEVICE
inline void graph_edge_making_kernel(
    const collection_types<int>::const_view d_bin_pair_views_view,
    const collection_types<float>::const_view d_bin_pair_dphi_view, // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
    const collection_types<float>::const_view d_node_params_view,
    const gbts_graph_building_params d_graph_building_params,
    const collection_types<unsigned int>::view d_counters_view, 
    const collection_types<int2>::view d_edge_nodes_view, const collection_types<half4>::view d_edge_params_view,
    const collection_types<int>::view d_num_outgoing_edges_view, 
    const unsigned int nMaxEdges,
    const unsigned int nPhiBins) {

    const collection_types<int>::const_device d_bin_pair_views(d_bin_pair_views_view); // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
    const collection_types<float>::const_device d_bin_pair_dphi(d_bin_pair_dphi_view);
    const collection_types<float>::const_device d_node_params(d_node_params_view);
    collection_types<unsigned int>::device d_counters(d_counters_view);
    collection_types<int2>::device d_edge_nodes(d_edge_nodes_view);
    collection_types<half4>::device d_edge_params(d_edge_params_view);
    collection_types<int>::device d_num_outgoing_edges(d_num_outgoing_edges_view);

    __shared__ unsigned int begin_bin1;
    __shared__ unsigned int begin_bin2;
    __shared__ unsigned int num_nodes1;
    __shared__ unsigned int num_nodes2;
    __shared__ float deltaPhi;

    __shared__ float tau_min[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float tau_max[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float r[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float z[traccc::device::gbts_consts::node_buffer_length];

    if (threadIdx.x == 0) {
        deltaPhi = d_bin_pair_dphi[blockIdx.x];

        begin_bin1 = static_cast<unsigned int>(d_bin_pair_views[4*blockIdx.x]); // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
        begin_bin2 = static_cast<unsigned int>(d_bin_pair_views[4*blockIdx.x + 2]); // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
        num_nodes1 = static_cast<unsigned int>(d_bin_pair_views[4*blockIdx.x + 1]) - begin_bin1; // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer
        num_nodes2 = static_cast<unsigned int>(d_bin_pair_views[4*blockIdx.x + 3]) - begin_bin2; // Here the type changed as a quick fix to avoid the issue with the bin_pair_views buffer

        minDeltaRad = d_graph_building_params.minDeltaRadius;
        min_z0 = d_graph_building_params.min_z0;
        max_z0 = d_graph_building_params.max_z0;
        maxOuterRad = d_graph_building_params.maxOuterRadius;
        min_zU = d_graph_building_params.cut_zMinU;
        max_zU = d_graph_building_params.cut_zMaxU;
        max_kappa = d_graph_building_params.max_Kappa;
        low_Kappa_d0 = d_graph_building_params.low_Kappa_d0;
        high_Kappa_d0 = d_graph_building_params.high_Kappa_d0;
    }

    barrier().blockBarrier();
    for (int idx = static_cast<int>(threadIdx.x); idx < static_cast<int>(num_nodes1); idx += static_cast<int>(blockDim.x)) {
        // loading a chunk of nodes1 into shared mem buffers
        int offset = 5 * (idx + static_cast<int>(begin_bin1));
        tau_min[idx] = d_node_params[static_cast<unsigned int>(offset)];
        tau_max[idx] = d_node_params[static_cast<unsigned int>(offset + 1)];
        phi[idx] = d_node_params[static_cast<unsigned int>(offset + 2)];
        r[idx] = d_node_params[static_cast<unsigned int>(offset + 3)];
        z[idx] = d_node_params[static_cast<unsigned int>(offset + 4)];
    }

    barrier().blockBarrier();

    int last_n1 = 0;  // initial value for the sliding window

    float phi_bin_width = 2.0f * CUDART_PI_F / static_cast<float>(nPhiBins);

    for (int n2Idx = static_cast<int>(threadIdx.x); n2Idx < static_cast<int>(num_nodes2); n2Idx += static_cast<int>(blockDim.x)) {

        int n1Idx = last_n1;

        int globalIdx2 = static_cast<int>(begin_bin2) + n2Idx;
        int o2 = 5 * globalIdx2;

        float phi2 = d_node_params[static_cast<unsigned int>(2 + o2)];

        float min_phi1 = phi2 - deltaPhi;
        float max_phi1 = phi2 + deltaPhi;

        if (min_phi1 < -CUDART_PI_F) {
            min_phi1 += 2.0f * CUDART_PI_F;
        }
        if (max_phi1 > CUDART_PI_F) {
            max_phi1 -= 2.0f * CUDART_PI_F;
        }
        bool boundary = max_phi1 < min_phi1;  // +/- pi wraparound

        // expand over nearest bin boundary
        max_phi1 += phi_bin_width;
        min_phi1 -= phi_bin_width;

        if (!boundary) {
            if (phi[0] > max_phi1) {
                continue;
            }
            if (phi[num_nodes1 - 1] < min_phi1) {
                // if bin1 can't be part of a wraparound
                // from a high-phi node skip it
                if (phi[0] > deltaPhi + phi_bin_width - CUDART_PI_F) {
                    break;
                }
                continue;
            }
        } else {
            if (phi[0] < max_phi1) {
                // if not to large for lower wraparound don't skip it
                n1Idx = 0;
            } else if (phi[num_nodes1 - 1] < min_phi1) {
                continue;
            }
        }

        float tau_min2 = d_node_params[static_cast<unsigned int>(o2)];
        float tau_max2 = d_node_params[static_cast<unsigned int>(1 + o2)];
        float r2 = d_node_params[static_cast<unsigned int>(3 + o2)];
        float z2 = d_node_params[static_cast<unsigned int>(4 + o2)];

        for (; n1Idx < static_cast<int>(num_nodes1); n1Idx++) {
            float phi1 = phi[n1Idx];

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
                    // skip to high wraparound after the lower part is done
                    if (n1Idx < last_n1) {
                        n1Idx = last_n1 - 1;
                    }
                    continue;
                }
            }

            float r1 = r[n1Idx];
            float dr = r2 - r1;

            if (dr < minDeltaRad) {
                continue;
            }
            float z1 = z[n1Idx];
            float dz = z2 - z1;
            float tau = dz / dr;
            float ftau = fabsf(tau);

            if (ftau > 36.0f) {
                continue;  // detector acceptance
            }
            if ((ftau < tau_min2) || (ftau > tau_max2)) {
                continue;
            }
            if ((ftau < tau_min[n1Idx]) || (ftau > tau_max[n1Idx])) {
                continue;
            }
            // RZ doublet filter cuts
            float z0 = z1 - r1 * tau;
            if ((z0 < min_z0) || (z0 > max_z0)) {
                continue;
            }
            float zouter = z0 + maxOuterRad * tau;

            if (zouter < min_zU || zouter > max_zU) {
                continue;
            }
            float dphi = phi2 - phi1;
            if (boundary) {
                if (dphi < -CUDART_PI_F)
                    dphi += 2.0f * CUDART_PI_F;
                else if (dphi > CUDART_PI_F)
                    dphi -= 2.0f * CUDART_PI_F;
            }

            // needed for sliding phi window consistancy
            if (fabsf(dphi) > deltaPhi) {
                continue;
            }
            float curv = dphi / dr;
            float d0_for_max_curv = r1 * r2 * (fabsf(curv) - max_kappa);
            float d0_max = (ftau < 4.0f) ? low_Kappa_d0 : high_Kappa_d0;
            if (d0_for_max_curv > d0_max) {
                continue;
            }
            unsigned int nEdges = vecmem::device_atomic_ref<unsigned int>(d_counters[0]).fetch_add(1);
            if (nEdges < nMaxEdges) {
                float16 exp_eta = static_cast<float16>(sqrtf(1 + tau * tau) - tau);
                // edge linking order is inside->out
                vecmem::device_atomic_ref<int>(d_num_outgoing_edges[begin_bin1 + static_cast<unsigned int>(n1Idx)]).fetch_add(1);

                d_edge_nodes[nEdges] =
                    make_int2(globalIdx2, static_cast<int>(begin_bin1) + n1Idx);

                d_edge_params[nEdges] = make_half4(
                    exp_eta, static_cast<float16>(curv), static_cast<float16>(phi2 + curv * r2),
                    static_cast<float16>(phi1 + curv * r1));
            }
        }
    }
}

} // namespace traccc::device