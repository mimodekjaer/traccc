/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// CUDA include(s)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_functions.h>

#include <cmath>

// Project include(s)
#include <vecmem/memory/device_atomic_ref.hpp>

#include "../../utils/barrier.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::cuda::kernels {

using uint2 = traccc::uint2;
using float4 = traccc::float4;
using float2 = traccc::float2;
using int2 = traccc::int2;
using uint4 = traccc::uint4;
using short2 = traccc::short2;
using float16 = std::float_t;
TRACCC_DEVICE float16 (*habs)(float16) = fabs;

using traccc::make_float2;
using traccc::make_float4;
using traccc::make_int2;
using traccc::make_short2;
using traccc::make_uint2;
using traccc::make_uint4;

struct TRACCC_ALIGN(16) half4 {  // should be 8, for the actual float16 type
    float16 x, y, z, w;
};

inline TRACCC_HOST_DEVICE half4 make_half4(const float16 x, const float16 y,
                                           const float16 z, const float16 w) {
    half4 t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}

__global__ static void graphEdgeMakingKernel(
    const collection_types<int>::const_view d_bin_pair_views_view,
    const collection_types<float>::const_view
        d_bin_pair_dphi_view,  // Here the type changed as a quick fix to avoid
                               // the issue with the bin_pair_views buffer
    const collection_types<float>::const_view d_node_params_view,
    const gbts_graph_building_params d_graph_building_params,
    const collection_types<unsigned int>::view d_counters_view,
    const collection_types<int2>::view d_edge_nodes_view,
    const collection_types<half4>::view d_edge_params_view,
    const collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins) {

    const collection_types<int>::const_device d_bin_pair_views(
        d_bin_pair_views_view);  // Here the type changed as a quick fix to
                                 // avoid the issue with the bin_pair_views
                                 // buffer
    const collection_types<float>::const_device d_bin_pair_dphi(
        d_bin_pair_dphi_view);
    const collection_types<float>::const_device d_node_params(
        d_node_params_view);
    collection_types<unsigned int>::device d_counters(d_counters_view);
    collection_types<int2>::device d_edge_nodes(d_edge_nodes_view);
    collection_types<half4>::device d_edge_params(d_edge_params_view);
    collection_types<int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    __shared__ unsigned int begin_bin1;
    __shared__ unsigned int begin_bin2;
    __shared__ unsigned int num_nodes1;
    __shared__ unsigned int num_nodes2;
    __shared__ float deltaPhi;

    __shared__ float minDeltaRad;
    __shared__ float min_z0;
    __shared__ float max_z0;
    __shared__ float maxOuterRad;
    __shared__ float min_zU;
    __shared__ float max_zU;
    __shared__ float max_kappa;
    __shared__ float low_Kappa_d0;
    __shared__ float high_Kappa_d0;

    __shared__ float tau_min[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float tau_max[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float r[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float z[traccc::device::gbts_consts::node_buffer_length];

    if (threadIdx.x == 0) {
        deltaPhi = d_bin_pair_dphi[blockIdx.x];

        begin_bin1 =
            d_bin_pair_views[4 * blockIdx.x];  // Here the type changed as a
                                               // quick fix to avoid the issue
                                               // with the bin_pair_views buffer
        begin_bin2 = d_bin_pair_views[4 * blockIdx.x +
                                      2];  // Here the type changed as a quick
                                           // fix to avoid the issue with the
                                           // bin_pair_views buffer
        num_nodes1 =
            d_bin_pair_views[4 * blockIdx.x + 1] -
            begin_bin1;  // Here the type changed as a quick fix to avoid the
                         // issue with the bin_pair_views buffer
        num_nodes2 =
            d_bin_pair_views[4 * blockIdx.x + 3] -
            begin_bin2;  // Here the type changed as a quick fix to avoid the
                         // issue with the bin_pair_views buffer

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
    for (int idx = threadIdx.x; idx < num_nodes1; idx += blockDim.x) {
        // loading a chunk of nodes1 into shared mem buffers
        int offset = 5 * (idx + begin_bin1);
        tau_min[idx] = d_node_params[offset];
        tau_max[idx] = d_node_params[offset + 1];
        phi[idx] = d_node_params[offset + 2];
        r[idx] = d_node_params[offset + 3];
        z[idx] = d_node_params[offset + 4];
    }

    barrier().blockBarrier();

    int last_n1 = 0;  // initial value for the sliding window

    float phi_bin_width = 2.0f * CUDART_PI_F / nPhiBins;

    for (int n2Idx = threadIdx.x; n2Idx < num_nodes2; n2Idx += blockDim.x) {

        int n1Idx = last_n1;

        int globalIdx2 = begin_bin2 + n2Idx;
        int o2 = 5 * globalIdx2;

        float phi2 = d_node_params[2 + o2];

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

        float tau_min2 = d_node_params[o2];
        float tau_max2 = d_node_params[1 + o2];
        float r2 = d_node_params[3 + o2];
        float z2 = d_node_params[4 + o2];

        for (; n1Idx < num_nodes1; n1Idx++) {
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
            unsigned int nEdges =
                vecmem::device_atomic_ref<unsigned int>(d_counters[0])
                    .fetch_add(1);
            if (nEdges < nMaxEdges) {
                float16 exp_eta =
                    static_cast<float16>(sqrtf(1 + tau * tau) - tau);
                // edge linking order is inside->out
                vecmem::device_atomic_ref<int>(
                    d_num_outgoing_edges[begin_bin1 + n1Idx])
                    .fetch_add(1);

                d_edge_nodes[nEdges] =
                    make_int2(globalIdx2, begin_bin1 + n1Idx);

                d_edge_params[nEdges] =
                    make_half4(exp_eta, static_cast<float16>(curv),
                               static_cast<float16>(phi2 + curv * r2),
                               static_cast<float16>(phi1 + curv * r1));
            }
        }
    }
}

__global__ static void graphEdgeLinkingKernel(
    const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<int>::view d_edge_links_view,
    const collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nEdges) {

    const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    collection_types<int>::device d_edge_links(d_edge_links_view);
    collection_types<int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx >= nEdges) {
        return;
    }
    int sharedNode = d_edge_nodes[edge_idx].y;

    // this converts num_outgoing_edges to the start postion for each node in
    // d_edge_links
    int pos = vecmem::device_atomic_ref<int>(d_num_outgoing_edges[sharedNode])
                  .fetch_add(-1);
    // provides views of edges leaving the sharedNode for linking
    d_edge_links[pos - 1] = edge_idx;
}

__global__ static void graphEdgeMatchingKernel(
    const gbts_graph_building_params d_graph_building_params,
    const collection_types<half4>::const_view d_edge_params_view,
    const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<int>::const_view d_num_outgoing_edges_view,
    const collection_types<int>::const_view d_edge_links_view,
    const collection_types<unsigned char>::view d_num_neighbours_view,
    const collection_types<int>::view d_neighbours_view,
    const collection_types<int>::view d_reIndexer_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nEdges, const unsigned int nMaxNei) {
    const collection_types<half4>::const_device d_edge_params(
        d_edge_params_view);
    collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<int>::const_device d_num_outgoing_edges(
        d_num_outgoing_edges_view);
    const collection_types<int>::const_device d_edge_links(d_edge_links_view);
    collection_types<unsigned char>::device d_num_neighbours(
        d_num_neighbours_view);
    collection_types<int>::device d_neighbours(d_neighbours_view);
    collection_types<int>::device d_reIndexer(d_reIndexer_view);
    collection_types<unsigned int>::device d_counters(d_counters_view);
    __shared__ float16 cut_dphi_max;
    __shared__ float16 cut_dcurv_max;
    __shared__ float16 cut_tau_ratio_max;
    __shared__ float16 PI_h;
    __shared__ float16 PI_2_h;
    __shared__ float16 ONE_h;
    if (threadIdx.x == 0) {
        cut_dphi_max =
            static_cast<float16>(d_graph_building_params.cut_dphi_max);
        cut_dcurv_max =
            static_cast<float16>(d_graph_building_params.cut_dcurv_max);
        cut_tau_ratio_max =
            static_cast<float16>(d_graph_building_params.cut_tau_ratio_max);

        PI_h = 3.141592654;
        PI_2_h = 6.283185307;
        ONE_h = 1.0;
    }
    barrier().blockBarrier();

    int edge1_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge1_idx >= nEdges) {
        return;
    }

    int sharedNode = d_edge_nodes[edge1_idx].x;

    int link_begin = d_num_outgoing_edges[sharedNode];
    // the number of edges leaving the sharedNode
    int nLinks = d_num_outgoing_edges[sharedNode + 1] - link_begin;
    if (nLinks == 0) {
        return;
    }
    half4 params1 = d_edge_params[edge1_idx];  // [exp_eta, curv, Phi1, Phi2]

    float16 uat_2 = ONE_h / params1.x;
    float16 Phi2 = params1.z;
    float16 curv2 = params1.y;

    int nei_pos = nMaxNei * edge1_idx;

    unsigned char num_nei = 0;

    for (int k = 0; k < nLinks; k++) {  // loop over potential neighbours

        if (num_nei >= nMaxNei) {
            break;
        }
        int edge2_idx = d_edge_links[link_begin + k];

        half4 params2 = d_edge_params[edge2_idx];

        float16 tau_ratio = static_cast<float16>(params2.x * uat_2 - ONE_h);

        if (habs(tau_ratio) > cut_tau_ratio_max) {  // bad match
            continue;
        }

        float16 dPhi = Phi2 - params2.w;  // Phi2

        if (dPhi < -PI_h) {
            dPhi += PI_2_h;
        } else if (dPhi > PI_h) {
            dPhi -= PI_2_h;
        }
        if (habs(dPhi) > cut_dphi_max) {
            continue;
        }

        float16 dcurv = curv2 - params2.y;

        if (habs(dcurv) > cut_dcurv_max) {
            continue;
        }

        d_neighbours[nei_pos + num_nei] = edge2_idx;
        d_reIndexer[edge2_idx] = 1;

        ++num_nei;
    }

    d_num_neighbours[edge1_idx] = num_nei;

    if (num_nei != 0) {
        d_reIndexer[edge1_idx] = 1;
        vecmem::device_atomic_ref<unsigned int>(d_counters[1])
            .fetch_add(num_nei);
    }
}

__global__ void edgeReIndexingKernel(
    const collection_types<int>::view d_reIndexer_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nEdges) {

    // each thread gets an edge
    collection_types<int>::device d_reIndexer(d_reIndexer_view);
    collection_types<unsigned int>::device d_counters(d_counters_view);

    int edge_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (edge_idx >= nEdges) {
        return;
    }
    if (d_reIndexer[edge_idx] == -1) {
        return;
    }
    d_reIndexer[edge_idx] =
        vecmem::device_atomic_ref<unsigned int>(d_counters[2]).fetch_add(1);
}

__global__ static void graphCompressionKernel(
    const collection_types<int>::const_view d_orig_node_index_view,
    const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<unsigned char>::const_view d_num_neighbours_view,
    const collection_types<int>::const_view d_neighbours_view,
    const collection_types<int>::const_view d_reIndexer_view,
    const collection_types<int>::view d_output_graph_view,
    const unsigned int nEdgesPerBlock, const unsigned int nEdges,
    const unsigned int nMaxNei) {

    const collection_types<int>::const_device d_orig_node_index(
        d_orig_node_index_view);
    const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<unsigned char>::const_device d_num_neighbours(
        d_num_neighbours_view);
    const collection_types<int>::const_device d_neighbours(d_neighbours_view);
    const collection_types<int>::const_device d_reIndexer(d_reIndexer_view);
    collection_types<int>::device d_output_graph(d_output_graph_view);

    int begin_edge = blockIdx.x * nEdgesPerBlock;
    int edge_size = 2 + 1 + nMaxNei;

    for (int idx = threadIdx.x + begin_edge; idx < begin_edge + nEdgesPerBlock;
         idx += blockDim.x) {

        if (idx >= nEdges) {
            continue;
        }
        int newIdx = d_reIndexer[idx];
        if (newIdx == -1) {
            continue;
        }
        int pos = edge_size * newIdx;
        int2 edge_nodes = d_edge_nodes[idx];
        int node1_idx = d_orig_node_index[edge_nodes.x];
        d_output_graph[pos + traccc::device::gbts_consts::node1] = node1_idx;
        int node2_idx = d_orig_node_index[edge_nodes.y];
        d_output_graph[pos + traccc::device::gbts_consts::node2] = node2_idx;

        unsigned char nNei = d_num_neighbours[idx];
        d_output_graph[pos + traccc::device::gbts_consts::nNei] = nNei;
        int nei_pos = nMaxNei * idx;
        for (int k = 0; k < nNei; k++) {
            d_output_graph[pos + traccc::device::gbts_consts::nei_start + k] =
                d_reIndexer[d_neighbours[nei_pos + k]];
        }
    }
}

}  // namespace traccc::cuda::kernels
