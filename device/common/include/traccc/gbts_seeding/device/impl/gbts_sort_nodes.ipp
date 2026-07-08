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
#include "traccc/gbts_seeding/device/details/gbts_sort_keys.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

TRACCC_HOST_DEVICE inline unsigned int gbts_compute_histo_bin(
    unsigned int globalIndex, const gbts_sort_nodes_payload& payload) {

    const vecmem::device_vector<const unsigned int> d_node_eta_index(
        payload.node_eta_index);
    const vecmem::device_vector<const unsigned int> d_node_phi_index(
        payload.node_phi_index);

    const unsigned int eta_index = d_node_eta_index[globalIndex];
    return d_node_phi_index[globalIndex] + payload.nPhiBins * eta_index;
}

TRACCC_HOST_DEVICE inline gbts_node_geometry gbts_compute_node_geometry(
    unsigned int globalIndex, const gbts_sort_nodes_payload& payload) {

    const vecmem::device_vector<const float4> d_sp_params(payload.sp_params);
    const vecmem::device_vector<const unsigned int> d_original_sp_idx(
        payload.original_sp_idx);
    const vecmem::device_vector<const float> d_tau_lut(payload.tau_lut);

    const gbts_sort_nodes_params& ap = payload.gbts_sort_nodes_params;

    const float4 sp = d_sp_params[globalIndex];

    const float Phi = math::atan2(sp.y, sp.x);
    const float r = math::sqrt(sp.x * sp.x + sp.y * sp.y);
    const float z = sp.z;

    // Default to the full |tau| acceptance for nodes that carry no usable
    // cluster width (sp.w <= 0); the per-edge cuts then rely on these
    // bounds.
    float min_tau = 0.0f;
    float max_tau = ap.maxTau;

    if (sp.w > 0) {  // type 0 only
        if (ap.useTauLUT) {
            // LUT is laid out as [w_bin_edge, min_tau_0, max_tau_0,
            // min_tau_1, max_tau_1] per bin.
            const int tau_bin =
                5 * static_cast<int>(
                        math::floor(ap.tau_lut_inv_bin * sp.w) - 1.0f);
            if (tau_bin > -1 && tau_bin < static_cast<int>(ap.tauLutSize)) {
                min_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 1u];
                max_tau = d_tau_lut[static_cast<unsigned int>(tau_bin) + 2u];
            }
            if (max_tau < 0.0f) {
                max_tau = ap.maxTau;
            }
            if (min_tau < 0.0f) {
                min_tau = 0.0f;
            }
        } else {
            // linear fit + correction for short clusters
            min_tau = ap.tMin_slope * (sp.w - ap.offset);
            max_tau = ap.tMax_min + ap.tMax_correction / (sp.w + ap.offset) +
                      ap.tMax_slope * (sp.w - ap.offset);
        }
    }

    return gbts_node_geometry{float4{min_tau, max_tau, r, z}, Phi,
                              d_original_sp_idx[globalIndex], sp};
}

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_fill_sort_keys(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload,
    vecmem::data::vector_view<std::uint64_t> keys_view,
    vecmem::data::vector_view<unsigned int> perm_view) {

    vecmem::device_vector<std::uint64_t> keys(keys_view);
    vecmem::device_vector<unsigned int> perm(perm_view);
    const vecmem::device_vector<const float4> d_sp_params(payload.sp_params);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int i = globalIdx; i < payload.nNodes;
         i += blockDimX * gridDimX) {
        const unsigned int histo_bin = gbts_compute_histo_bin(i, payload);
        const float4 sp = d_sp_params[i];
        const float phi = math::atan2(sp.y, sp.x);
        keys[i] = (static_cast<std::uint64_t>(histo_bin) << 32) |
                  details::orderable_float(phi);
        perm[i] = i;
    }
}

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_gather_sorted_nodes(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload,
    vecmem::data::vector_view<unsigned int> perm_view) {

    vecmem::device_vector<unsigned int> perm(perm_view);
    vecmem::device_vector<float4> d_node_params(payload.node_params);
    vecmem::device_vector<float> d_node_phi(payload.node_phi);
    vecmem::device_vector<unsigned int> d_node_index(payload.node_index);
    vecmem::device_vector<float4> d_node_xyzw(payload.node_xyzw);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int slot = globalIdx; slot < payload.nNodes;
         slot += blockDimX * gridDimX) {
        const unsigned int src = perm[slot];
        const gbts_node_geometry geo = gbts_compute_node_geometry(src, payload);
        d_node_params[slot] = geo.params;
        d_node_phi[slot] = geo.phi;
        d_node_index[slot] = geo.node_index;
        d_node_xyzw[slot] = geo.sp;
    }
}

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_sort_nodes(
    const thread_id_t& thread_id, const gbts_sort_nodes_payload& payload) {

    vecmem::device_vector<unsigned int> d_phi_cusums(payload.phi_cusums);
    vecmem::device_vector<float4> d_node_params(payload.node_params);
    vecmem::device_vector<float> d_node_phi(payload.node_phi);
    vecmem::device_vector<unsigned int> d_node_index(payload.node_index);
    vecmem::device_vector<float4> d_node_xyzw(payload.node_xyzw);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int globalIndex = globalIdx; globalIndex < payload.nNodes;
         globalIndex += blockDimX * gridDimX) {

        const unsigned int histo_bin =
            gbts_compute_histo_bin(globalIndex, payload);
        const gbts_node_geometry geo =
            gbts_compute_node_geometry(globalIndex, payload);

        const unsigned int pos =
            vecmem::device_atomic_ref<unsigned int>(d_phi_cusums[histo_bin])
                .fetch_add(1);

        d_node_params[pos] = geo.params;
        d_node_phi[pos] = geo.phi;
        d_node_index[pos] = geo.node_index;
        d_node_xyzw[pos] = geo.sp;
    }
}

}  // namespace traccc::device
