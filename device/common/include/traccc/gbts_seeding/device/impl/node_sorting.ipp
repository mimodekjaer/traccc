/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cmath>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void node_sorting(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<int>::const_view& d_node_eta_index_view,
    const collection_types<int>::const_view& d_node_phi_index_view,
    collection_types<int>::view d_phi_cusums_view,
    collection_types<float>::view d_node_params_view,
    collection_types<int>::view d_node_index_view,
    const collection_types<int>::const_view& d_original_sp_idx_view,
    const gbts_graph_building_params& ap, const unsigned int nNodes,
    const unsigned int nPhiBins) {

    if (globalIndex >= nNodes) {
        return;
    }

    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<int>::const_device d_node_eta_index(
        d_node_eta_index_view);
    const collection_types<int>::const_device d_node_phi_index(
        d_node_phi_index_view);
    collection_types<int>::device d_phi_cusums(d_phi_cusums_view);
    collection_types<float>::device d_node_params(d_node_params_view);
    collection_types<int>::device d_node_index(d_node_index_view);
    const collection_types<int>::const_device d_original_sp_idx(
        d_original_sp_idx_view);

    const float4 sp = d_sp_params[globalIndex];

    const float Phi = atan2f(sp.y, sp.x);
    const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
    const float z = sp.z;

    float min_tau = -100.0f;
    float max_tau = 100.0f;

    if (sp.w > 0) {
        min_tau = ap.tMin_slope * (sp.w - ap.offset);
        max_tau = ap.tMax_min + ap.tMax_correction / (sp.w + ap.offset) +
                  ap.tMax_slope * (sp.w - ap.offset);
    }

    const int eta_index = d_node_eta_index[globalIndex];
    const int histo_bin = d_node_phi_index[globalIndex] +
                          static_cast<int>(nPhiBins) * eta_index;

    const int pos =
        vecmem::device_atomic_ref<int>(
            d_phi_cusums[static_cast<unsigned int>(histo_bin)])
            .fetch_add(1);

    const unsigned int o = 5u * static_cast<unsigned int>(pos);

    d_node_params[o] = min_tau;
    d_node_params[o + 1u] = max_tau;
    d_node_params[o + 2u] = Phi;
    d_node_params[o + 3u] = r;
    d_node_params[o + 4u] = z;
    d_node_index[static_cast<unsigned int>(pos)] =
        d_original_sp_idx[globalIndex];
}

}  // namespace traccc::device
