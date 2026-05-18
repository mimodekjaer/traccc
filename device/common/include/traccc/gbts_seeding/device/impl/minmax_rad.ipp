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

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void minmax_rad(
    const global_index_t globalIndex,
    const collection_types<int>::const_view& d_eta_bin_views_view,
    const collection_types<float>::const_view& d_node_params_view,
    collection_types<float>::view d_bin_rads_view,
    const unsigned int maxEtaBin) {

    if (globalIndex >= maxEtaBin) {
        return;
    }

    const collection_types<int>::const_device d_eta_bin_views(
        d_eta_bin_views_view);
    const collection_types<float>::const_device d_node_params(
        d_node_params_view);
    collection_types<float>::device d_bin_rads(d_bin_rads_view);

    const int node_start = d_eta_bin_views[2u * globalIndex];
    const int node_end = d_eta_bin_views[2u * globalIndex + 1u];

    if (node_start == node_end) {
        return;
    }

    float min_r = 1e8f;
    float max_r = -1e8f;

    for (int idx = node_start; idx < node_end; idx++) {
        const float r =
            d_node_params[static_cast<unsigned int>(5 * idx + 3)];
        if (r > max_r) {
            max_r = r;
        }
        if (r < min_r) {
            min_r = r;
        }
    }

    d_bin_rads[2u * globalIndex] = min_r;
    d_bin_rads[2u * globalIndex + 1u] = max_r;
}

}  // namespace traccc::device
