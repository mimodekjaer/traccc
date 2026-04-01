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

TRACCC_HOST_DEVICE
inline void minmax_rad_kernel( const int eta_bin_idx,
    const collection_types<int>::const_view d_eta_bin_views_view, // Here the type changed as a quick fix to avoid the issue with the eta_bin_views buffer
    const collection_types<float>::const_view d_node_params_view,
    const collection_types<float>::view d_bin_rads_view,
    const unsigned int maxEtaBin) {


    if (eta_bin_idx >= maxEtaBin) {
        return;
    }
    const collection_types<int>::const_device d_eta_bin_views(d_eta_bin_views_view); // Here the type changed as a quick fix to avoid the issue with the eta_bin_views buffer
    const collection_types<float>::const_device d_node_params(d_node_params_view);
    collection_types<float>::device d_bin_rads(d_bin_rads_view); // Here the type changed as a quick fix to avoid the issue with the bin_rads buffer

    int node_start = d_eta_bin_views[static_cast<unsigned int>(2*eta_bin_idx)]; // Here the type changed as a quick fix to avoid the issue with the eta_bin_views buffer
    int node_end = d_eta_bin_views[static_cast<unsigned int>(2*eta_bin_idx + 1)];

    if (node_start == node_end) {
        return;
    }
    float min_r = static_cast<float>(1e8);
    float max_r = static_cast<float>(-1e8);

    for (int idx = node_start; idx < node_end; idx++) {
        float r = d_node_params[static_cast<unsigned int>(5 * idx + 3)];
        if (r > max_r) {
            max_r = r;
        }
        if (r < min_r) {
            min_r = r;
        }
    }

    d_bin_rads[static_cast<unsigned int>(2*eta_bin_idx)] = min_r; // Here the type changed as a quick fix to avoid the issue with the bin_rads buffer
    d_bin_rads[static_cast<unsigned int>(2*eta_bin_idx + 1)] = max_r;
}


} // namespace traccc::device