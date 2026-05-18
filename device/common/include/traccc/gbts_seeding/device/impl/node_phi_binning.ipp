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
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <cmath>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void node_phi_binning(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_params_view,
    collection_types<int>::view d_node_phi_index_view,
    const unsigned int nNodes, const unsigned int nPhiBins) {

    if (globalIndex >= nNodes) {
        return;
    }

    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    collection_types<int>::device d_node_phi_index(d_node_phi_index_view);

    const float inv_phiSliceWidth =
        1.0f / (2.0f * traccc::device::gbts_pi_f /
                static_cast<float>(nPhiBins));

    const float4 sp = d_sp_params[globalIndex];
    const float Phi = atan2f(sp.y, sp.x);
    int phiIdx = static_cast<int>((Phi + traccc::device::gbts_pi_f) *
                                  inv_phiSliceWidth);

    if (phiIdx >= static_cast<int>(nPhiBins)) {
        phiIdx %= static_cast<int>(nPhiBins);
    } else if (phiIdx < 0) {
        phiIdx += static_cast<int>(nPhiBins);
        phiIdx %= static_cast<int>(nPhiBins);
    }
    d_node_phi_index[globalIndex] = phiIdx;
}

}  // namespace traccc::device
