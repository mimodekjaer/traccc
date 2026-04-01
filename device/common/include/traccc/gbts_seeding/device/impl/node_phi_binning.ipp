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

// Math include(s).
#include <cmath>
#include <numbers>

// TODO Finish the implementation



namespace traccc::device {


TRACCC_HOST_DEVICE
inline void node_phi_binning_kernel(
    const int idx,
    const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<int>::view d_node_phi_index_view,
    const unsigned int nNodesPerBlock,
    const unsigned int nNodes,
    const unsigned int nPhiBins) {


    if (idx >= nNodes) {
        return;
    }

    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    collection_types<int>::device d_node_phi_index(d_node_phi_index_view);


    const float inv_phiSliceWidth = 1 / (2.0f * std::numbers::pi_v<float> / static_cast<float>(nPhiBins)); // I think this could even be constexpr if phibins not cofigureable

    float4 sp = d_sp_params[static_cast<unsigned int>(idx)];

    float Phi = atan2f(sp.y, sp.x);

    int phiIdx = static_cast<int>((Phi + std::numbers::pi_v<float>) * inv_phiSliceWidth);

    if (phiIdx >= static_cast<int>(nPhiBins)) {
        phiIdx %= static_cast<int>(nPhiBins);
    } else if (phiIdx < 0) {
        phiIdx += static_cast<int>(nPhiBins);
        phiIdx %= static_cast<int>(nPhiBins);
    }
    d_node_phi_index[static_cast<unsigned int>(idx)] = phiIdx;

}

} // namespace traccc::device