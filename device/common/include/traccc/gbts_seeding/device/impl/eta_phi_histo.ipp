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

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cmath>
#include <cstdint>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void eta_phi_histo(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::view& d_node_phi_index_view,
    const collection_types<unsigned int>::const_view& d_node_eta_index_view,
    const collection_types<unsigned int>::view& d_eta_phi_histo_view,
    const collection_types<float4>::const_view& d_sp_params_view,
    const unsigned int nNodes, 
    const unsigned int nPhiBins) {

    collection_types<unsigned int>::device d_node_phi_index(d_node_phi_index_view);
    const collection_types<unsigned int>::const_device d_node_eta_index(
        d_node_eta_index_view);
    collection_types<unsigned int>::device d_eta_phi_histo(d_eta_phi_histo_view);
    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);

    const float inv_phiSliceWidth =
        1.0f /
        (2.0f * traccc::device::gbts_pi_f / static_cast<float>(nPhiBins));

    if (globalIndex >= nNodes) {
        return;
    }

    const float4 sp = d_sp_params[globalIndex];
    const float Phi = atan2f(sp.y, sp.x);

    //const unsigned int phiIdx =
    //        static_cast<unsigned int>((Phi + traccc::device::gbts_pi_f) * inv_phiSliceWidth) % nPhiBins;
    //d_node_phi_index[globalIndex] = phiIdx;

    unsigned int phiIdx = static_cast<unsigned int>((Phi + traccc::device::gbts_pi_f) * inv_phiSliceWidth);
    if (phiIdx >= nPhiBins) {
        phiIdx -= nPhiBins;
    }
    d_node_phi_index[globalIndex] = phiIdx;

    const unsigned int eta_index = d_node_eta_index[globalIndex];
    const unsigned int histo_bin = phiIdx + nPhiBins * eta_index;
    vecmem::device_atomic_ref<unsigned int>(d_eta_phi_histo[histo_bin]).fetch_add(1);
}

}  // namespace traccc::device
