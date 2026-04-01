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
inline void eta_phi_prefix_sum_kernel(
    const int eta_bin_idx,
    const collection_types<int>::const_view d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned int nPhiBins,
    const unsigned int maxEtaBin) {
    

    if (eta_bin_idx >= maxEtaBin) {
        return;
    }
    if (eta_bin_idx == 0) {
        return;
    }
    const collection_types<int>::const_device d_eta_node_counter(d_eta_node_counter_view);
    collection_types<int>::device d_phi_cusums(d_phi_cusums_view);

    int offset = static_cast<int>(nPhiBins) * eta_bin_idx;

    int val0 = d_eta_node_counter[static_cast<unsigned int>(eta_bin_idx - 1)];

    for (int phiIdx = 0; phiIdx < static_cast<int>(nPhiBins); phiIdx++) {
        d_phi_cusums[static_cast<unsigned int>(offset + phiIdx)] += val0;
    }
}


} // namespace traccc::device