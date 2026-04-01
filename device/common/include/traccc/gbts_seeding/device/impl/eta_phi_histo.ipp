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
inline void eta_phi_histo_kernel(
    const int idx,
    const collection_types<int>::const_view d_node_phi_index_view,
    const collection_types<int>::const_view d_node_eta_index_view,
    const collection_types<int>::view d_eta_phi_histo_view,
    const unsigned int nNodes,
    const unsigned int nPhiBins) {

    if (idx >= nNodes) {
        return;
    }

    const collection_types<int>::const_device d_node_phi_index(d_node_phi_index_view);
    const collection_types<int>::const_device d_node_eta_index(d_node_eta_index_view);
    collection_types<int>::device d_eta_phi_histo(d_eta_phi_histo_view);

    int eta_index = d_node_eta_index[static_cast<unsigned int>(idx)];
    int histo_bin = d_node_phi_index[static_cast<unsigned int>(idx)] + static_cast<int>(nPhiBins) * eta_index;
    vecmem::device_atomic_ref<int> d_eta_phi_histo_ref(d_eta_phi_histo[static_cast<unsigned int>(histo_bin)]);
    d_eta_phi_histo_ref.fetch_add(1);
}


} // namespace traccc::device