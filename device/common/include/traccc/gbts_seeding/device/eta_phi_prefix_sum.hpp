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
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void eta_phi_prefix_sum_kernel(
    const int eta_bin_idx,
    const collection_types<int>::const_view d_eta_node_counter_view,
    const collection_types<int>::view d_phi_cusums_view,
    const unsigned int nPhiBins);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/eta_phi_prefix_sum.ipp"
