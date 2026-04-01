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
inline void minmax_rad_kernel(
    const int eta_bin_idx,
    const collection_types<int>::const_view d_eta_bin_views_view,
    const collection_types<float>::const_view d_node_params_view,
    const collection_types<float>::view d_bin_rads_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/minmax_rad.ipp"
