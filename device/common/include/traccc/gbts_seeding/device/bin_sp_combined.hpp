/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
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

// System include(s).
#include <utility>

namespace traccc::device {

/// Per-spacepoint kernel that fuses three previously-separate stages:
///   1. @c bin_sp_by_layer  — atomically claim a layer-ordered slot.
///   2. @c node_eta_binning — compute the eta-bin index for the same SP.
///   3. @c eta_phi_histo    — compute the phi-bin index and bump the
///                            (eta, phi) histogram bucket.
///
/// Each thread reads its source spacepoint once and keeps it in registers
/// across all three stages, eliminating the
/// `sp_params[binedIdx]`-write-then-read round trip and a kernel launch
/// boundary between each stage. Output buffers are written exactly the
/// same way as the three originals.
TRACCC_HOST_DEVICE
inline void bin_sp_combined(
    const global_index_t globalIndex,
    const collection_types<float4>::view sp_params_view,
    const collection_types<float4>::const_view& reducedSP_view,
    const collection_types<unsigned int>::view layerCounts_view,
    const collection_types<short>::const_view& spacepointsLayer_view,
    const collection_types<unsigned int>::view original_sp_idx_view,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view&
        layer_info_view,
    const collection_types<std::pair<float, float>>::const_view&
        layer_geo_view,
    const collection_types<unsigned int>::view node_eta_index_view,
    const collection_types<unsigned int>::view node_phi_index_view,
    const collection_types<unsigned int>::view eta_phi_histo_view,
    const unsigned int nSp, const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/bin_sp_combined.ipp"
