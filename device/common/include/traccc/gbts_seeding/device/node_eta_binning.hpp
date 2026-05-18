/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <utility>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void node_eta_binning(
    unsigned int blockIndex, unsigned int threadIndex, unsigned int blockSize,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<std::pair<int, int>>::const_view& d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view&
        d_layer_geo_view,
    collection_types<int>::view d_node_eta_index_view,
    collection_types<int>::view layerCounts_view, unsigned int nLayers);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/node_eta_binning.ipp"
