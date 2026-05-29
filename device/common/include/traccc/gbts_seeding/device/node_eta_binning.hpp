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

/// Assign each node a global eta-bin index based on its layer geometry.
///
/// One CUDA block processes one GBTS layer.  Threads cooperatively iterate
/// over that layer's spacepoints, compute eta from the layer's geometry pair
/// and the SP's (r, z), and clamp into the layer's eta-bin range.
///
/// @param[in]  blockIndex          CUDA block index (== GBTS layer index)
/// @param[in]  threadIndex         CUDA thread index within the block
/// @param[in]  blockSize           CUDA block size
/// @param[in]  d_sp_params_view    Layer-ordered (x, y, z, r) per SP
/// @param[in]  d_layer_info_view   Per-layer (first eta bin, number of bins)
/// @param[in]  d_layer_geo_view    Per-layer geometry pair driving eta calc
/// @param[out] d_node_eta_index_view Global eta-bin index per node
/// @param[in]  layerCounts_view    Per-layer spacepoint count prefix sum
/// @param[in]  nLayers             Number of GBTS layers
///
TRACCC_HOST_DEVICE inline void node_eta_binning(
    const unsigned int blockIndex, 
    const unsigned int threadIndex, 
    const unsigned int blockSize,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<std::pair<int, int>>::const_view& d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view&
        d_layer_geo_view,
    const collection_types<int>::view d_node_eta_index_view,
    const collection_types<int>::view layerCounts_view, const unsigned int nLayers);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/node_eta_binning.ipp"
