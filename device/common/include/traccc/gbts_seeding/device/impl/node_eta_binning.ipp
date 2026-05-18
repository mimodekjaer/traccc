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
#include <cmath>
#include <utility>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void node_eta_binning(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<std::pair<int, int>>::const_view& d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view&
        d_layer_geo_view,
    collection_types<int>::view d_node_eta_index_view,
    collection_types<int>::view layerCounts_view, const unsigned int nLayers) {

    collection_types<int>::device layerCounts(layerCounts_view);
    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<std::pair<int, int>>::const_device d_layer_info(
        d_layer_info_view);
    const collection_types<std::pair<float, float>>::const_device d_layer_geo(
        d_layer_geo_view);
    collection_types<int>::device d_node_eta_index(d_node_eta_index_view);

    const unsigned int layerIdx = blockIndex;
    if (layerIdx >= nLayers) {
        return;
    }

    const std::pair<int, int> layerInfo = d_layer_info[layerIdx];
    const int bin0 = layerInfo.first;
    const int num_eta_bins = layerInfo.second;
    const int layer_begin = layerCounts[layerIdx];
    const int layer_end = layerCounts[layerIdx + 1];
    const std::pair<float, float> layerGeo = d_layer_geo[layerIdx];
    const float min_eta = layerGeo.first;
    const float eta_bin_width = layerGeo.second;

    if (num_eta_bins == 1) {
        for (int idx = static_cast<int>(threadIndex) + layer_begin;
             idx < layer_end; idx += static_cast<int>(blockSize)) {
            d_node_eta_index[static_cast<unsigned int>(idx)] = bin0;
        }
    } else {
        for (int idx = static_cast<int>(threadIndex) + layer_begin;
             idx < layer_end; idx += static_cast<int>(blockSize)) {
            const float4 sp = d_sp_params[static_cast<unsigned int>(idx)];
            const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
            const float t1 = sp.z / r;
            const float eta = -logf(sqrtf(1 + t1 * t1) - t1);
            int binIdx = static_cast<int>((eta - min_eta) / eta_bin_width);
            if (binIdx < 0) {
                binIdx = 0;
            }
            if (binIdx >= num_eta_bins) {
                binIdx = num_eta_bins - 1;
            }
            d_node_eta_index[static_cast<unsigned int>(idx)] = bin0 + binIdx;
        }
    }
}

}  // namespace traccc::device
