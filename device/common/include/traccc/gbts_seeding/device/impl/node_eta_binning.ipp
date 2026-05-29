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
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view& d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view&
        d_layer_geo_view,
    const collection_types<unsigned int>::view& d_node_eta_index_view,
    const collection_types<unsigned int>::view& layerCounts_view, const unsigned int nLayers) {

    const collection_types<unsigned int>::device layerCounts(layerCounts_view);
    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<std::pair<unsigned int, unsigned int>>::const_device d_layer_info(
        d_layer_info_view);
    const collection_types<std::pair<float, float>>::const_device d_layer_geo(
        d_layer_geo_view);
    collection_types<unsigned int>::device d_node_eta_index(d_node_eta_index_view);

    const unsigned int layerIdx = blockIndex;
    if (layerIdx >= nLayers) {
        return;
    }

    const std::pair<unsigned int, unsigned int> layerInfo = d_layer_info[layerIdx];
    const unsigned int bin0 = layerInfo.first;
    const unsigned int num_eta_bins = layerInfo.second;
    const unsigned int layer_begin = layerCounts[layerIdx];
    const unsigned int layer_end = layerCounts[layerIdx + 1];
    const std::pair<float, float> layerGeo = d_layer_geo[layerIdx];
    const float min_eta = layerGeo.first;
    const float eta_bin_width = layerGeo.second;
    for (unsigned int idx = threadIndex + layer_begin;
             idx < layer_end; idx += blockSize) {
        if (num_eta_bins == 1) {
                d_node_eta_index[idx] = bin0;
        } else {
            const float4 sp = d_sp_params[static_cast<unsigned int>(idx)];
            const float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
            const float t1 = sp.z / r;
            const float eta = -logf(sqrtf(1 + t1 * t1) - t1);
            unsigned int binIdx = static_cast<unsigned int>(fmaxf(0.f, 
                fminf((eta - min_eta) / eta_bin_width, 
                static_cast<float>(num_eta_bins - 1))));
            d_node_eta_index[idx] = bin0 + binIdx;
        }
    }
}

}  // namespace traccc::device
