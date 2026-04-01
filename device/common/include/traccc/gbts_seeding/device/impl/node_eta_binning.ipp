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

#include <cmath>


namespace traccc::device {


// TODO Finish the implementation

TRACCC_HOST_DEVICE
inline void node_eta_binning_kernel(
    const unsigned int layerIdx,
    const unsigned int idx,
    const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<std::pair<int, int>>::const_view d_layer_info_view,
    const collection_types<std::pair<float, float>>::const_view d_layer_geo_view,
    const collection_types<int>::view d_node_eta_index_view
    ) {

        
    // Parameters passed to the kernel
    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<std::pair<int, int>>::const_device d_layer_info(d_layer_info_view);
    const collection_types<std::pair<float, float>>::const_device d_layer_geo(d_layer_geo_view);
    collection_types<int>::device d_node_eta_index(d_node_eta_index_view);

    const std::pair<int, int> layerInfo = d_layer_info[layerIdx];
    const int bin0 = layerInfo.first;
    const int num_eta_bins = layerInfo.second;
    const std::pair<float, float> layerGeo = d_layer_geo[layerIdx];
    const float min_eta = layerGeo.first;
    const float eta_bin_width = layerGeo.second;



    if (num_eta_bins == 1) {  // all nodes are in the same bin
        d_node_eta_index[idx] = bin0;
    } else {  // binIndex needs to be calculated first
    float4 sp = d_sp_params[idx];

    float r = sqrtf(sp.x * sp.x + sp.y * sp.y);

    float t1 = sp.z / r;

    float eta = -logf(sqrtf(1 + t1 * t1) - t1);

    int binIdx = static_cast<int>(std::roundf((eta - min_eta) / eta_bin_width));

    if (binIdx < 0) {
        binIdx = 0;
    }
    if (binIdx >= num_eta_bins) {
        binIdx = num_eta_bins - 1;
    }
        d_node_eta_index[idx] = bin0 + binIdx;
    }
    }



} // namespace traccc::device