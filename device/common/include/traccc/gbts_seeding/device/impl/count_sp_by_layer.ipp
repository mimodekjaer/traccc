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
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/edm/container.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_sp_by_layer(
    const global_index_t globalIndex,
    const traccc::edm::spacepoint_collection::const_view spacepoints_view,
    const edm::measurement_collection::const_view measurements_view,
    const collection_types<short>::const_view volumeToLayerMap_view, 
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view surfaceToLayerMap_view,
    const collection_types<char>::const_view layerType_view, 
    const collection_types<float4>::view reducedSP_view, 
    const collection_types<int>::view layerCounts_view,
    const collection_types<unsigned short>::view spacepointsLayer_view, 
    const float type1_max_width,
    const unsigned int nSp,
    const unsigned int volumeMapSize,
    const unsigned int surfaceMapSize, 
    const bool doTauCut) {

    if (globalIndex >= nSp) {
        return;
    }

    // Parameters passed to the kernel
    const traccc::edm::spacepoint_collection::const_device spacepoints(
        spacepoints_view);
    const edm::measurement_collection::const_device measurements(measurements_view);
    const collection_types<short>::const_device volumeToLayerMap(volumeToLayerMap_view);
    const collection_types<std::pair<unsigned int, unsigned int>>::const_device surfaceToLayerMap(surfaceToLayerMap_view);
    const collection_types<char>::const_device layerType(layerType_view);


    // Parameters calculated by the kernel
    collection_types<int>::device layerCounts(layerCounts_view);
    collection_types<unsigned short>::device spacepointsLayer(spacepointsLayer_view);
    collection_types<float4>::device reducedSP(reducedSP_view);


    // get the layer of the spacepoint
    const traccc::edm::spacepoint_collection::const_device::const_proxy_type
        spacepoint = spacepoints.at(globalIndex);
    const auto measurement =
        measurements.at(spacepoint.measurement_index_1());

    detray::geometry::barcode barcode = measurement.surface_link();

    // some volume_ids map one to one with layer others need searching
    if (barcode.volume() > volumeMapSize) {
        reducedSP[globalIndex].w = -CHAR_MAX - 1;
        return;  // unconfigured volume
    }
    short begin_or_bin = volumeToLayerMap[barcode.volume()];
    if (begin_or_bin == SHRT_MAX) {
        reducedSP[globalIndex].w = -CHAR_MAX - 1;
        return;  // unconfigured volume
    }
    unsigned int layerIdx;
    bool foundLayer = false;
    if (begin_or_bin < 0) {
        unsigned int surface_index =
            static_cast<unsigned int>(barcode.index());

        for (unsigned int surface = static_cast<unsigned int>(-1 * (begin_or_bin + 1));
                surface < surfaceMapSize; surface++) {

            std::pair<unsigned int, unsigned int> surfaceBinPair = surfaceToLayerMap[surface];
            if (surfaceBinPair.first == surface_index) {
                layerIdx = surfaceBinPair.second;
                foundLayer = true;
                break;
            }
        }
        if (!foundLayer) {
            reducedSP[globalIndex].w = -CHAR_MAX - 1;
            return;  // unconfigured surface
        }
    } else {
        layerIdx = static_cast<unsigned int>(begin_or_bin);
    }
    float cluster_diameter = measurement.diameter();
    int type = static_cast<int>(layerType[layerIdx]);
    if (type == 1 && cluster_diameter > type1_max_width) {
        //-ve cluster_diameter to skip cot(theta) prediction
        // large -ve to skip spacepoint entirely
        reducedSP[globalIndex].w = -CHAR_MAX - 1;
        return;
    }
    cluster_diameter =
        (doTauCut && type != 0) ? static_cast<float>(-1 * type) : cluster_diameter;

    // count and store x,y,z,cw info
    vecmem::device_atomic_ref<int> layerCounts_ref(layerCounts[layerIdx]);
    layerCounts_ref.fetch_add(1);
    spacepointsLayer[globalIndex] = static_cast<unsigned short>(layerIdx);
    const traccc::point3 pos = spacepoint.global();
    reducedSP[globalIndex] =
        make_float4(static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2]), cluster_diameter);

}

} // namespace traccc::device