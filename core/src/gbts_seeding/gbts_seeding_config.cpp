/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

#include "traccc/geometry/detector_type_list.hpp"
#include "traccc/geometry/host_detector.hpp"

#include <algorithm>
#include <ranges>

#include <detray/geometry/detail/vertexer.hpp>
#include <detray/geometry/surface.hpp>


namespace traccc {

// binTables contains pairs of linked layer-eta bins
// the layerInfo should really be calculated from the barcodeBinning
// BarcodeBinning pair is detray barcode and bin index (corrisponding to the
// layers in layerInfo) minPt in MeV
bool gbts_seedfinder_config::setLinkingScheme(
    const std::vector<std::pair<int, std::vector<int>>>& input_binTables,
    std::vector<std::pair<uint64_t, short>>& detrayBarcodeBinning,
    const traccc::host_detector& detector,
    float minPt = 900.0f,
    std::unique_ptr<const traccc::Logger> callers_logger =
        getDummyLogger().clone()) {
    TRACCC_LOCAL_LOGGER(std::move(callers_logger));

    // unroll binTables
    for (std::pair<int, std::vector<int>> binPairs : input_binTables) {
        for (int bin2 : binPairs.second) {
            binTables.push_back(
                std::make_pair(static_cast<unsigned int>(binPairs.first),
                               static_cast<unsigned int>(bin2)));
        }
    }

    
    // bin by volume
    std::ranges::sort(
        detrayBarcodeBinning,
        [](const std::pair<uint64_t, short> a,
           const std::pair<uint64_t, short> b) { return a.first > b.first; });
    unsigned int largest_volume_index =
        detray::geometry::barcode(detrayBarcodeBinning[0].first).volume();
    auto current_volume = static_cast<short>(largest_volume_index);
    if (largest_volume_index >= SHRT_MAX) {
        TRACCC_ERROR(
            "volume index to large to fit in type-short GBTS volume to layer "
            "map");
        return false;
    }
    bool layerChange = false;
    short current_layer = detrayBarcodeBinning[0].second;
    size_t bound_current_layer = static_cast<size_t>(current_layer);
    size_t max_layer = bound_current_layer;
    for (std::pair<uint64_t, short> barcodeLayerPair : detrayBarcodeBinning) { 
        max_layer = std::max(max_layer, static_cast<size_t>(barcodeLayerPair.second));
    }
    int split_volumes = 0;
    std::vector<std::pair<short, unsigned int>> volumeToLayerMap_unordered;
    detrayBarcodeBinning.push_back(
        std::make_pair(UINT_MAX, -1));  // end-of-vector element
    std::vector<std::array<unsigned int, 2>> surfacesInVolume;
    std::vector<std::array<float, 6>> boundingBoxesInVolume(max_layer + 1); // xmin, xmax, ymin, ymax, zmin, zmax for the layer
    int nSurfacesInVolume = 0;
    std::array<float, 6> layerBoundingBox = {4000.0, -4000.0, 0, 4000.0, -4000.0, 0}; // rmin, rmax, ravg, zmin, zmax, zavg for the layer
    std::array<float, 6> bounding_box_xy = {4000.0, -4000.0, 0, 4000.0, -4000.0, 0}; // rmin, rmax, ravg, zmin, zmax, zavg for the surface
    for (std::pair<uint64_t, short> barcodeLayerPair : detrayBarcodeBinning) {
        detray::geometry::barcode barcode(barcodeLayerPair.first);
        if (current_volume != static_cast<short>(barcode.volume())) {
            // reached the end of this volume so add it to the maps
            short bin = current_layer;
            if (layerChange) {
                split_volumes++;
                bin = -1 *
                      static_cast<short>(
                          surfaceToLayerMap.size() +
                          1);  // start of this volume's surfaces in the map + 1
                for (std::array<unsigned int, 2> pair : surfacesInVolume){
                    surfaceToLayerMap.push_back(pair);
                }
            }
            volumeToLayerMap_unordered.push_back(std::make_pair(
                bin, current_volume));  // layerIdx if not split, begin-index in
                                        // the surface map otherwise

            current_volume = static_cast<short>(barcode.volume());
            current_layer = barcodeLayerPair.second;
            layerChange = false;
    
            surfacesInVolume.clear();
        }
        if (barcodeLayerPair.second != -1) {
        bounding_box_xy = traccc::host_detector_visitor<traccc::detector_type_list>(
            detector,
            [&barcode]<typename detector_traits_t>(
                const typename detector_traits_t::host& det) {
                detray::geometry::surface<typename detector_traits_t::host> surf(
                    det, barcode);
                const typename detector_traits_t::host::geometry_context ctx{};
                auto vertices =
                    detray::detail::get_global_vertices<
                        typename detector_traits_t::host>(ctx, surf, 1u);
                auto position = surf.center(ctx);
                std::array<float, 6> bounding_box = {4000.0f, -4000.0f, static_cast<float>(std::sqrt(position[0]*position[0] + position[1]*position[1])), 4000.0f, -4000.0f, static_cast<float>(position[2])}; // rmin, rmax, ravg, zmin, zmax, zavg
                for (auto& vertex : vertices) {
                    for (auto& v : vertex) {
                        auto r = std::sqrt(v[0]*v[0] + v[1]*v[1]);
                        bounding_box[0] = std::min(bounding_box[0], static_cast<float>(r));
                        bounding_box[1] = std::max(bounding_box[1], static_cast<float>(r));
                        bounding_box[3] = std::min(bounding_box[3], static_cast<float>(v[2]));
                        bounding_box[4] = std::max(bounding_box[4], static_cast<float>(v[2]));
                    }
                }
                return bounding_box;
            });
        }
        // is volume encompassed by a layer
        layerChange |= (current_layer != barcodeLayerPair.second);
        if (bound_current_layer != static_cast<size_t>(barcodeLayerPair.second)) {
            layerBoundingBox[2] = layerBoundingBox[2] / static_cast<float>(nSurfacesInVolume);
            layerBoundingBox[5] = layerBoundingBox[5] / static_cast<float>(nSurfacesInVolume);
            boundingBoxesInVolume[bound_current_layer] = layerBoundingBox;
            bound_current_layer = static_cast<size_t>(barcodeLayerPair.second);
            layerBoundingBox = {4000.0, -4000.0, 0, 4000.0, -4000.0, 0}; // rmin, rmax, ravg, zmin, zmax, zavg for the layer
            nSurfacesInVolume = 0;
        }
        layerBoundingBox[0] = std::min(layerBoundingBox[0], bounding_box_xy[0]);
        layerBoundingBox[1] = std::max(layerBoundingBox[1], bounding_box_xy[1]);
        layerBoundingBox[2] = (layerBoundingBox[2] + bounding_box_xy[2]);
        layerBoundingBox[3] = std::min(layerBoundingBox[3], bounding_box_xy[3]);
        layerBoundingBox[4] = std::max(layerBoundingBox[4], bounding_box_xy[4]);
        layerBoundingBox[5] = (layerBoundingBox[5] + bounding_box_xy[5]);
        nSurfacesInVolume++;
        // save surfaces incase volume is not encommpassed by a layer
        surfacesInVolume.push_back(std::array<unsigned int, 2>{
                static_cast<unsigned int>(barcode.index()),
                static_cast<unsigned int>(barcodeLayerPair.second)});
    }

    layerInfo.reserve(static_cast<unsigned int>(boundingBoxesInVolume.size()));

    int first_bin = 0;
    for (size_t i = 0; i < boundingBoxesInVolume.size(); i++) {
        float dr = boundingBoxesInVolume[i][1] - boundingBoxesInVolume[i][0];
        float dz = boundingBoxesInVolume[i][4] - boundingBoxesInVolume[i][3];
        float min_eta, max_eta, min_tau, max_tau;
        char type;
        if(dz > dr) {
            min_tau = boundingBoxesInVolume[i][3] / boundingBoxesInVolume[i][2];
            max_tau = boundingBoxesInVolume[i][4] / boundingBoxesInVolume[i][2];
            min_eta = -std::log(std::sqrt(1 + min_tau*min_tau) - min_tau);
            max_eta = -std::log(std::sqrt(1 + max_tau*max_tau) - max_tau);
            type = '0';
        } else {
            min_tau = boundingBoxesInVolume[i][5] / boundingBoxesInVolume[i][0];
            max_tau = boundingBoxesInVolume[i][5] / boundingBoxesInVolume[i][1];
            min_eta = -std::log(std::sqrt(1 + min_tau*min_tau) - min_tau);
            max_eta = -std::log(std::sqrt(1 + max_tau*max_tau) - max_tau);
            type = '1';
        }
        if(min_eta > max_eta) {
            std::swap(min_eta, max_eta);
        }
        min_eta -= 1e-6f;
        max_eta += 1e-6f;
        int bins = static_cast<int>(std::round((max_eta - min_eta)/0.2f));
        float eta_bin_width = (max_eta - min_eta) / static_cast<float>(bins);
        layerInfo.addLayer(type, first_bin, bins, static_cast<float>(min_eta), static_cast<float>(eta_bin_width));
        TRACCC_DEBUG("Layer:" << i << " type: " << type << " first_bin: " << first_bin << " bins: " << bins << " min_eta: " << min_eta << " eta_bin_width: " << eta_bin_width);
        first_bin += bins;   
    }

    n_eta_bins = static_cast<unsigned int>(first_bin);
    // make volume by layer map
    volumeToLayerMap.resize(largest_volume_index + 1);
    for (unsigned int i = 0; i < largest_volume_index + 1; ++i)
        volumeToLayerMap.push_back(SHRT_MAX);
    for (std::pair<short, unsigned int> vLpair : volumeToLayerMap_unordered)
        volumeToLayerMap[vLpair.second] = vLpair.first;
    // scale cuts
    float ptScale = 900.0f / minPt;
    graph_building_params.min_delta_phi *= ptScale;
    graph_building_params.dphi_coeff *= ptScale;
    graph_building_params.min_delta_phi_low_dr *= ptScale;
    graph_building_params.dphi_coeff_low_dr *= ptScale;
    graph_building_params.max_Kappa *= ptScale;

    // contianers sizes
    nLayers = static_cast<unsigned int>(layerInfo.type.size());
    TRACCC_INFO("volume layer map has " << volumeToLayerMap_unordered.size()
                                        << " volumes");
    TRACCC_INFO("The maxium volume index in the layer map is "
                << volumeToLayerMap.size());
    TRACCC_INFO("surface to layer map has "
                << surfaceToLayerMap.size() << " barcodes from "
                << split_volumes << " multi-layer volumes");
    TRACCC_INFO("layer info found for " << nLayers << " layers");
    TRACCC_INFO(binTables.size() << " linked layer-eta bins for GBTS");
    TRACCC_INFO("Total number of eta bins: " << n_eta_bins);
    if (nLayers == 0) {
        TRACCC_ERROR("no layers input");
        return false;
    } else if (volumeToLayerMap.size() == 0) {
        TRACCC_ERROR("empty volume to layer map");
        return false;
    } else if (surfaceToLayerMap.size() > SHRT_MAX) {
        TRACCC_ERROR("surface to layer map is to large");
        return false;
    }
    return true;
}

}  // namespace traccc
