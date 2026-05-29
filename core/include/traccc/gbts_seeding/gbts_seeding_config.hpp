/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s)
#include <memory>

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/utils/messaging.hpp"

// Detray include(s).
#include <detray/geometry/identifier.hpp>

namespace traccc::device {

struct gbts_layerInfo {
    std::vector<char> type;
    // etaBin0 and numBins
    std::vector<std::pair<unsigned int, unsigned int>> info;
    // minEta and deltaEta
    std::vector<std::pair<float, float>> geo;

    void reserve(unsigned int n) {
        type.reserve(n);
        info.reserve(n);
        geo.reserve(n);
    }

    void addLayer(char layerType, unsigned int firstBin, unsigned int nBins, float minEta,
                  float etaBinWidth) {
        type.push_back(layerType);
        info.push_back(std::make_pair(firstBin, nBins));
        geo.push_back(std::make_pair(minEta, etaBinWidth));
    }
};

struct gbts_consts {

    // CCA max iterations -> maxium seed length
    static constexpr unsigned short max_cca_iter = 15;
    // shared memory allocation sizes
    static constexpr unsigned short node_buffer_length = 128;
    static constexpr unsigned short live_path_buffer = 1024;

    // access into output graph (column indices in the SoA layout)
    static constexpr char node1 = 0;
    static constexpr char node2 = 1;
    static constexpr char nNei = 2;
    static constexpr char nei_start = 3;
};

/// SoA index into the compacted output_graph buffer.
///
/// The compacted graph is laid out column-major over edges: each of the
/// (2 + 1 + nMaxNei) columns owns a contiguous array of `stride` entries
/// (where `stride == nConnectedEdges`). Adjacent threads of a warp that hold
/// adjacent edge indices therefore see contiguous addresses for the same
/// column, giving coalesced global loads. (Row-major would put adjacent
/// threads `edge_size * 4` bytes apart and force a separate sector per
/// thread.)
TRACCC_HOST_DEVICE inline unsigned int gbts_og_index(
    unsigned int col, unsigned int edge, unsigned int stride) {
    return col * stride + edge;
}

}  // namespace traccc::device

namespace traccc {

// Tau-prediction cuts read by device::node_sorting.
struct gbts_node_sorting_params {
    // Slope of the lower-bound tau line.
    float tMin_slope = 6.7f;
    // Cluster-width offset applied to both tau bounds.
    float offset = 0.2f;
    // Asymptotic lower bound on the upper-tau line.
    float tMax_min = 1.6f;
    // Inverse-width correction term on the upper-tau line.
    float tMax_correction = 0.15f;
    // Slope of the upper-tau line.
    float tMax_slope = 6.1f;
};

// Geometric / kinematic edge-making cuts read by device::graph_edge_making.
struct gbts_edge_making_params {
    // Minimum radius difference between two nodes to form an edge.
    float minDeltaRadius = 2.0f;
    // Allowed range on the impact-parameter z0.
    float min_z0 = -160.0f;
    float max_z0 = 160.0f;
    // Outer-radius used to extrapolate the edge for the ROI dz cut.
    float maxOuterRadius = 350.0f;
    // Derived z-range of interest at the outer radius.
    float cut_zMinU = min_z0 - maxOuterRadius * 45.0f;
    float cut_zMaxU = max_z0 + maxOuterRadius * 45.0f;
    // Maximum curvature allowed for an edge.
    float max_Kappa = 3.75e-4f;
    // Per-curvature-regime d0 cuts.
    float low_Kappa_d0 = 0.00f;
    float high_Kappa_d0 = 0.0f;
};

// Pair-matching cuts read by device::graph_edge_matching.
struct gbts_edge_matching_params {
    // Maximum phi difference between two edges sharing a node.
    float cut_dphi_max = 0.012f;
    // Maximum curvature difference between two edges.
    float cut_dcurv_max = 0.001f;
    // Maximum allowed tau-ratio difference between two edges.
    float cut_tau_ratio_max = 0.01f;
};

// Host-side dphi window used to compute bin_pair_dphi before launching
// device::graph_edge_making (not passed to any kernel).
struct gbts_dphi_window_params {
    // Baseline phi window and dr-dependent slope.
    float min_delta_phi = 0.015f;
    float dphi_coeff = 2.2e-4f;
    // Tighter window used when delta-R is small (< 60 mm).
    float min_delta_phi_low_dr = 0.002f;
    float dphi_coeff_low_dr = 4.33e-4f;
};

struct gbts_seed_extraction_params {
    // for 900 MeV track at eta=0
    float sigmaMS = 0.016f;
    // 2.5% per layer
    float radLen = 0.025f;

    float sigma_x = 0.08f;
    float sigma_y = 0.25f;

    float weight_x = 0.5f;
    float weight_y = 0.5f;

    float maxDChi2_x = 5.0f;
    float maxDChi2_y = 6.0f;
    // controls if seeds of shorter lengths
    // can win bidding against longer seeds
    float add_hit = 14.0f;
    // seed quality is an int scaled up from a float
    // max qual = add_hit*max_length*qual_scale
    float qual_scale =
        0.01f * static_cast<float>(INT_MAX) /
        (add_hit *
         static_cast<float>(traccc::device::gbts_consts::max_cca_iter));

    float inv_max_curvature = 900.0f;
    float max_z0 = 160.0f;
};

struct gbts_seed_ambi_params {
    // sample multiple triplets when forming seeds to hedge against outliers
    bool use_dropout = true;
    // these curvatures are in 1/m
    float dropout_dcurv_m = 0.007f;
    float force_dropout_max_curv_m = 0.03f;
    float best_hit_frac = 0.49f;
    float tight_bid_cot_threshold = 1.0f;
};

struct gbts_sp_counting_params {
    // Maximum cluster width allowed on "type 1" (barrel) layers,
    // passed as a scalar to device::count_sp_by_layer.
    float type1_max_width = 0.2f;
    // If true, apply the cluster-width / tau cut at SP-counting time.
    bool doTauCut = true;
};

struct gbts_seedfinder_config {
    bool setLinkingScheme(
        const std::vector<std::pair<unsigned int, std::vector<unsigned int>>>& binTables,
        const device::gbts_layerInfo layerInfo,
        std::vector<std::pair<uint64_t, int16_t>>& detrayGeoIDBinning,
        const float minPt, std::unique_ptr<const traccc::Logger> logger);

    // layer linking and geometry
    std::vector<std::pair<unsigned int, unsigned int>> binTables{};
    traccc::device::gbts_layerInfo layerInfo{};
    unsigned int nLayers = 0;

    std::vector<int16_t> volumeToLayerMap{};
    std::vector<std::pair<unsigned int, unsigned int>> surfaceToLayerMap{};

    // Per-kernel parameter blocks (passed by value to the respective kernels),
    // tuned for a 900 MeV pT cut and scaled by the input minPt.
    gbts_node_sorting_params node_sorting{};
    gbts_edge_making_params edge_making{};
    gbts_edge_matching_params edge_matching{};
    // Host-only — used to compute bin_pair_dphi before launching
    // device::graph_edge_making.
    gbts_dphi_window_params dphi_window{};
    gbts_sp_counting_params sp_counting_params{};
    gbts_seed_extraction_params seed_extraction_params{};
    gbts_seed_ambi_params seed_ambi_params{};

    // node making bin counts
    unsigned int n_eta_bins = 0;  // calculated from input layerInfo
    unsigned int n_phi_bins = 128;
    // graph making maxiums
    unsigned int max_num_neighbours = 10;
    // graph extraction cuts
    unsigned char minLevel = 3;  // equivlent to a cut of #seed edges or #spacepoints-1
    // maxium number of edges to be created per node(spacepoint)
    unsigned int max_edges_factor = 10;
};

}  // namespace traccc
