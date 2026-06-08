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

    void addLayer(char layerType, unsigned int firstBin, unsigned int nBins,
                  float minEta, float etaBinWidth) {
        type.push_back(layerType);
        info.push_back(std::make_pair(firstBin, nBins));
        geo.push_back(std::make_pair(minEta, etaBinWidth));
    }
};

// Named indices into the flat device counter buffer, mirroring the layout in
// traccc/gbts_changes. One memset zeros all of them.
enum gbts_counter : unsigned int {
    nEdges,           // edges created by graph_edge_making
    nConnections,     // edge-to-edge connections from graph_edge_matching
    nConnectedEdges,  // edges kept after edge_re_indexing
    nEdgesLeft,       // edges remaining for CCA (kept for reference parity)
    nPaths,           // total paths reachable from any terminus edge
    nTerminusEdges,   // #terminus edges; then reused as path-store write cursor
    nProps,           // seed proposals from fit_segments
    nRejected,        // rejected seed proposals
    nCounters         // total number of counters
};

struct gbts_consts {

    // CCA max iterations -> maximum seed length (in edges). Tuning / hard cap.
    static constexpr unsigned short max_cca_iter = 15;
    // shared memory allocation sizes (element counts per block). Algorithmic.
    static constexpr unsigned short node_buffer_length = 128;
    static constexpr unsigned short live_path_buffer = 1024;

    // Per-edge offsets into the row-major output graph
    // (each edge occupies edge_size = 2 + 1 + max_num_neighbours ints).
    static constexpr unsigned char node1 = 0;
    static constexpr unsigned char node2 = 1;
    static constexpr unsigned char nNei = 2;
    static constexpr unsigned char nei_start = 3;
};

}  // namespace traccc::device

namespace traccc {

// Tau-prediction cuts read by device::node_sorting.
//
// tau = cot(theta) = sinh(eta) is predicted from the cluster width w (= the SP's
// stored width). node_sorting bounds tau by two empirical lines in w:
//   min_tau = tMin_slope * (w - offset)
//   max_tau = tMax_min + tMax_correction / (w + offset) + tMax_slope * (w - offset)
// The slopes/offsets have no closed form from pT: they are a calibration of the
// "cluster width grows with track incidence angle" relation, i.e. of w vs tau,
// fixed by the sensor geometry (thickness / pitch).
struct gbts_node_sorting_params {
    // Slope of the lower-bound tau line: min_tau = tMin_slope * (w - offset).
    float tMin_slope = 6.7f;
    // Minimum cluster width: offset = w_min (a ~1-cluster-cell width), subtracted
    // in both tau lines so a minimal cluster predicts tau ~ 0.
    float offset = 0.2f;
    // Asymptotic lower bound on the upper-tau line (w -> large).
    float tMax_min = 1.6f;
    // Inverse-width correction term on the upper-tau line: tMax_correction/(w+offset).
    float tMax_correction = 0.15f;
    // Slope of the upper-tau line: ... + tMax_slope * (w - offset).
    float tMax_slope = 6.1f;
    // |tau| acceptance for nodes without a usable cluster width.
    //   sinh(|eta_max|) = maxTau   ->   |eta_max| = asinh(36) = 4.28
    // i.e. the ~4.3 eta detector acceptance.
    float maxTau = 36.0f;
    // Opt-in: use a tau lookup table instead of the analytic formula above.
    // The LUT is laid out as [w_bin_edge, min_tau, max_tau, ...] per bin.
    bool useTauLUT = false;
    // Inverse cluster-width bin size used to index the LUT.
    float tau_lut_inv_bin = 0.0f;
    // Number of float entries in the LUT (bounds the index).
    unsigned int tauLutSize = 0;
};

// Geometric / kinematic edge-making cuts read by device::graph_edge_making.
//
// For an edge between two nodes (r1,z1)->(r2,z2): tau = dz/dr = cot(theta) =
// sinh(eta); curv = dphi/dr = kappa/2 (chord geometry of a track from the
// origin: phi(r) = asin(kappa*r/2), so dphi/dr -> kappa/2). The magnetic
// rigidity kappa = 1/R with R[m] = pT/(0.3*B), where 0.3 = 0.299792458 =
// c*1e-9 [GeV/(T*m*e)] from p[GeV] = q*B*R*c.
struct gbts_edge_making_params {
    // Two nodes must be radially separated to form an edge: dr >= minDeltaRadius
    // (mm). Geometry input: just above the module/double-layer thickness so two
    // hits are never treated as coplanar.
    float minDeltaRadius = 2.0f;
    // z at the beamline: z0 = z1 - r1*tau, required min_z0 <= z0 <= max_z0.
    // +/-160 mm = chosen luminous-region (beamspot) half-length.
    float min_z0 = -160.0f;
    float max_z0 = 160.0f;
    // Outer radius (mm) to which the edge is extrapolated for the ROI z cut.
    // Geometry input (~ outer pixel radius).
    float maxOuterRadius = 350.0f;
    // ROI band for zouter = z0 + maxOuterRadius*tau, required in [cut_zMinU,
    // cut_zMaxU]:  cut_zMin/MaxU = -/+ (|z0|_max + maxOuterRadius * tau_roi),
    // tau_roi = 45 (chosen, slightly above maxTau = 36 for margin).
    float cut_zMinU = min_z0 - maxOuterRadius * 45.0f;
    float cut_zMaxU = max_z0 + maxOuterRadius * 45.0f;
    // Maximum edge curvature = minimum pT:
    //   max_Kappa = curv_max = kappa/2 = 0.3*B/(2*pT)
    //   = 0.299792458*2/(2*0.9) m^-1 = 0.333 m^-1 = 3.33e-4 mm^-1  (2 T, 0.9 GeV).
    // (0.3 = c*1e-9 rigidity, /2 = chord geometry.)
    float max_Kappa = 3.75e-4f;
    // Max transverse impact parameter, per curvature regime:
    //   d0 = r1*r2*(|curv| - max_Kappa) <= {low,high}_Kappa_d0  (mm).
    // 0 = prompt tracks only (no d0 allowance beyond the pT cut).
    float low_Kappa_d0 = 0.00f;
    float high_Kappa_d0 = 0.0f;
};

// Pair-matching cuts read by device::graph_edge_matching. For the two edges of a
// triplet sharing the middle node, these bound how much the edge parameters may
// differ -- the spread is set by multiple scattering at pT_min, so the values are
// tuned (no clean closed form), but the cut expressions themselves are exact.
struct gbts_edge_matching_params {
    // |dphi_1 - dphi_2| <= cut_dphi_max  (azimuthal consistency, rad).
    float cut_dphi_max = 0.012f;
    // |curv_1 - curv_2| <= cut_dcurv_max  (curvature consistency, curv = dphi/dr).
    float cut_dcurv_max = 0.001f;
    // |tau_2/tau_1 - 1| <= cut_tau_ratio_max  (polar-angle consistency, ~1%).
    float cut_tau_ratio_max = 0.01f;
};

// Host-side dphi window used to compute bin_pair_dphi before launching
// device::graph_edge_making (not passed to any kernel).
//
// The azimuthal search window between two eta-bins separated by dr (mm) is
//   deltaPhi = min_delta_phi + dphi_coeff * dr            (dr >= low_dr_threshold)
//   deltaPhi = min_delta_phi_low_dr + dphi_coeff_low_dr*dr (dr <  low_dr_threshold)
// The dr-slope is the azimuthal bend per unit dr at pT_min,
//   dphi_coeff = curv = kappa/2 = 0.3*B/(2*pT) = 3.33e-4 mm^-1  (2 T, 0.9 GeV),
// and the constant floor absorbs position resolution + scattering over a doublet.
// The two regimes use tuned coefficients (margin differs at small vs large dr).
struct gbts_dphi_window_params {
    // dr-independent floor (rad) and dr-slope (rad/mm ~ kappa/2) for the window.
    float min_delta_phi = 0.015f;
    float dphi_coeff = 2.2e-4f;
    // Tighter floor + slope used when delta-R is below low_dr_threshold.
    float min_delta_phi_low_dr = 0.002f;
    float dphi_coeff_low_dr = 4.33e-4f;
    // delta-R (mm) below which the tighter "low dr" window is used.
    float low_dr_threshold = 60.0f;
};

// Conformal-fit Kalman-filter cuts read by device::fit_segments.
struct gbts_seed_extraction_params {
    // Per-layer multiple-scattering angle, Highland formula:
    //   sigmaMS = 13.6 MeV / pT = 13.6/900 = 0.0151  (900 MeV, eta=0).
    // 13.6 MeV is the Highland constant (PDG empirical fit; the full form is
    // theta0 = (13.6 MeV/(beta*c*p))*z*sqrt(x/X0)*[1+0.038*ln(x/X0)], beta~1,
    // z=1; the sqrt(x/X0) length factor is applied separately via radLen).
    float sigmaMS = 0.016f;
    // Material per layer in radiation lengths: radLen = x/X0 = 2.5% (detector
    // input). Enters the MS covariance as radLen*sigmaMS^2.
    float radLen = 0.025f;

    // Measurement resolutions in the conformal (x) and r-z (y) coordinates:
    //   sigma = pitch / sqrt(12)  (RMS of a hit uniform over a cell of width
    // pitch; variance = pitch^2/12). Effective pitches differ -> 0.08 / 0.25 mm.
    float sigma_x = 0.08f;
    float sigma_y = 0.25f;

    // Relative weights of the x / r-z chi2 terms in the seed quality. Tuning.
    float weight_x = 0.5f;
    float weight_y = 0.5f;

    // Per-hit chi2 acceptance in x / r-z (a few sigma^2). Tuning.
    float maxDChi2_x = 5.0f;
    float maxDChi2_y = 6.0f;
    // controls if seeds of shorter lengths
    // can win bidding against longer seeds. Tuning.
    float add_hit = 14.0f;
    // seed quality is an int scaled up from a float
    // max qual = add_hit*max_length*qual_scale
    // Exact int-scaling so the longest seed maps to ~1% of INT_MAX:
    //   qual_scale = 0.01 * INT_MAX / (add_hit * max_cca_iter)
    // (0.01 = chosen headroom; INT_MAX = int range; max_cca_iter = max length).
    float qual_scale =
        0.01f * static_cast<float>(INT_MAX) /
        (add_hit *
         static_cast<float>(traccc::device::gbts_consts::max_cca_iter));

    // Minimum-pT gate in the fit: reject if |X2| * inv_max_curvature > 1, i.e.
    // inv_max_curvature = 1/curv_max. The conformal curvature state X2 scales as
    // ~1/pT[MeV], so inv_max_curvature = pT_min[MeV] = 900.
    float inv_max_curvature = 900.0f;
    // Beamspot half-length (mm), same as edge_making: |z0| <= max_z0.
    float max_z0 = 160.0f;
};

struct gbts_seed_ambi_params {
    // sample multiple triplets when forming seeds to hedge against outliers.
    // Tuning.
    bool use_dropout = true;
    // Curvature thresholds (1/m) for the dropout logic. Convert to pT with the
    // rigidity relation pT = 0.3*B/kappa (0.3 = c*1e-9): at B = 2 T,
    //   0.007 /m -> pT ~ 86 GeV,  0.03 /m -> pT ~ 20 GeV.
    // i.e. near-straight, high-pT regime; the thresholds themselves are tuned.
    float dropout_dcurv_m = 0.007f;
    float force_dropout_max_curv_m = 0.03f;
    // Fraction of shared hits above which a seed loses a bid (~1/2). Tuning.
    float best_hit_frac = 0.49f;
    // Central-region switch for "tight" bidding:
    //   cot(theta) = sinh(eta) = tight_bid_cot_threshold
    //   -> |eta| < asinh(1) = 0.88   (theta > 45 deg).
    float tight_bid_cot_threshold = 1.0f;
};

struct gbts_sp_counting_params {
    // Maximum cluster width allowed on "type 1" (barrel) layers, passed as a
    // scalar to device::count_sp_by_layer. Geometry input (same scale as the
    // node_sorting `offset` minimum cluster width); wider clusters are dropped
    // as not from prompt, high-pT tracks.
    float type1_max_width = 0.2f;
    // If true, apply the cluster-width / tau cut at SP-counting time. Algorithmic.
    bool doTauCut = true;
};

struct gbts_seedfinder_config {
    bool setLinkingScheme(
        const std::vector<std::pair<unsigned int, std::vector<unsigned int>>>&
            binTables,
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

    // Optional tau lookup table consumed by device::node_sorting when
    // node_sorting.useTauLUT is set. Layout: [w_bin_edge, min_tau, max_tau,
    // ...] per cluster-width bin. Empty by default (analytic formula is used).
    std::vector<float> tau_lut{};

    // node making bin counts
    unsigned int n_eta_bins = 0;  // calculated from input layerInfo (geometry)
    // Phi bin width = 2*pi/n_phi_bins = 2*pi/128 = 0.049 rad (>= the max
    // azimuthal doublet bend so neighbours fall in adjacent bins). 2*pi = full
    // azimuth.
    unsigned int n_phi_bins = 128;
    // graph making maxiums: max neighbours kept per edge (memory/occupancy).
    // Tuning.
    unsigned int max_num_neighbours = 10;
    // graph extraction cuts: minimum seed length as a CCA level,
    //   nSP_seed = minLevel + 1  (minLevel = #edges, spacepoints = +1).
    unsigned char minLevel =
        3;  // equivlent to a cut of #seed edges or #spacepoints-1
    // maxium number of edges to be created per node(spacepoint):
    //   nMaxEdges = max_edges_factor * nNodes  (buffer sizing). Tuning.
    unsigned int max_edges_factor = 10;
    // number of seed-vs-edge bidding rounds during disambiguation (convergence).
    // Tuning.
    unsigned int edge_bidding_rounds = 5;
};

}  // namespace traccc
