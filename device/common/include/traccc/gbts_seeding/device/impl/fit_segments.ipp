/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/device/add_seed_proposal.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cstdint>

// System include(s).
#include <cmath>
#include <cstring>

namespace traccc::device {

namespace gbts_detail {

// ===========================================================================
// edgeState -- Kalman-filter state for a track-segment fit.
// ===========================================================================
// The seed is fitted as TWO decoupled Kalman filters in a 2D frame rotated onto
// the seed's first doublet. node params are float4 = (x, y, z, w) with w the
// cluster width (w < 0 encodes a strip/odd sensor type).
//
// Frame (fixed in initialize(), then carried unchanged through kf_update():
// m_c, m_s are just copied): (m_c, m_s) = (dx, dy)/L is the unit vector of the
// first doublet's chord in the transverse x-y plane. Every hit (x, y, z) is
// expressed in this frame as
//   xi  = x*m_c + y*m_s          (along-chord coordinate)      -> m_refX
//   eta = -x*m_s + y*m_c         (perpendicular / bending)     -> x-measurement
//   r   = hypot(x, y)            (radius)                       -> m_refY
//   z                            (longitudinal)                -> y-measurement
//
// x-filter (bending plane): m_X = [eta, eta', eta''] = [perp position, slope,
//   curvature]; a parabola eta(xi) = eta0 + eta'*A + 0.5*eta''*A^2 stepped by
//   the along-chord distance A = d(xi). m_X[2] is the (conformal) curvature, so
//   it maps to pT and is gated by |m_X[2]| * inv_max_curvature <= 1. m_Cx is its
//   3x3 symmetric covariance.
// y-filter (r-z plane): m_Y = [z, tau] with tau = dz/dr = cot(theta) = sinh(eta);
//   a straight line z(r) = z0 + tau*dr. 1 + tau^2 = 1/sin^2(theta) sets the
//   multiple-scattering path-length factor in kf_update. m_Cy is its 2x2
//   symmetric covariance.
// m_J: accumulated seed quality; each accepted hit adds
//   add_hit - weight_x*dchi2_x - weight_y*dchi2_y.
// m_head_node_type = (node.w < 0): pixel vs strip flag; selects the r-z
//   measurement error (sigma_y vs sigma_y*tau) and the MS length factor.
//
// Retuning for a new geometry -- everything detector-specific reduces to:
//   sig_rphi = bending-plane (r-phi) spacepoint resolution ~ pitch_rphi/sqrt(12)
//   sig_z    = longitudinal resolution ~ pixel z-pitch or strip length / sqrt(12)
//   dr       = radial spacing of the seed doublet's layers
//   B, pT_min = field and target min pT, giving curvature 0.3*B/(2*pT_min)
// sig_rphi/sig_z are the measurement errors sigma_x/sigma_y in the config; the
// initial prior covariances below scale from the same quantities (see initialize).
//
// Packed symmetric covariance (upper triangle), so a flat float[] holds it:
//   m_Cx(i,j) = Cx[i + j + (i!=0 && j!=0)]  ->  3x3 in 6: 00=0 01=1 02=2 11=3 12=4 22=5
//   m_Cy(i,j) = Cy[i + j]                   ->  2x2 in 3: 00=0 01=1 11=2
struct edgeState {

    // Packed accessor for the symmetric 3x3 x-covariance (see header above).
    TRACCC_HOST_DEVICE inline float& m_Cx(const int i, const int j) {
        return Cx[i + j + 1 * (i != 0) * (j != 0)];
    }
    // Packed accessor for the symmetric 2x2 r-z covariance.
    TRACCC_HOST_DEVICE inline float& m_Cy(const int i, const int j) {
        return Cy[i + j];
    }
    TRACCC_HOST_DEVICE inline const float& m_Cx(const int i,
                                                const int j) const {
        return Cx[i + j + 1 * (i != 0) * (j != 0)];
    }
    TRACCC_HOST_DEVICE inline const float& m_Cy(const int i,
                                                const int j) const {
        return Cy[i + j];
    }

    // Seed the filter from the first doublet's two nodes. The state is anchored
    // at node2_params (the reference node); the chord direction and tau come from
    // both nodes. (fit_segments calls initialize(node2, node1), so node1_params
    // here is the outer hit and node2_params the inner reference hit.)
    TRACCC_HOST_DEVICE inline void initialize(
        const traccc::float4& node1_params,
        const traccc::float4& node2_params) {
        m_J = 0.0f;
        m_head_node_type = (node1_params.w < 0);

        // Chord of the doublet in the transverse plane: direction (m_c, m_s).
        const float dx = node1_params.x - node2_params.x;
        const float dy = node1_params.y - node2_params.y;
        const float L = sqrtf(dx * dx + dy * dy);

        const float r1 = sqrtf(node1_params.x * node1_params.x +
                               node1_params.y * node1_params.y);
        const float r2 = sqrtf(node2_params.x * node2_params.x +
                               node2_params.y * node2_params.y);

        // Frozen rotation: m_c = cos, m_s = sin of the chord direction.
        m_s = dy / L;
        m_c = dx / L;

        // Reference point of the fit = the inner node: radius and along-chord xi.
        m_refY = r2;
        m_refX = node2_params.x * m_c + node2_params.y * m_s;

        // x-state = [perp position, slope, curvature]. Position is the reference
        // node's perpendicular (eta) coordinate; slope and curvature are seeded
        // to ZERO = start as a straight track along the chord, to be bent by the
        // hits added in kf_update.
        m_X[0] = -node2_params.x * m_s + node2_params.y * m_c;
        m_X[1] = 0.0f;
        m_X[2] = 0.0f;

        // y-state = [z, tau]. z from the reference node; tau = dz/dr measured
        // directly from the two seed nodes (already a good estimate -> the real
        // slope, not 0).
        m_Y[0] = node2_params.z;
        m_Y[1] = (node1_params.z - node2_params.z) / (r1 - r2);

        // Prior covariances. Each entry is a VARIANCE (prior 1-sigma = sqrt).
        // Off-diagonals start at 0 (uncorrelated). The values are loose priors:
        // > the per-hit measurement variances (sigma_x^2=0.0064, sigma_y^2=0.0625
        // mm^2) so the first real hits dominate the fit. For a new geometry derive
        // them from sig_rphi/sig_z (~pitch/sqrt(12)), the layer spacing dr, and
        // the curvature scale 0.3*B/(2*pT_min) -- see the struct header.
        std::memset(&m_Cx(0, 0), 0, sizeof(Cx));
        std::memset(&m_Cy(0, 0), 0, sizeof(Cy));

        // sigma = 0.5 mm perpendicular position prior ~ a few * sig_rphi
        // (intentionally >> pixel sig_rphi ~ 14 um).
        m_Cx(0, 0) = 0.25f;
        // sigma = 0.032 slope prior (dimensionless) >= sig_rphi / dr.
        m_Cx(1, 1) = 0.001f;
        // sigma = 0.032 curvature prior (conformal, 1/len) >= 0.3*B/(2*pT_min);
        // loose because the bend is unknown at seeding.
        m_Cx(2, 2) = 0.001f;

        // sigma = 1.22 mm z position prior ~ a few * sig_z (looser than the
        // transverse prior because sig_z >> sig_rphi for strips).
        m_Cy(0, 0) = 1.5f;
        // sigma = 0.032 tau (dz/dr) slope prior >= sig_z / dr.
        m_Cy(1, 1) = 0.001f;
    }

    float m_X[3], m_Y[2];
    float m_c, m_s, m_refX, m_refY;
    float m_J;
    bool m_head_node_type;
    float Cx[6];
    float Cy[3];
};

// One Kalman step: predict ts -> new_ts at the next hit (node1_params) -- the
// parabola m_X and line m_Y are extrapolated and their covariances inflated by
// multiple scattering -- then apply the measurement gain update and accumulate
// quality m_J. Returns false (hit rejected) if the x or r-z chi2 exceeds
// maxDChi2, the curvature exceeds 1/inv_max_curvature, or |z0| exceeds max_z0.
TRACCC_HOST_DEVICE inline bool kf_update(
    edgeState* new_ts, const edgeState* ts, const traccc::float4& node1_params,
    const gbts_seed_extraction_params& KF_params) {

    const float tau2 = ts->m_Y[1] * ts->m_Y[1];
    const float invSin2 = 1 + tau2;

    const float lenCorr = (node1_params.w != -1) ? invSin2 : invSin2 / tau2;
    const float minPtFrac = fabsf(ts->m_X[2]) * KF_params.inv_max_curvature;

    const float corrMS = KF_params.sigmaMS * minPtFrac;
    const float sigma2 = KF_params.radLen * lenCorr * corrMS * corrMS;

    const float m_Cx11 = ts->m_Cx(1, 1) + sigma2;
    const float m_Cy11 = ts->m_Cy(1, 1) + sigma2;

    float mx, my;
    const float r = sqrtf(node1_params.x * node1_params.x +
                          node1_params.y * node1_params.y);

    new_ts->m_refX = node1_params.x * ts->m_c + node1_params.y * ts->m_s;
    mx = -node1_params.x * ts->m_s + node1_params.y * ts->m_c;
    new_ts->m_refY = r;
    my = node1_params.z;

    const float A = new_ts->m_refX - ts->m_refX;
    const float B = (0.5f) * A * A;
    const float dr = new_ts->m_refY - ts->m_refY;

    new_ts->m_X[0] = ts->m_X[0] + ts->m_X[1] * A + ts->m_X[2] * B;
    new_ts->m_X[1] = ts->m_X[1] + ts->m_X[2] * A;
    new_ts->m_X[2] = ts->m_X[2];

    new_ts->m_Cx(0, 0) = ts->m_Cx(0, 0) + 2 * ts->m_Cx(0, 1) * A +
                         2 * ts->m_Cx(0, 2) * B + A * m_Cx11 * A +
                         2 * A * ts->m_Cx(1, 2) * B + B * ts->m_Cx(2, 2) * B;
    new_ts->m_Cx(0, 1) = ts->m_Cx(0, 1) + m_Cx11 * A + ts->m_Cx(1, 2) * B +
                         ts->m_Cx(0, 2) * A + A * ts->m_Cx(1, 2) * A +
                         A * ts->m_Cx(2, 2) * B;
    new_ts->m_Cx(0, 2) =
        ts->m_Cx(0, 2) + ts->m_Cx(1, 2) * A + ts->m_Cx(2, 2) * B;
    new_ts->m_Cx(1, 1) =
        m_Cx11 + 2 * A * ts->m_Cx(1, 2) + A * ts->m_Cx(2, 2) * A;
    new_ts->m_Cx(1, 2) = ts->m_Cx(1, 2) + ts->m_Cx(2, 2) * A;
    new_ts->m_Cx(2, 2) = ts->m_Cx(2, 2);

    new_ts->m_Y[0] = ts->m_Y[0] + ts->m_Y[1] * dr;
    new_ts->m_Y[1] = ts->m_Y[1];
    new_ts->m_Cy(0, 0) =
        ts->m_Cy(0, 0) + 2 * ts->m_Cy(0, 1) * dr + dr * m_Cy11 * dr;
    new_ts->m_Cy(0, 1) = ts->m_Cy(0, 1) + dr * m_Cy11;
    new_ts->m_Cy(1, 1) = m_Cy11;

    const float resid_x = mx - new_ts->m_X[0];
    const float resid_y = my - new_ts->m_Y[0];

    float sigma_rz = 0;
    if (!ts->m_head_node_type) {
        sigma_rz = KF_params.sigma_y;
    } else {
        sigma_rz = KF_params.sigma_y * ts->m_Y[1];
    }

    const float inv_Dx =
        new_ts->m_Cx(0, 0) + KF_params.sigma_x * KF_params.sigma_x;
    const float Dx = 1 / inv_Dx;
    const float Dy = 1 / (new_ts->m_Cy(0, 0) + sigma_rz * sigma_rz);

    const float dchi2_x = resid_x * resid_x * Dx;
    const float dchi2_y = resid_y * resid_y * Dy;

    if (dchi2_x > KF_params.maxDChi2_x || dchi2_y > KF_params.maxDChi2_y) {
        return false;
    }

    new_ts->m_J = ts->m_J + (KF_params.add_hit - dchi2_x * KF_params.weight_x -
                             dchi2_y * KF_params.weight_y);

    for (unsigned int i = 0u; i < 3u; i++) {
        new_ts->m_X[i] += Dx * new_ts->m_Cx(0, static_cast<int>(i)) * resid_x;
    }

    if (fabsf(new_ts->m_X[2]) * KF_params.inv_max_curvature > 1.0f) {
        return false;
    }

    for (unsigned int i = 0u; i < 2u; i++) {
        new_ts->m_Y[i] += Dx * new_ts->m_Cy(0, static_cast<int>(i)) * resid_y;
    }

    const float z0 = new_ts->m_Y[0] - new_ts->m_refY * ts->m_Y[1];
    if (fabsf(z0) > KF_params.max_z0) {
        return false;
    }

    new_ts->m_Cx(2, 2) = Dx * (new_ts->m_Cx(2, 2) * inv_Dx -
                               new_ts->m_Cx(0, 2) * new_ts->m_Cx(0, 2));
    new_ts->m_Cx(1, 2) = Dx * (new_ts->m_Cx(1, 2) * inv_Dx -
                               new_ts->m_Cx(0, 1) * new_ts->m_Cx(0, 2));
    new_ts->m_Cx(1, 1) = Dx * (new_ts->m_Cx(1, 1) * inv_Dx -
                               new_ts->m_Cx(0, 1) * new_ts->m_Cx(0, 1));
    new_ts->m_Cx(0, 2) = Dx * (new_ts->m_Cx(0, 2) * inv_Dx -
                               new_ts->m_Cx(0, 0) * new_ts->m_Cx(0, 2));
    new_ts->m_Cx(0, 1) = Dx * (new_ts->m_Cx(0, 1) * inv_Dx -
                               new_ts->m_Cx(0, 0) * new_ts->m_Cx(0, 1));
    new_ts->m_Cx(0, 0) *= Dx * (KF_params.sigma_x * KF_params.sigma_x);

    new_ts->m_Cy(1, 1) -= Dy * new_ts->m_Cy(0, 1) * new_ts->m_Cy(0, 1);
    new_ts->m_Cy(0, 1) -= Dy * new_ts->m_Cy(0, 0) * new_ts->m_Cy(0, 1);
    new_ts->m_Cy(0, 0) -= Dy * new_ts->m_Cy(0, 0) * new_ts->m_Cy(0, 0);

    new_ts->m_c = ts->m_c;
    new_ts->m_s = ts->m_s;
    new_ts->m_head_node_type = (node1_params.w < 0);

    return true;
}

}  // namespace gbts_detail

TRACCC_HOST_DEVICE
inline void fit_segments(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_reduced_view,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    unsigned int& nPropsCounter,
    const unsigned int nTerminusEdges, const unsigned char minLevel,
    const unsigned int max_num_neighbours,
    const gbts_seed_extraction_params& seed_extraction_params) {

    const collection_types<float4>::const_device d_sp_reduced(
        d_sp_reduced_view);
    const collection_types<unsigned int>::const_device d_output_graph(
        d_output_graph_view);
    const collection_types<int2>::const_device d_path_store(d_path_store_view);

    const unsigned int path_idx = globalIndex + nTerminusEdges;
    // Row-major output graph: each edge owns a contiguous block of
    // edge_size = 2 + 1 + max_num_neighbours ints.
    const unsigned int edge_size = 2u + 1u + max_num_neighbours;

    unsigned char length = 1;
    bool toggle = false;
    gbts_detail::edgeState state1;
    gbts_detail::edgeState state2;

    int2 path = d_path_store[path_idx];

    const unsigned int nodeidx1 =
        d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                       gbts_consts::node1];
    traccc::float4 node1 = d_sp_reduced[nodeidx1];
    const unsigned int nodeidx2 =
        d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                       gbts_consts::node2];
    traccc::float4 node2 = d_sp_reduced[nodeidx2];

    state1.initialize(node2, node1);
    while (path.y >= 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        node2 =
            d_sp_reduced[d_output_graph[edge_size *
                                            static_cast<unsigned int>(path.x) +
                                        gbts_consts::node2]];
        if (toggle) {
            if (!gbts_detail::kf_update(&state1, &state2, node2,
                                        seed_extraction_params)) {
                state1 = state2;
                break;
            }
        } else if (!gbts_detail::kf_update(&state2, &state1, node2,
                                           seed_extraction_params)) {
            break;
        }
        toggle = !toggle;
        length++;
    }
    if (length < minLevel) {
        return;
    }
    int qual = 0;
    if (toggle) {
        qual = static_cast<int>(seed_extraction_params.qual_scale * state2.m_J);
    } else {
        qual = static_cast<int>(seed_extraction_params.qual_scale * state1.m_J);
    }
    const unsigned int prop_idx =
        vecmem::device_atomic_ref<unsigned int>(nPropsCounter).fetch_add(1u);
    add_seed_proposal(qual, static_cast<int>(path_idx), prop_idx,
                      d_seed_ambiguity_view, d_seed_proposals_view,
                      d_edge_bids_view, d_path_store_view, 1);
}

}  // namespace traccc::device
