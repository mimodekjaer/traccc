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
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// Assign each node a phi-bin index and accumulate the (eta, phi) histogram.
///
/// One thread per node computes phi from (x, y), maps it into the phi-bin
/// range, writes the phi index back, and atomically increments the histogram
/// cell at (eta, phi).
///
/// @param[in]  globalIndex             Current thread index
/// @param[out] d_node_phi_index_view   Phi-bin index per node
/// @param[in]  d_node_eta_index_view   Eta-bin index per node
/// @param[out] d_eta_phi_histo_view    Flat (eta, phi) histogram
/// @param[in]  d_sp_params_view        Layer-ordered (x, y, z, r) per SP
/// @param[in]  nNodes                  Number of nodes
/// @param[in]  nPhiBins                Number of phi bins per eta slice
///
TRACCC_HOST_DEVICE
inline void eta_phi_histo(
    const global_index_t globalIndex,
    const collection_types<int>::view d_node_phi_index_view,
    const collection_types<int>::const_view& d_node_eta_index_view,
    const collection_types<int>::view d_eta_phi_histo_view,
    const collection_types<float4>::const_view& d_sp_params_view,
    const unsigned int nNodes, const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/eta_phi_histo.ipp"
