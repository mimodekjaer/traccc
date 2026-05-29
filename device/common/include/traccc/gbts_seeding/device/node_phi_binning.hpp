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

/// Standalone phi-bin assignment for each node (variant of eta_phi_histo).
///
/// One thread per node computes phi from (x, y) and stores the matching
/// phi-bin index.  Currently provided alongside eta_phi_histo, which
/// fuses this step with histogram accumulation.
///
/// @param[in]  globalIndex                Current thread index
/// @param[in]  d_sp_params_view           Layer-ordered (x, y, z, r) per SP
/// @param[out] d_node_phi_index_view      Phi-bin index per node
/// @param[in]  nNodes                     Number of nodes
/// @param[in]  nPhiBins                   Number of phi bins per eta slice
///
TRACCC_HOST_DEVICE
inline void node_phi_binning(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<int>::view d_node_phi_index_view,
    const unsigned int nNodes,
    const unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/node_phi_binning.ipp"
