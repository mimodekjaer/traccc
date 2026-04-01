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


namespace traccc::device {

TRACCC_HOST_DEVICE
inline void node_sorting_kernel(
    const int idx,
    const collection_types<float4>::const_view d_sp_params_view, 
    const collection_types<int>::const_view d_node_eta_index_view,
    const collection_types<int>::const_view d_node_phi_index_view, 
    const collection_types<int>::view d_phi_cusums_view, 
    const collection_types<float>::view d_node_params_view,
    const collection_types<int>::view d_node_index_view, 
    const collection_types<int>::const_view d_original_sp_idx_view, 
    const gbts_graph_width_cuts width_cuts,
    const unsigned int nPhiBins) {

    if (idx >= nNodes){
        return;
    }

    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    const collection_types<int>::const_device d_node_eta_index(d_node_eta_index_view);
    const collection_types<int>::const_device d_node_phi_index(d_node_phi_index_view);
    collection_types<int>::device d_phi_cusums(d_phi_cusums_view);
    collection_types<float>::device d_node_params(d_node_params_view);
    collection_types<int>::device d_node_index(d_node_index_view);
    const collection_types<int>::const_device d_original_sp_idx(d_original_sp_idx_view);


    float4 sp = d_sp_params[static_cast<unsigned int>(idx)];

    float Phi = atan2f(sp.y, sp.x);
    float r = sqrtf(sp.x * sp.x + sp.y * sp.y);
    float z = sp.z;

    float min_tau = static_cast<float>(-100.0);
    float max_tau = static_cast<float>(100.0);

    if (sp.w > 0) {  // type 0 only
        // linear fit
        min_tau = width_cuts.tMin_slope * (sp.w - width_cuts.offset);
        // linear fit + correction for short clusters
        max_tau = width_cuts.tMax_min + width_cuts.tMax_correction / (sp.w + width_cuts.offset) +
                    width_cuts.tMax_slope * (sp.w - width_cuts.offset);
    }

    int eta_index = d_node_eta_index[static_cast<unsigned int>(idx)];
    int histo_bin = d_node_phi_index[static_cast<unsigned int>(idx)] + static_cast<int>(nPhiBins) * eta_index;

    vecmem::device_atomic_ref<int> d_phi_cusums_ref(d_phi_cusums[static_cast<unsigned int>(histo_bin)]);
    int pos = d_phi_cusums_ref.fetch_add(1);

    int o = 5 * pos;

    d_node_params[static_cast<unsigned int>(o)] = min_tau;
    d_node_params[static_cast<unsigned int>(o + 1)] = max_tau;
    d_node_params[static_cast<unsigned int>(o + 2)] = Phi;
    d_node_params[static_cast<unsigned int>(o + 3)] = r;
    d_node_params[static_cast<unsigned int>(o + 4)] = z;
    // keep the original index of the input spacepoint
    d_node_index[static_cast<unsigned int>(pos)] = d_original_sp_idx[static_cast<unsigned int>(idx)];
    
}


} // namespace traccc::device