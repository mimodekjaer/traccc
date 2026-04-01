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


// TODO not finished

namespace traccc::device {


TRACCC_HOST_DEVICE
inline void add_terminus_to_path_store_kernel(
    const global_index_t global_index,
    const collection_types<int2>::view d_path_store_view,
    const collection_types<short2>::const_view d_outgoing_paths_view,
    const unsigned int nEdges
) {
    if (global_index >= nEdges) {
        return;
    }
    collection_types<int2>::device d_path_store(d_path_store_view);
    collection_types<short2>::const_device d_outgoing_paths(d_outgoing_paths_view);

short2 out_paths = d_outgoing_paths[global_index];
if (out_paths.y == -1) {
    return;
    }
// -1 flags as the terminus of a path
d_path_store[static_cast<unsigned int>(out_paths.y)] = make_int2(static_cast<int>(global_index), -1);
}



} // namespace traccc::device