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
inline void count_terminus_edges_kernel(
    const global_index_t global_index,
    const collection_types<short2>::view d_outgoing_paths_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nEdges
) {

if (global_index >= nEdges) {
    return;
}

collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);
collection_types<unsigned int>::device d_counters(d_counters_view);

// count in shared first to reduce global atomics
int outgoingCount = 0;


short2 out_paths = d_outgoing_paths[global_index];
// only count terminus edges that could lead to a seed
// fill the first part of path_store so fitting can skip go-nowhere
// paths
if (out_paths.y != -1) {
    vecmem::device_atomic_ref<unsigned int> d_counters_7_ref(d_counters[7]);
    d_outgoing_paths[global_index].y = static_cast<short>(d_counters_7_ref.fetch_add(1));
    outgoingCount += out_paths.x;
}

vecmem::device_atomic_ref<unsigned int> d_counters_6_ref(d_counters[6]);
d_counters_6_ref.fetch_add(static_cast<unsigned int>(outgoingCount));

}



} // namespace traccc::device