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
inline void edge_re_indexing_kernel(
    const global_index_t global_index,
    const collection_types<int>::view d_reIndexer_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nEdges
) {

    if (global_index >= nEdges) {
        return;
    }

    collection_types<int>::device d_reIndexer(d_reIndexer_view);
    collection_types<unsigned int>::device d_counters(d_counters_view);
    
    if (d_reIndexer[global_index] == -1) {
        return;
    }
    vecmem::device_atomic_ref<unsigned int> d_counters_2_ref(d_counters[2]);
    d_reIndexer[global_index] = static_cast<int>(d_counters_2_ref.fetch_add(1));
}




} // namespace traccc::device