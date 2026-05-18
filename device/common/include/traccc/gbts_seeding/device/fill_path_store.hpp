/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void fill_path_store(
    unsigned int blockIndex, unsigned int threadIndex, unsigned int blockSize,
    const barrier_t& barrier, traccc::int2* live_paths, int& n_live_paths,
    collection_types<int2>::view d_path_store_view,
    const collection_types<int>::const_view& d_output_graph_view,
    const collection_types<char>::const_view& d_levels_view,
    unsigned int& nPathStoreSizeCounter, unsigned int nTerminus,
    unsigned int nTerminusPerBlock, unsigned int max_num_neighbours,
    unsigned int nPaths);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/fill_path_store.ipp"
