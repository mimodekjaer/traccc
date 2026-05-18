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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void fill_path_store(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize, const barrier_t& barrier,
    traccc::int2* live_paths, int& n_live_paths,
    collection_types<int2>::view d_path_store_view,
    const collection_types<int>::const_view& d_output_graph_view,
    const collection_types<char>::const_view& d_levels_view,
    unsigned int& nPathStoreSizeCounter,
    const unsigned int nTerminus, const unsigned int nTerminusPerBlock,
    const unsigned int max_num_neighbours, const unsigned int nPaths) {

    collection_types<int2>::device d_path_store(d_path_store_view);
    const collection_types<int>::const_device d_output_graph(
        d_output_graph_view);
    const collection_types<char>::const_device d_levels(d_levels_view);

    if (threadIndex == 0) {
        n_live_paths = 0;
    }
    barrier.blockBarrier();

    const unsigned int edge_size = 2u + 1u + max_num_neighbours;
    unsigned int path_idx = threadIndex + blockIndex * nTerminusPerBlock;

    if (threadIndex < nTerminusPerBlock && path_idx < nTerminus) {
        const int2 path = d_path_store[path_idx];
        const int nNei = d_output_graph[traccc::device::gbts_consts::nNei +
                                        edge_size *
                                            static_cast<unsigned int>(path.x)];
        const char level = d_levels[static_cast<unsigned int>(path.x)];
        for (int nei = 0; nei < nNei; ++nei) {
            const int edge_idx =
                d_output_graph[traccc::device::gbts_consts::nei_start +
                               static_cast<unsigned int>(nei) +
                               edge_size *
                                   static_cast<unsigned int>(path.x)];
            if (level != d_levels[static_cast<unsigned int>(edge_idx)] + 1) {
                continue;
            }
            const int live_idx =
                vecmem::device_atomic_ref<int>(n_live_paths).fetch_add(1);
            if (live_idx >=
                static_cast<int>(traccc::device::gbts_consts::live_path_buffer)) {
                break;
            }
            const unsigned int new_path_idx =
                vecmem::device_atomic_ref<unsigned int>(nPathStoreSizeCounter)
                    .fetch_add(1u);
            d_path_store[new_path_idx] =
                make_int2(edge_idx, static_cast<int>(path_idx));
            live_paths[static_cast<unsigned int>(live_idx)] =
                make_int2(edge_idx, static_cast<int>(new_path_idx));
        }
    }
    barrier.blockBarrier();

    int2 path = make_int2(0, 0);
    bool has_path = false;

    while (n_live_paths > 0) {
        has_path = false;
        if (threadIndex == 0) {
            const int buf_size = static_cast<int>(
                traccc::device::gbts_consts::live_path_buffer);
            n_live_paths = (n_live_paths < buf_size) ? n_live_paths : buf_size;
        }
        barrier.blockBarrier();
        if (static_cast<int>(threadIndex) < n_live_paths) {
            path = live_paths[static_cast<unsigned int>(
                n_live_paths - static_cast<int>(threadIndex) - 1)];
            has_path = true;
        }
        barrier.blockBarrier();
        if (threadIndex == 0) {
            n_live_paths = (n_live_paths < static_cast<int>(blockSize))
                               ? 0
                               : n_live_paths - static_cast<int>(blockSize);
        }
        barrier.blockBarrier();
        if (has_path) {
            const int nNei =
                d_output_graph[traccc::device::gbts_consts::nNei +
                               edge_size *
                                   static_cast<unsigned int>(path.x)];
            const char level = d_levels[static_cast<unsigned int>(path.x)];
            for (int nei = 0; nei < nNei; ++nei) {
                const int edge_idx =
                    d_output_graph[traccc::device::gbts_consts::nei_start +
                                   static_cast<unsigned int>(nei) +
                                   edge_size *
                                       static_cast<unsigned int>(path.x)];
                if (level !=
                    d_levels[static_cast<unsigned int>(edge_idx)] + 1) {
                    continue;
                }
                path_idx =
                    vecmem::device_atomic_ref<unsigned int>(
                        nPathStoreSizeCounter)
                        .fetch_add(1u);
                if (path_idx >= nPaths) {
                    break;
                }
                const int live_idx =
                    vecmem::device_atomic_ref<int>(n_live_paths).fetch_add(1);
                if (live_idx >= static_cast<int>(
                                    traccc::device::gbts_consts::live_path_buffer)) {
                    break;
                }
                d_path_store[path_idx] = make_int2(edge_idx, path.y);
                live_paths[static_cast<unsigned int>(live_idx)] =
                    make_int2(edge_idx, static_cast<int>(path_idx));
            }
        }
        barrier.blockBarrier();
    }
}

}  // namespace traccc::device
