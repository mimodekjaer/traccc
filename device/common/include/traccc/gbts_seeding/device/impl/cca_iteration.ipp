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
TRACCC_HOST_DEVICE inline void cca_iteration(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize, const unsigned int gridSize,
    const barrier_t& barrier,
    const collection_types<int>::const_view& d_output_graph_view,
    collection_types<char>::view d_levels_view,
    collection_types<int>::view d_active_edges_view,
    collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nActiveEdgesA, unsigned int& nActiveEdgesB,
    unsigned int& nCCABlockCounter, const int iter,
    const unsigned int nEdges, const unsigned int max_num_neighbours,
    const int minLevel) {

    const collection_types<int>::const_device d_output_graph(
        d_output_graph_view);
    collection_types<char>::device d_levels(d_levels_view);
    collection_types<int>::device d_active_edges(d_active_edges_view);
    collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);
    const unsigned int edge_size = 2u + 1u + max_num_neighbours;

    const int toggle = iter % 2;
    const int levelLoad = toggle * static_cast<int>(nEdges);
    const int levelStore = (1 - toggle) * static_cast<int>(nEdges);

    unsigned int& nActiveLoad = (toggle == 0) ? nActiveEdgesA : nActiveEdgesB;
    unsigned int& nActiveStore = (toggle == 0) ? nActiveEdgesB : nActiveEdgesA;

    const unsigned int nEdgesLeft = nActiveLoad;

    for (unsigned int globalIdx = threadIndex + blockIndex * blockSize;
         globalIdx < nEdgesLeft; globalIdx += gridSize) {

        const int edgeIdx = (iter == 0)
                                ? static_cast<int>(globalIdx)
                                : d_active_edges[globalIdx];
        const unsigned int edge_pos =
            edge_size * static_cast<unsigned int>(edgeIdx);
        const int nNei =
            d_output_graph[edge_pos + traccc::device::gbts_consts::nNei];

        char next_level = d_levels[static_cast<unsigned int>(
            levelLoad + edgeIdx)];

        bool localChange = false;
        for (int nIdx = 0; nIdx < nNei; nIdx++) {
            const int nextEdgeIdx =
                d_output_graph[edge_pos +
                               traccc::device::gbts_consts::nei_start +
                               static_cast<unsigned int>(nIdx)];
            const char forward_level =
                d_levels[static_cast<unsigned int>(levelLoad + nextEdgeIdx)];
            if (next_level == forward_level) {
                next_level = static_cast<char>(forward_level + 1);
                localChange = true;
                break;
            }
        }
        if (localChange) {
            if (iter == traccc::device::gbts_consts::max_cca_iter - 1) {
                d_outgoing_paths[static_cast<unsigned int>(edgeIdx)].y = -1;
            } else {
                const unsigned int edgesLeftPlace =
                    vecmem::device_atomic_ref<unsigned int>(nActiveStore)
                        .fetch_add(1u);
                d_active_edges[edgesLeftPlace] = edgeIdx;
            }
        } else {
            short out_paths = 0;
            for (int nIdx = 0; nIdx < nNei; ++nIdx) {
                const int nextEdgeIdx =
                    d_output_graph[edge_pos +
                                   traccc::device::gbts_consts::nei_start +
                                   static_cast<unsigned int>(nIdx)];
                if (next_level ==
                    1 + d_levels[static_cast<unsigned int>(nextEdgeIdx)]) {
                    out_paths = static_cast<short>(
                        out_paths + 1 +
                        d_outgoing_paths[static_cast<unsigned int>(
                                             nextEdgeIdx)]
                            .x);
                }
                d_outgoing_paths[static_cast<unsigned int>(nextEdgeIdx)].y =
                    -1;
            }
            d_outgoing_paths[static_cast<unsigned int>(edgeIdx)] =
                make_short2(out_paths,
                            static_cast<short>((next_level >= minLevel) - 1));
        }
        d_levels[static_cast<unsigned int>(levelStore + edgeIdx)] = next_level;
    }
    barrier.blockBarrier();

    if (threadIndex == 0) {
        if (vecmem::device_atomic_ref<unsigned int>(nCCABlockCounter)
                .fetch_add(1u) == (gridSize / blockSize) - 1u) {
            nActiveLoad = 0;
            nCCABlockCounter = 0;
        }
    }
}

}  // namespace traccc::device
