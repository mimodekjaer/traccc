/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::device {

template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_run_cca_iteration(
    const thread_id_t& thread_id,
    const gbts_run_cca_iteration_payload& payload) {

    const vecmem::device_vector<const unsigned int> d_output_graph(
        payload.output_graph);
    vecmem::device_vector<unsigned char> d_levels(payload.levels);
    vecmem::device_vector<char> d_active_edges(payload.active_edges);
    vecmem::device_vector<short2> d_outgoing_paths(payload.outgoing_paths);

    const unsigned char iter = payload.iter;

    const unsigned int edge_size = 2 + 1 + payload.max_num_neighbours;

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    if (payload.count_pass) {
        // Post-CCA path-count / terminus-flag sweep. The levels are final, so
        // every value read here settled at a kernel boundary. Pass k processes
        // exactly the edges at level k+1: their level-k children's counts were
        // written in pass k-1. The only cross-thread writes store the single
        // value -1 into a neighbour's .y (idempotent 2-byte stores; the
        // owner's .x is a separate 2-byte store, so nothing tears) -- the
        // result is a pure function of the graph, independent of thread
        // ordering.
        for (unsigned int globalIndex = globalIdx;
             globalIndex < payload.nConnectedEdges;
             globalIndex += blockDimX * gridDimX) {

            const unsigned char level = d_levels[globalIndex];
            if (level != static_cast<unsigned char>(iter + 1)) {
                continue;
            }

            const unsigned int edge_pos = edge_size * globalIndex;
            const unsigned int nNeighbours =
                d_output_graph[edge_pos + gbts_consts::nNei];

            short out_paths = 0;
            for (unsigned int nIdx = 0; nIdx < nNeighbours; nIdx++) {
                const unsigned int nextglobalIndex =
                    d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
                if (level == 1 + d_levels[nextglobalIndex]) {
                    // a path continues from the neighbour into this edge
                    out_paths = static_cast<short>(
                        out_paths + 1 + d_outgoing_paths[nextglobalIndex].x);
                }
                // every neighbour has this edge as a predecessor, so it can
                // never start a path of its own
                d_outgoing_paths[nextglobalIndex].y = -1;
            }
            d_outgoing_paths[globalIndex].x = out_paths;
            if (level < payload.minLevel) {
                // too short to seed a path of its own
                d_outgoing_paths[globalIndex].y = -1;
            }
        }
        return;
    }

    const unsigned int toggle = iter % 2;
    const unsigned int levelLoad = toggle * payload.nConnectedEdges;
    const unsigned int levelStore = (1 - toggle) * payload.nConnectedEdges;

    for (unsigned int globalIndex = globalIdx;
         globalIndex < payload.nConnectedEdges;
         globalIndex += blockDimX * gridDimX) {

        if (iter != 0) {
            if (d_active_edges[globalIndex] != iter) {
                continue;
            }
        }

        const unsigned int edge_pos = edge_size * globalIndex;
        const unsigned int nNeighbours =
            d_output_graph[edge_pos + gbts_consts::nNei];

        unsigned char next_level = d_levels[levelLoad + globalIndex];

        bool localChange = false;
        for (unsigned int nIdx = 0; nIdx < nNeighbours; nIdx++) {
            const unsigned int nextglobalIndex =
                d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
            const unsigned char forward_level =
                d_levels[levelLoad + nextglobalIndex];
            if (next_level == forward_level) {
                next_level = forward_level + 1;
                localChange = true;
                break;
            }
        }
        if (localChange) {
            if (iter == traccc::device::gbts_consts::max_cca_iter - 1) {
                // hit the iteration cap while still growing: never a terminus
                // (own-slot write, race-free)
                d_outgoing_paths[globalIndex].y = -1;
                d_active_edges[globalIndex] = -1;
            } else {
                d_active_edges[globalIndex] = static_cast<char>(iter + 1u);
            }
        } else {
            // settled -- the path counting / terminus flagging happens in the
            // count passes once ALL levels are final (the old in-place version
            // raced against same-iteration neighbour settles: its full short2
            // store could be lost to a concurrent neighbour .y = -1 mark, and
            // it read neighbour counts that were not yet written)
            d_active_edges[globalIndex] = -1;
        }
        // store new level
        d_levels[levelStore + globalIndex] = next_level;
    }
}

}  // namespace traccc::device
