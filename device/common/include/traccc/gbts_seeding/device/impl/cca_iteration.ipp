/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void cca_iteration(
    const global_index_t globalIndex,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<unsigned char>::view d_levels_view,
    const collection_types<char>::view d_active_edges_view,
    const collection_types<short2>::view d_outgoing_paths_view,
    const unsigned char iter, const unsigned int nConnectedEdges,
    const unsigned int max_num_neighbours, const unsigned char minLevel) {

    // Hoist raw base pointers out of the vecmem device vectors (the
    // device_vector::data() idiom, as in clusterization/ccl_kernel). The
    // scattered terminus loop below is memory-latency bound, and indexing plain
    // pointers — as the gbts_changes reference does — lets the compiler overlap
    // the independent neighbour loads (more memory-level parallelism) instead
    // of serialising them through the device_vector object.
    const unsigned int* d_output_graph =
        collection_types<unsigned int>::const_device(d_output_graph_view)
            .data();
    unsigned char* d_levels =
        collection_types<unsigned char>::device(d_levels_view).data();
    char* d_active_edges =
        collection_types<char>::device(d_active_edges_view).data();
    short2* d_outgoing_paths =
        collection_types<short2>::device(d_outgoing_paths_view).data();

    const unsigned int edge_size = 2 + 1 + max_num_neighbours;

    const unsigned int toggle = iter % 2;
    const unsigned int levelLoad = toggle * nConnectedEdges;
    const unsigned int levelStore = (1 - toggle) * nConnectedEdges;

    // One edge per call; the grid-stride loop over edges lives in the CUDA
    // kernel wrapper (which is what lets the compiler schedule the scattered
    // neighbour loads with high memory-level parallelism).
    if (iter != 0) {
        if (d_active_edges[globalIndex] != iter) {
            return;
        }
    }

    const unsigned int edge_pos = edge_size * globalIndex;
    const unsigned int nNei = d_output_graph[edge_pos + gbts_consts::nNei];

    unsigned char next_level = d_levels[levelLoad + globalIndex];

    bool localChange = false;
    for (unsigned int nIdx = 0; nIdx < nNei; nIdx++) {
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
            d_outgoing_paths[globalIndex].y = -1;
            d_active_edges[globalIndex] = -1;
        } else {
            d_active_edges[globalIndex] = static_cast<char>(iter + 1u);
        }
    } else {
        d_active_edges[globalIndex] = -1;
        short out_paths = 0;
        for (unsigned int nIdx = 0; nIdx < nNei; ++nIdx) {
            const unsigned int nextglobalIndex =
                d_output_graph[edge_pos + gbts_consts::nei_start + nIdx];
            if (next_level == 1 + d_levels[nextglobalIndex]) {
                // calculate the #d_state_store nodes for segment extraction
                // starting at this edge
                out_paths = static_cast<short>(
                    out_paths + 1 + d_outgoing_paths[nextglobalIndex].x);
            }
            // flag as not terminus edge
            d_outgoing_paths[nextglobalIndex].y = -1;
        }
        // flag as long enough segement to become a seed
        d_outgoing_paths[globalIndex] =
            make_short2(out_paths, (next_level >= minLevel) - 1);
    }
    // store new level and ensure all final
    // levels are on both sides of the array
    d_levels[levelStore + globalIndex] = next_level;
}

}  // namespace traccc::device
