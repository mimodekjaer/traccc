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
    const collection_types<unsigned char>::view d_active_edges_view,
    const collection_types<short2>::view d_outgoing_paths_view,
    const unsigned char iter, const unsigned int nEdges,
    const unsigned int max_num_neighbours, const unsigned char minLevel) {

    const collection_types<unsigned int>::const_device d_output_graph(
        d_output_graph_view);
    collection_types<unsigned char>::device d_levels(d_levels_view);
    collection_types<unsigned char>::device d_active_edges(d_active_edges_view);
    collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);
    // Sentinel value for "settled / no longer active". The active_edges array
    // is unsigned char; iter is in [0, max_cca_iter) which is <= 15, so 0xFF
    // never collides with a real iteration index.
    constexpr unsigned char k_active_done = 0xFFu;
    // max_num_neighbours is accepted for API stability; the SoA layout makes
    // it unused inside the loop (each column owns a contiguous nEdges-long
    // array, indexed via gbts_og_index below).
    (void)max_num_neighbours;

    const unsigned int toggle = iter % 2u;
    const unsigned int levelLoad = toggle * nEdges;
    const unsigned int levelStore = (1u - toggle) * nEdges;

    if (globalIndex >= nEdges) {
        return;
    }

    if (iter != 0) {
        if (d_active_edges[globalIndex] != iter) {
            return;
        }
    }

    const unsigned int nNei = d_output_graph[traccc::device::gbts_og_index(
        traccc::device::gbts_consts::nNei, globalIndex, nEdges)];

    char next_level = d_levels[levelLoad + globalIndex];

    bool localChange = false;
    for (unsigned int nIdx = 0; nIdx < nNei; nIdx++) {
        const unsigned int nextglobalIndex =
            d_output_graph[traccc::device::gbts_og_index(
                traccc::device::gbts_consts::nei_start + nIdx, globalIndex,
                nEdges)];
        const char forward_level = d_levels[levelLoad + nextglobalIndex];
        if (next_level == forward_level) {
            next_level = forward_level + 1;
            localChange = true;
            break;
        }
    }
    if (localChange) {
        if (iter == traccc::device::gbts_consts::max_cca_iter - 1) {
            d_outgoing_paths[globalIndex].y = -1;
            d_active_edges[globalIndex] = k_active_done;
        } else {
            d_active_edges[globalIndex] =
                static_cast<unsigned char>(iter + 1u);
        }
    } else {
        d_active_edges[globalIndex] = k_active_done;
        short out_paths = 0;
        for (unsigned int nIdx = 0; nIdx < nNei; ++nIdx) {
            const unsigned int nextglobalIndex =
                d_output_graph[traccc::device::gbts_og_index(
                    traccc::device::gbts_consts::nei_start + nIdx, globalIndex,
                    nEdges)];
            if (next_level ==
                1 + d_levels[nextglobalIndex]) {
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
