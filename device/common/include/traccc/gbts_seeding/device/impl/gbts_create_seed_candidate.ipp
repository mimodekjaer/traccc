/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device::details {

TRACCC_HOST_DEVICE
inline void gbts_place_seed_bid(
    const int qual, const int path_idx, const unsigned int prop_idx,
    const vecmem::data::vector_view<unsigned long long int> d_edge_bids_view,
    const vecmem::data::vector_view<const int2>& d_path_store_view,
    char depth) {

    vecmem::device_vector<unsigned long long int> d_edge_bids(d_edge_bids_view);
    const vecmem::device_vector<const int2> d_path_store(d_path_store_view);

    const unsigned long long int seed_bid =
        (static_cast<unsigned long long int>(qual) << 32) |
        (static_cast<unsigned long long int>(prop_idx));

    int2 path = int2{0, path_idx};
    while (path.y >= 0 && depth != 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        depth--;

        vecmem::device_atomic_ref<unsigned long long int> atomic_bid(
            d_edge_bids[static_cast<unsigned int>(path.x)]);
        atomic_bid.fetch_max(seed_bid);
    }
}

TRACCC_HOST_DEVICE
inline bool gbts_seed_bid_won(
    const int qual, const int path_idx, const unsigned int prop_idx,
    const vecmem::data::vector_view<unsigned long long int> d_edge_bids_view,
    const vecmem::data::vector_view<const int2>& d_path_store_view,
    char depth) {

    const vecmem::device_vector<const unsigned long long int> d_edge_bids(
        d_edge_bids_view);
    const vecmem::device_vector<const int2> d_path_store(d_path_store_view);

    const unsigned long long int seed_bid =
        (static_cast<unsigned long long int>(qual) << 32) |
        (static_cast<unsigned long long int>(prop_idx));

    int2 path = int2{0, path_idx};
    while (path.y >= 0 && depth != 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        depth--;

        if (d_edge_bids[static_cast<unsigned int>(path.x)] != seed_bid) {
            return false;
        }
    }
    return true;
}

}  // namespace traccc::device::details
