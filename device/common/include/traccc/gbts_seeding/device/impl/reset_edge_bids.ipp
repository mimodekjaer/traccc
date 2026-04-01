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


// TODO not started  

namespace traccc::device {

    TRACCC_HOST_DEVICE
    inline void reset_edge_bids_kernel(
        const global_index_t global_index,
        const collection_types<int2>::const_view d_path_store_view, 
        const collection_types<int2>::view d_seed_proposals_view,
        const collection_types<unsigned long long int>::view d_edge_bids_view,
        const collection_types<char>::view d_seed_ambiguity_view,
        const collection_types<unsigned int>::view d_counters_view,
        const int round) {


collection_types<unsigned int>::device d_counters(d_counters_view);
unsigned int nProps = d_counters[8];
if (global_index >= nProps) {
    return;
}


const collection_types<int2>::const_device d_path_store(d_path_store_view);
collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
const collection_types<unsigned long long int>::const_device d_edge_bids(d_edge_bids_view);
collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);

// first round find best seed starting at each edge
char ambi = d_seed_ambiguity[global_index];
if (round == -1) {
if (ambi == 0) {
// rebid 'best seed from edge' in later rounds
d_seed_ambiguity[global_index] = 1;
return;
} else {
d_seed_ambiguity[global_index] = -2;
// count rejected props to calculate nSeeds
vecmem::device_atomic_ref<unsigned int> d_counters_9_ref(d_counters[9]);
d_counters_9_ref.fetch_add(1);
return;
}
} else if (ambi == -2 | ambi == 0) {
// only reset maybes
return;
}
int2 prop = d_seed_proposals[global_index];

bool isgood = true;

// dummy path to start the loop
int2 path = make_int2(0, prop.y);
while (path.y >= 0) {
path = d_path_store[static_cast<unsigned int>(path.y)];

unsigned long long int best_bid = d_edge_bids[static_cast<unsigned int>(path.x)];
if (d_seed_ambiguity[static_cast<unsigned int>(best_bid & 0xFFFFFFFFLL)] == 0) {
isgood = false;
break;
}
}
if (isgood) {
d_seed_ambiguity[global_index] = 1;
}  // flag as maybe seed, shares with a loser
else {
d_seed_ambiguity[global_index] = -2;
vecmem::device_atomic_ref<unsigned int> d_counters_9_ref(d_counters[9]);
d_counters_9_ref.fetch_add(1);
// definate fake, shares with a winner
}







}


} // namespace traccc::device