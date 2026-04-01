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
inline void fill_path_store_kernel(
       const global_index_t global_index,
       const collection_types<int2>::view d_path_store_view,
       const collection_types<int>::const_view d_output_graph_view,
       const collection_types<char>::const_view d_levels_view,
       const collection_types<unsigned int>::view d_counters_view,
       const unsigned int nTerminus,
       const unsigned int nTerminusPerBlock,
       const unsigned int max_num_neighbours,
       const unsigned int nPaths) {

collection_types<int2>::device d_path_store(d_path_store_view);
const collection_types<int>::const_device d_output_graph(d_output_graph_view);
const collection_types<char>::const_device d_levels(d_levels_view);
collection_types<unsigned int>::device d_counters(d_counters_view);

       __shared__ int2 live_paths[traccc::device::gbts_consts::live_path_buffer];
       __shared__ int n_live_paths;

       if (threadIdx.x == 0) {
              n_live_paths = 0;
       }
       barrier().blockBarrier();

       int edge_size = static_cast<int>(2 + 1 + max_num_neighbours);
       unsigned int path_idx = threadIdx.x + blockIdx.x * nTerminusPerBlock;
// populate live_paths with terminus to start exploration from
       if (threadIdx.x < nTerminusPerBlock && path_idx < nTerminus) {
              int2 path = d_path_store[path_idx];
              int nNei = d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::nNei +
                     edge_size * path.x)];
              char level = d_levels[static_cast<unsigned int>(path.x)];
              for (int nei = 0; nei < nNei; ++nei) {
                     int edge_idx =
                     d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::nei_start + nei +
                     edge_size * path.x)];
                     // only search down longest path
                     if (level != d_levels[static_cast<unsigned int>(edge_idx)] + 1) {
                            continue;
                     }
                     int live_idx = vecmem::device_atomic_ref<int>(n_live_paths).fetch_add(1);
                     if (live_idx >= traccc::device::gbts_consts::live_path_buffer) {
                            break;
                     }
                     int new_path_idx = static_cast<int>(vecmem::device_atomic_ref<unsigned int>(d_counters[7]).fetch_add(1));
                     // head edge idx, link back
                     d_path_store[static_cast<unsigned int>(new_path_idx)] = make_int2(edge_idx, static_cast<int>(path_idx));
                     live_paths[live_idx] = make_int2(edge_idx, new_path_idx);
              }
       }
       barrier().blockBarrier();

       int2 path = make_int2(0, 0);
       bool has_path = false;

       while (n_live_paths > 0) {
              has_path = false;
              if (threadIdx.x == 0) {
                     n_live_paths = min(n_live_paths,
                            traccc::device::gbts_consts::live_path_buffer);
              }
              barrier().blockBarrier();
              // get path
              if (threadIdx.x < n_live_paths) {
                     path = live_paths[n_live_paths - threadIdx.x - 1];
                     has_path = true;
              }
              barrier().blockBarrier();
              if (threadIdx.x == 0) {
              n_live_paths =
                     n_live_paths < static_cast<int>(blockDim.x) ? 0 : n_live_paths - static_cast<int>(blockDim.x);
              }
              barrier().blockBarrier();
              if (has_path) {
                     int nNei = d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::nNei +
                            edge_size * path.x)];
                     char level = d_levels[static_cast<unsigned int>(path.x)];
                     for (int nei = 0; nei < nNei; ++nei) {
                            int edge_idx =
                            d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::nei_start +
                                   nei + edge_size * path.x)];
                            // only search down longest segments
                            if (level != d_levels[static_cast<unsigned int>(edge_idx)] + 1) {
                                   continue;
                            }
                            path_idx = vecmem::device_atomic_ref<unsigned int>(d_counters[7]).fetch_add(1);
                            if (path_idx >= nPaths) {
                                   break;
                            }
                            int live_idx = vecmem::device_atomic_ref<int>(n_live_paths).fetch_add(1);
                            if (live_idx >= traccc::device::gbts_consts::live_path_buffer) {
                                   break;
                            }
                            // head edge idx, link back
                            d_path_store[path_idx] = make_int2(edge_idx, static_cast<int>(path.y));
                            live_paths[live_idx] = make_int2(edge_idx, static_cast<int>(path_idx));
                     }
              }
       // wait for live_paths to repopulate
       barrier().blockBarrier();
       }
}



} // namespace traccc::device