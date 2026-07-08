/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"
#include "traccc/utils/logging.hpp"

// Project include(s).
#include "traccc/gbts_seeding/device/gbts_add_terminus_to_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_bid_seeds_for_hits.hpp"
#include "traccc/gbts_seeding/device/gbts_bin_spacepoints.hpp"
#include "traccc/gbts_seeding/device/gbts_compress_graph.hpp"
#include "traccc/gbts_seeding/device/gbts_convert_seeds.hpp"
#include "traccc/gbts_seeding/device/gbts_count_eta_phi_bins.hpp"
#include "traccc/gbts_seeding/device/gbts_count_spacepoints_by_layer.hpp"
#include "traccc/gbts_seeding/device/gbts_count_terminus_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_fill_path_store.hpp"
#include "traccc/gbts_seeding/device/gbts_find_minmax_radius.hpp"
#include "traccc/gbts_seeding/device/gbts_fit_segments.hpp"
#include "traccc/gbts_seeding/device/gbts_link_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_make_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_match_graph_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_prefix_sum_eta_phi_bins.hpp"
#include "traccc/gbts_seeding/device/gbts_rebid_seeds_for_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reindex_edges.hpp"
#include "traccc/gbts_seeding/device/gbts_reset_edge_bids.hpp"
#include "traccc/gbts_seeding/device/gbts_run_cca_iteration.hpp"
#include "traccc/gbts_seeding/device/gbts_sort_nodes.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <algorithm>
#include <cstdint>
#include <memory_resource>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace traccc::cuda {

namespace kernels {

using float4 = traccc::float4;
using uint2 = traccc::uint2;
using int2 = traccc::int2;

// ---------------------------------------------------------------------------
// Stage 1 — nodes-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_count_spacepoints_by_layer
__global__ void gbts_count_spacepoints_by_layer(
    const device::gbts_count_spacepoints_by_layer_payload payload) {

    device::gbts_count_spacepoints_by_layer(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bin_spacepoints
__global__ void gbts_bin_spacepoints(
    const device::gbts_bin_spacepoints_payload payload) {

    device::gbts_bin_spacepoints(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_count_eta_phi_bins
__global__ void gbts_count_eta_phi_bins(
    const device::gbts_count_eta_phi_bins_payload payload) {

    device::gbts_count_eta_phi_bins(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_fill_sort_keys
__global__ void gbts_fill_sort_keys(
    const device::gbts_sort_nodes_payload payload,
    vecmem::data::vector_view<std::uint64_t> keys_view,
    vecmem::data::vector_view<unsigned int> perm_view) {

    device::gbts_fill_sort_keys(details::thread_id1{}, payload, keys_view,
                                perm_view);
}

/// CUDA kernel for running @c traccc::device::gbts_gather_sorted_nodes
__global__ void gbts_gather_sorted_nodes(
    const device::gbts_sort_nodes_payload payload,
    vecmem::data::vector_view<unsigned int> perm_view) {

    device::gbts_gather_sorted_nodes(details::thread_id1{}, payload, perm_view);
}

/// CUDA kernel for running @c traccc::device::gbts_find_minmax_radius
__global__ void gbts_find_minmax_radius(
    const device::gbts_find_minmax_radius_payload payload) {

    device::gbts_find_minmax_radius(details::thread_id1{}, payload);
}

// ---------------------------------------------------------------------------
// Stage 2 — graph-making kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_make_graph_edges
__global__ void gbts_make_graph_edges(
    const device::gbts_make_graph_edges_payload payload) {

    __shared__ float phi[traccc::device::gbts_consts::node_buffer_length];
    __shared__ float4
        node_pack[traccc::device::gbts_consts::node_buffer_length];
    const traccc::cuda::barrier barrier;

    device::gbts_make_graph_edges(
        details::thread_id1{}, barrier, payload,
        {vecmem::data::vector_view<float>(
             traccc::device::gbts_consts::node_buffer_length, phi),
         vecmem::data::vector_view<float4>(
             traccc::device::gbts_consts::node_buffer_length, node_pack)});
}

/// CUDA kernel for running @c traccc::device::gbts_link_graph_edges
__global__ void gbts_link_graph_edges(
    const device::gbts_link_graph_edges_payload payload) {

    device::gbts_link_graph_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_match_graph_edges
__global__ void gbts_match_graph_edges(
    const device::gbts_match_graph_edges_payload payload) {

    device::gbts_match_graph_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_reindex_edges
__global__ void gbts_reindex_edges(
    const device::gbts_reindex_edges_payload payload) {

    device::gbts_reindex_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_compress_graph
__global__ void gbts_compress_graph(
    const device::gbts_compress_graph_payload payload) {

    device::gbts_compress_graph(details::thread_id1{}, payload);
}

// ---------------------------------------------------------------------------
// Stage 3 — graph-processing kernels
// ---------------------------------------------------------------------------

/// CUDA kernel for running @c traccc::device::gbts_run_cca_iteration
__global__ void gbts_run_cca_iteration(
    const device::gbts_run_cca_iteration_payload payload) {

    device::gbts_run_cca_iteration(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_count_terminus_edges
__global__ void gbts_count_terminus_edges(
    const device::gbts_count_terminus_edges_payload payload) {

    device::gbts_count_terminus_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_add_terminus_to_path_store
__global__ void gbts_add_terminus_to_path_store(
    const device::gbts_add_terminus_to_path_store_payload payload) {

    device::gbts_add_terminus_to_path_store(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_fill_path_store
__global__ void gbts_fill_path_store(
    const device::gbts_fill_path_store_payload payload) {

    __shared__ traccc::uint2
        live_paths[traccc::device::gbts_consts::live_path_buffer];
    __shared__ int n_live_paths;
    const traccc::cuda::barrier barrier;

    device::gbts_fill_path_store(
        details::thread_id1{}, barrier, payload,
        {vecmem::data::vector_view<traccc::uint2>(
             traccc::device::gbts_consts::live_path_buffer, live_paths),
         n_live_paths});
}

/// CUDA kernel for running @c traccc::device::gbts_fit_segments
__global__ void gbts_fit_segments(
    const device::gbts_fit_segments_payload payload) {

    device::gbts_fit_segments(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bid_canonical_proposals
__global__ void gbts_bid_canonical_proposals(
    const device::gbts_fit_segments_payload payload) {

    device::gbts_bid_canonical_proposals(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_mark_canonical_losers
__global__ void gbts_mark_canonical_losers(
    const device::gbts_fit_segments_payload payload) {

    device::gbts_mark_canonical_losers(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_reset_edge_bids
__global__ void gbts_reset_edge_bids(
    const device::gbts_reset_edge_bids_payload payload) {

    device::gbts_reset_edge_bids(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_rebid_promote_seeds
__global__ void gbts_rebid_promote_seeds(
    const device::gbts_rebid_seeds_for_edges_payload payload) {

    device::gbts_rebid_promote_seeds(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_rebid_seeds_for_edges
__global__ void gbts_rebid_seeds_for_edges(
    const device::gbts_rebid_seeds_for_edges_payload payload) {

    device::gbts_rebid_seeds_for_edges(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_mark_rebid_losers
__global__ void gbts_mark_rebid_losers(
    const device::gbts_rebid_seeds_for_edges_payload payload) {

    device::gbts_mark_rebid_losers(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_bid_seeds_for_hits
__global__ void gbts_bid_seeds_for_hits(
    const device::gbts_bid_seeds_for_hits_payload payload) {

    device::gbts_bid_seeds_for_hits(details::thread_id1{}, payload);
}

/// CUDA kernel for running @c traccc::device::gbts_convert_seeds
__global__ void gbts_convert_seeds(
    const device::gbts_convert_seeds_payload payload) {

    device::gbts_convert_seeds(details::thread_id1{}, payload);
}

}  // namespace kernels

// ===========================================================================
// gbts_seeding_algorithm: kernel launchers
// ===========================================================================

gbts_seeding_algorithm::gbts_seeding_algorithm(
    const gbts_seedfinder_config& cfg, const memory_resource& mr,
    const vecmem::copy& copy, const stream_wrapper& str,
    std::unique_ptr<const Logger> logger)
    : device::gbts_seeding_algorithm(cfg, mr, copy, std::move(logger)),
      cuda::algorithm_base{str} {}

void gbts_seeding_algorithm::gbts_count_spacepoints_by_layer_kernel(
    const device::gbts_count_spacepoints_by_layer_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::gbts_count_spacepoints_by_layer<<<n_blocks, n_threads, 0,
                                               details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
    vecmem::device_vector<unsigned int> d_layer_sums(payload.layerCounts);
    thrust::inclusive_scan(
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(details::get_stream(stream())),
        d_layer_sums.begin(), d_layer_sums.end(), d_layer_sums.begin());
}

void gbts_seeding_algorithm::gbts_bin_spacepoints_kernel(
    const device::gbts_bin_spacepoints_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nSp - 1) / n_threads;
    kernels::gbts_bin_spacepoints<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_count_eta_phi_bins_kernel(
    const device::gbts_count_eta_phi_bins_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::gbts_count_eta_phi_bins<<<n_blocks, n_threads, 0,
                                       details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
    vecmem::device_vector<unsigned int> d_eta_sums(payload.eta_node_counter);
    thrust::inclusive_scan(
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(details::get_stream(stream())),
        d_eta_sums.begin(), d_eta_sums.end(), d_eta_sums.begin());
}

void gbts_seeding_algorithm::gbts_prefix_sum_eta_phi_bins_kernel(
    const device::gbts_prefix_sum_eta_phi_bins_payload&) const {

    // No-op on CUDA: phi_cusums is consumed only by the atomic
    // device::gbts_sort_nodes path, which CUDA does not launch (its sort uses
    // gbts_fill_sort_keys + thrust::sort_by_key + gbts_gather_sorted_nodes, none
    // of which read phi_cusums). The shared orchestrator still calls this so the
    // other backends -- which do read phi_cusums via
    // device::gbts_prefix_sum_eta_phi_bins -- stay correct.
}

void gbts_seeding_algorithm::gbts_sort_nodes_kernel(
    const device::gbts_sort_nodes_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nNodes - 1) / n_threads;

    // Deterministic replacement for the old atomic scatter: build the per-node
    // histo_bin key + identity permutation, sort the permutation by
    // (histo_bin, spacepoint position), then gather the geometry tuples into the
    // sorted slots. The within-bin tie-break is the spacepoint (x, y, z) -- a
    // stable per-spacepoint identity -- so the node order (and hence
    // make_graph_edges) is reproducible regardless of the race-ordered
    // spacepoint collection / layer scatter upstream. The 24-byte geometry tuple
    // is recomputed in the gather; only the 4-byte permutation index moves.
    //
    // keys_buf / perm_buf are stream-ordered scratch: every kernel and the sort
    // are enqueued on the single algorithm stream, so although the buffers are
    // destroyed when this method returns (par_nosync), any later reuse of their
    // pages is stream-ordered after the gather -> no race, no extra sync.
    vecmem::data::vector_buffer<std::uint64_t> keys_buf(payload.nNodes,
                                                        mr().main);
    copy().setup(keys_buf)->ignore();
    vecmem::data::vector_buffer<unsigned int> perm_buf(payload.nNodes,
                                                       mr().main);
    copy().setup(perm_buf)->ignore();

    kernels::gbts_fill_sort_keys<<<n_blocks, n_threads, 0,
                                   details::get_stream(stream())>>>(
        payload, keys_buf, perm_buf);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //

    thrust::sort_by_key(
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(details::get_stream(stream())),
        keys_buf.ptr(), keys_buf.ptr() + payload.nNodes, perm_buf.ptr());

    kernels::gbts_gather_sorted_nodes<<<n_blocks, n_threads, 0,
                                        details::get_stream(stream())>>>(
        payload, perm_buf);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_find_minmax_radius_kernel(
    const device::gbts_find_minmax_radius_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nEtaBins - 1) / n_threads;
    kernels::gbts_find_minmax_radius<<<n_blocks, n_threads, 0,
                                       details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_make_graph_edges_kernel(
    const device::gbts_make_graph_edges_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = payload.nUsedBinPairs;
    kernels::gbts_make_graph_edges<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
    vecmem::device_vector<unsigned int> d_num_outgoing_edges(
        payload.num_outgoing_edges);
    thrust::inclusive_scan(
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(details::get_stream(stream())),
        d_num_outgoing_edges.begin(), d_num_outgoing_edges.end(),
        d_num_outgoing_edges.begin());
}

void gbts_seeding_algorithm::gbts_link_graph_edges_kernel(
    const device::gbts_link_graph_edges_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::gbts_link_graph_edges<<<n_blocks, n_threads, 0,
                                     details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_match_graph_edges_kernel(
    const device::gbts_match_graph_edges_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::gbts_match_graph_edges<<<n_blocks, n_threads, 0,
                                      details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_reindex_edges_kernel(
    const device::gbts_reindex_edges_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::gbts_reindex_edges<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_compress_graph_kernel(
    const device::gbts_compress_graph_payload& payload) const {

    const unsigned int n_threads = 256;
    const unsigned int n_blocks = 1 + (payload.nEdges - 1) / n_threads;
    kernels::gbts_compress_graph<<<n_blocks, n_threads, 0,
                                   details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_run_cca_iteration_kernel(
    const device::gbts_run_cca_iteration_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;

    kernels::gbts_run_cca_iteration<<<n_blocks, n_threads, 0,
                                      details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_count_terminus_edges_kernel(
    const device::gbts_count_terminus_edges_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nConnectedEdges - 1) / n_threads;
    kernels::gbts_count_terminus_edges<<<n_blocks, n_threads, 0,
                                         details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_add_terminus_to_path_store_kernel(
    const device::gbts_add_terminus_to_path_store_payload& payload) const {

    const unsigned int n_threads = 128;
    // The kernel also initialises the nPaths-sized prop_hash buffer.
    const unsigned int n_items = (payload.nConnectedEdges > payload.nPaths)
                                     ? payload.nConnectedEdges
                                     : payload.nPaths;
    const unsigned int n_blocks = 1 + (n_items - 1) / n_threads;
    kernels::gbts_add_terminus_to_path_store<<<n_blocks, n_threads, 0,
                                               details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_fill_path_store_kernel(
    const device::gbts_fill_path_store_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int pathsPerTerminus =
        1 + (payload.nPaths - 1) / payload.nTerminusEdges;
    const unsigned int terminusPerBlock = std::min(
        n_threads, 1 + (traccc::device::gbts_consts::live_path_buffer - 1) /
                           pathsPerTerminus);
    const unsigned int n_blocks =
        1 + (payload.nTerminusEdges - 1) / terminusPerBlock;
    kernels::gbts_fill_path_store<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_fit_segments_kernel(
    const device::gbts_fit_segments_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nPaths - 1) / n_threads;

    // 1. Fit + create proposals at (race) slots, each tagged with a
    //    deterministic hash of its path's node indices. No bid yet.
    kernels::gbts_fit_segments<<<n_blocks, n_threads, 0,
                                 details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //

    // 2. Sort proposals by that hash so the proposal index (the bid tie-break)
    //    is canonical/reproducible. Unused slots carry a 0xFF.. sentinel hash
    //    and sort to the back. edge_bids was already zeroed by the orchestrator,
    //    and seed_ambiguity is (re)set per proposal inside the bid, so no extra
    //    reset is needed.
    const auto policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(details::get_stream(stream()));
    const unsigned int n = payload.nPaths;

    // Sort the proposals directly by their deterministic key. sort_by_key carries
    // the int2 proposals under the (cub stable-radix) key permutation, so prop_hash
    // ends sorted (0xFF.. sentinels at the back) and seed_proposals is reordered
    // identically -- exactly what gbts_bid_canonical_proposals consumes (prop_hash[c]
    // sentinel test + aligned seed_proposals[c]). One op, no scratch; bit-identical
    // to the old sequence + sort_by_key(perm) + gather + copy.
    thrust::sort_by_key(policy, payload.prop_hash.ptr(),
                        payload.prop_hash.ptr() + n,
                        payload.seed_proposals.ptr());

    // 3. Depth-1 bid on the canonical proposal order (bid-only; no marking).
    kernels::gbts_bid_canonical_proposals<<<n_blocks, n_threads, 0,
                                            details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //

    // 4. With the bids settled, each proposal derives its own won/lost state
    //    from the final per-edge maxima (own-slot writes only) -- the outcome
    //    is a pure function of the bid set, independent of atomic ordering.
    kernels::gbts_mark_canonical_losers<<<n_blocks, n_threads, 0,
                                          details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());  //
}

void gbts_seeding_algorithm::gbts_reset_edge_bids_kernel(
    const device::gbts_reset_edge_bids_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::gbts_reset_edge_bids<<<n_blocks, n_threads, 0,
                                    details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_rebid_seeds_for_edges_kernel(
    const device::gbts_rebid_seeds_for_edges_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    // Pass A: promote/reject and initialise the bidders' ambiguity to 0 (each
    // thread writes only its own slot). The kernel boundary guarantees this
    // completes before any loser-marking in the bid pass.
    kernels::gbts_rebid_promote_seeds<<<n_blocks, n_threads, 0,
                                        details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    // Pass B1: the flagged proposals place their bids (fetch_max only, no
    // marking).
    kernels::gbts_rebid_seeds_for_edges<<<n_blocks, n_threads, 0,
                                          details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    // Pass B2: with the bids settled, each active proposal derives its own
    // won/lost state from the final per-edge maxima (own-slot writes only), so
    // the round's outcome is independent of atomic ordering.
    kernels::gbts_mark_rebid_losers<<<n_blocks, n_threads, 0,
                                      details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_bid_seeds_for_hits_kernel(
    const device::gbts_bid_seeds_for_hits_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::gbts_bid_seeds_for_hits<<<n_blocks, n_threads, 0,
                                       details::get_stream(stream())>>>(
        payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void gbts_seeding_algorithm::gbts_convert_seeds_kernel(
    const device::gbts_convert_seeds_payload& payload) const {

    const unsigned int n_threads = 128;
    const unsigned int n_blocks = 1 + (payload.nProps - 1) / n_threads;
    kernels::gbts_convert_seeds<<<n_blocks, n_threads, 0,
                                  details::get_stream(stream())>>>(payload);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
