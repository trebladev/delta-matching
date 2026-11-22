/******************************************************************************
 * Route-A Block-Aligned Varlen Dynamic Persistent Tile Scheduler (Hopper SM90)
 *
 * Adapted from FlashAttention-3's VarlenDynamicPersistentTileScheduler with
 * unconditional seqlens_actual lookup so:
 *   - Q-tile count (fwd) = ceil_div(seqlens_actual[b], kBlockM)
 *   - K-tile count (bwd) = ceil_div(seqlens_actual[b], kBlockN)
 *   - seqlens_actual[b] == 0 yields 0 tiles cleanly regardless of max_seqlen.
 ******************************************************************************/
#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include "cute/tensor.hpp"
#include "tile_scheduler.hpp"
#include "named_barrier.hpp"

namespace flash {

template <
    int kBlock,
    int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
    int NumProducerThreads = cutlass::NumThreadsPerWarp,
    bool WarpSpecialized = true,
    bool LPT = false,
    bool IsBwd = false
>
class DMVarlenDynamicPersistentTileScheduler {
    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:
    struct Params {
        int num_head;
        int num_batch;
        int qhead_per_khead;
        int* const tile_count_semaphore;
        int const* const seqlens_actual;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        assert(args.seqused != nullptr);
        return {
            args.num_head,
            args.num_batch,
            args.qhead_per_khead,
            args.tile_count_semaphore,
            args.seqused
        };
    }

    static dim3
    get_grid_shape(Params const& /*params*/, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool is_valid(Params const& params) const {
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        auto get_block_coord(Params const& /*params*/) const {
            if constexpr (IsBwd) {
                return cute::make_tuple(int32_t(block), int32_t(bidh), int32_t(bidb));
            } else {
                return cute::make_tuple(int32_t(block), int32_t(bidh), int32_t(bidb), int32_t(0));
            }
        }
    };

    CUTLASS_DEVICE
    explicit DMVarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler)
        : work_info_smem(smem_scheduler) {}

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int const lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_blocks = [&](int bidb_start) {
            int const batch_idx = lane + bidb_start;
            if (batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1) {
                int const actual = params.seqlens_actual[batch_idx];
                return actual > 0 ? cute::ceil_div(actual, kBlock) : 0;
            }
            return 0;
        };

        int num_blocks = get_num_blocks(current_work.bidb);
        int num_blocks_cum = warp_prefix_sum(num_blocks);
        int blocks_in_group = __shfl_sync(0xffffffff, num_blocks_cum, cutlass::NumThreadsPerWarp - 1);
        int group_end_tile = current_work.tile_idx + blocks_in_group * params.num_head;
        int bidb = current_work.bidb;

        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_blocks = get_num_blocks(bidb);
            num_blocks_cum = warp_prefix_sum(num_blocks);
            blocks_in_group = __shfl_sync(0xffffffff, num_blocks_cum, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += blocks_in_group * params.num_head;
        }

        int group_start_tile = group_end_tile - blocks_in_group * params.num_head;
        int const batch_idx_in_group = __popc(__ballot_sync(
            0xffffffff, group_start_tile + num_blocks_cum * params.num_head <= next_tile_idx));
        bidb += batch_idx_in_group;
        num_blocks = __shfl_sync(0xffffffff, num_blocks, batch_idx_in_group);
        group_start_tile += (batch_idx_in_group == 0
            ? 0
            : __shfl_sync(0xffffffff, num_blocks_cum, batch_idx_in_group - 1)) * params.num_head;

        int const mh_block = next_tile_idx - group_start_tile;
        int block, bidh;
        if constexpr (LPT) {
            // Group Q heads sharing the same KV head for L2 cache reuse, longest tiles first.
            int const nheads_in_l2 = params.qhead_per_khead;
            int const mh_in_l2 = nheads_in_l2 * num_blocks;
            int const section_idx = mh_block / mh_in_l2;
            int const l2_mod = mh_block - section_idx * mh_in_l2;
            int const nheads_rem = params.num_head - section_idx * nheads_in_l2;
            int const nheads_sec = nheads_in_l2 <= nheads_rem ? nheads_in_l2 : nheads_rem;
            block = l2_mod / nheads_sec;
            int const bidh_res = l2_mod - block * nheads_sec;
            bidh = section_idx * nheads_in_l2 + bidh_res;
            if constexpr (!IsBwd) {
                // Causal fwd: larger m_block has more KV work, schedule descending.
                block = num_blocks - 1 - block;
            }
            // Causal bwd: smaller n_block has more Q work, schedule ascending (block as-is).
        } else {
            bidh = mh_block / num_blocks;
            block = mh_block - bidh * num_blocks;
        }
        return {group_start_tile, block, bidh, bidb};
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void init_consumer() const {}

    CUTLASS_DEVICE
    void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int const new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            WorkTileInfo work_info = {
                __shfl_sync(0xffffffff, current_work.tile_idx, 1),
                current_work.block, current_work.bidh, current_work.bidb
            };
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int4 const w = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return WorkTileInfo{w.x, w.y, w.z, w.w};
        }
    }
};

// Locate the unique sequence b in [0, num_seqs) such that
//   cu_seqlens_padded[b] <= pos < cu_seqlens_padded[b + 1],
// handling consecutive zero-length sequences cleanly (rightmost b with cu[b] <= pos).
CUTLASS_DEVICE
int dm_varlen_locate_seq(int const* __restrict__ cu_seqlens_padded, int num_seqs, int pos) {
    int lo = 0;
    int hi = num_seqs;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) >> 1;
        if (cu_seqlens_padded[mid] <= pos) { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// Single-tile (non-persistent) varlen backward scheduler.
// Eliminates persistent smem_k/v vs epilogue smem_dk/dv union races and needs no
// tile_count_semaphore memset or host sync (100% CUDA-Graph compatible).
// Grid: (total_padded / kBlockN, num_head, 1).
template <int kBlockN, bool Is_causal>
class DMVarlenSingleTileBwdScheduler {
public:
    using SharedStorage = int;

    struct Params {
        int num_block;
        int num_head;
        int num_batch;
        int const* cu_seqlens;
        int const* seqused;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.cu_seqlens != nullptr);
        assert(args.seqused != nullptr);
        return {
            args.num_blocks,
            args.num_head,
            args.num_batch,
            args.cu_seqlens,
            args.seqused
        };
    }

    static dim3
    get_grid_shape(Params const& params, int /*num_sm*/) {
        return {uint32_t(params.num_block), uint32_t(params.num_head), 1u};
    }

    struct WorkTileInfo {
        int32_t n_block;
        int32_t bidh;
        int32_t bidb;

        CUTLASS_DEVICE
        bool is_valid(Params const& /*params*/) const {
            return n_block >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& /*params*/) const {
            return {n_block, bidh, bidb, int32_t(0)};
        }
    };

    CUTLASS_DEVICE
    explicit DMVarlenSingleTileBwdScheduler(SharedStorage* const /*smem_scheduler*/) {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        int const global_nb = int(blockIdx.x);
        int const bidh = int(blockIdx.y);
        int const pos = global_nb * kBlockN;
        int const bidb = dm_varlen_locate_seq(params.cu_seqlens, params.num_batch, pos);
        int const local_nb = (pos - params.cu_seqlens[bidb]) / kBlockN;
        int const actual = params.seqused[bidb];
        if (local_nb * kBlockN >= actual) {
            return {-1, bidh, bidb};
        }
        return {local_nb, bidh, bidb};
    }

    CUTLASS_DEVICE
    void init_consumer() const {}

    CUTLASS_DEVICE
    void prefetch_next_work(Params const& /*params*/, WorkTileInfo& /*current_work*/) const {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& /*params*/, WorkTileInfo const& /*current_work*/) const {
        return {-1, 0, 0};
    }
};

} // namespace flash
