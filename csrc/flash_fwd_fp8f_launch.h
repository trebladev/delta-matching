/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/
/******************************************************************************
 * ## FQ8 (fa3fp8 Stage E): launch glue for the recipe on FA3's fp8 forward.
 * Fixed config: d256, e4m3 in / bf16 out, kBlockM=128, kBlockN=128, RS PV,
 * IntraWGOverlap, FAITHFUL per-row/per-32-key P-hat softmax (softmax_fq8f.h),
 * in-smem V transpose.  Per-block scales replace the per-tensor descales.
 ******************************************************************************/
#pragma once

#include "cuda_check.h"
#include "cute/tensor.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"
#include "cutlass/cluster_launch.hpp"
#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_fp8f.hpp"
#include "epilogue_fwd.hpp"
#include "flash_fwd_kernel_sm90_fq8f.h"
#include "flash_fwd_fp8f_params.h"

using namespace cute;

// ## Delta-Matching-128 (M19): the launch body is templated on the head dim; the d256 entry point
// (run_flash_fwd_fp8f_d256) keeps its name and configuration verbatim, the d128 entry point instantiates
// the same collective at kHeadDim 128 with the same 128 x 128 tile (FA3's own bf16 hd128 tile).
template <bool Is_causal, bool Varlen, int kHeadDim_>
void run_flash_fwd_fp8f_impl(Flash_fwd_fp8_params &params, cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    using ElementOut = cutlass::bfloat16_t;
    using ArchTag = cutlass::arch::Sm90;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDim_;
    // hd128 with BM192: 192 x 128 tiles, three consumer warpgroups (registers 160 / 32): the per-warpgroup chain is
    // latency-bound, so a third one raises tensor-pipe use.  Each 64-row warpgroup lies in one 128-row Q scale block.
    static constexpr int kBlockM = (kHeadDim_ < 256 && FA3FP11_HD128_BM192) ? 192 : 128;
    // ## Delta-Matching-128 F3 (FA3FP11_HD128_BN256=1, hd128 only): 256-key tiles (four 64-key groups per tile, half the tile
    // boundaries); the d256 configuration stays 128 x 128.
    static constexpr int kBlockN = (kHeadDim_ < 256 && FA3FP11_HD128_BN256) ? 256 : 128;
    static constexpr bool MmaPV_is_RS = true;
    static constexpr bool IntraWGOverlap = true;
    static constexpr int kStages = kHeadDim_ < 256 ? FA3FP11_HD128_FWDST : 2;   // hd128 forward K/V stages

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<_1, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90Fp8F<
        kStages, ClusterShape, TileShape_MNK, kHeadDimV, Element, float, ArchTag,
        Is_causal, false /*Is_local*/, false /*Has_softcap*/, Varlen,
        false /*PagedKVNonTMA*/, false /*AppendKV*/, false /*HasQv*/,
        MmaPV_is_RS, IntraWGOverlap, false /*PackGQA*/, false /*Split*/, false /*V_colmajor*/>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads,
        Varlen, false /*PackGQA*/, false /*Split*/, true /*FP8_TransposeV*/>;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    using SchedulerPersistent = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads,
            NumProducerThreads, false /*Split*/, false /*PackGQA*/, true /*WarpSpecialized*/,
            Is_causal /*LPT*/, true /*Sort*/, true /*Prepared*/>,
        std::conditional_t<!Is_causal,
            flash::StaticPersistentTileScheduler<false /*Split*/>,
            flash::DynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads,
                                                  false /*Split*/, false /*PackGQA*/, true /*WarpSpecialized*/>>>;
    using Scheduler = SchedulerPersistent;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90Fq8F<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    bool const is_varlen_q = Varlen && params.cu_seqlens_q;
    bool const is_varlen_k = Varlen && params.cu_seqlens_k;
    int const seqlen_q = is_varlen_q ? params.total_q : params.seqlen_q;
    int const seqlen_k = is_varlen_k ? params.total_k : params.seqlen_k;
    typename CollectiveMainloop::StrideV v_strides =
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, is_varlen_k ? 0 : params.v_batch_stride);
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, is_varlen_q ? 1 : params.b},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, is_varlen_q ? 0 : params.q_batch_stride},
        static_cast<Element*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, is_varlen_k ? 1 : params.b},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, is_varlen_k ? 0 : params.k_batch_stride},
        static_cast<Element*>(params.v_ptr),
        params.dv,  // headdim_v
        v_strides,
        nullptr,  // ptr_K_new
        {0, params.d, params.h_k, 1},  // shape_K_new
        {0, _1{}, 0, 0},
        nullptr,  // ptr_V_new
        {0, _1{}, 0, 0},
        nullptr,  // ptr_Qv
        {0, _1{}, 0, 0},
        nullptr,  // rotary_cos
        {params.seqlen_k, 0},  // shape_rotary
        {0, _1{}},
        nullptr,  // rotary_sin
        {0, _1{}},
        false,    // is_rotary_interleaved
        nullptr,  // page_table
        {params.b, 0},  // shape_page_table
        {0, _1{}},
        params.scale_softmax,
        // ## FQ8: per-block scales
        params.q8s_ptr, params.k8s_ptr, params.v8s_ptr,
        params.nblk_q, params.nblk_k,
        params.window_size_left, params.window_size_right, 0 /*attention_chunk*/,
        0.0f,  // softcap
        1,     // num_splits
        nullptr,  // kv_batch_idx
        params.cu_seqlens_q, params.cu_seqlens_k, nullptr,  // cu_seqlens
        params.seqused_q, params.seqused_k,           // seqused
        nullptr, nullptr            // leftpad_k, seqlens_rotary
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, is_varlen_q ? 1 : params.b, 1},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, is_varlen_q ? 0 : params.o_batch_stride, 0},
        nullptr,  // oaccum
        {0, _1{}, 0, 0, 0},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, Varlen ? int64_t(0) : int64_t(params.h * seqlen_q), 0},  // stride_LSE
        nullptr,
        {_1{}, 0, 0, 0},
        params.h_k,
        params.cu_seqlens_q, params.seqused_q
    };

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
        params.num_splits_dynamic_ptr, params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr, params.num_nheads_in_l2_ptr
    };

    if constexpr (Varlen) {
        // FA3's persistent varlen scheduler needs per-batch tile metadata
        // before constructing its kernel parameters.  The metadata launch is
        // a no-op for the fixed-length path and is intentionally kept here so
        // packed calls follow the same scheduler contract.
        prepare_varlen_num_blocks(params, stream, false /*PackGQA*/, kBlockM, kBlockN,
                                  params.prepare_varlen_pdl);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });
    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    if (std::getenv("DM_DEBUG")) {
        fprintf(stderr, "[delta_matching fwd] varlen=%d grid=(%d,%d,%d) block=(%d,%d,%d) smem=%d total=(%d,%d) max=(%d,%d)\\n",
                int(Varlen), grid_dims.x, grid_dims.y, grid_dims.z, block_dims.x, block_dims.y, block_dims.z,
                smem_size, params.total_q, params.total_k, params.seqlen_q, params.seqlen_k);
    }
    auto const launch_status = cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params,
                                                                    Varlen && params.prepare_varlen_pdl /*launch_with_pdl*/);
    if (launch_status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[delta_matching fwd] CUTLASS launch status=%s cuda=%s\\n",
                cutlass::cutlassGetStatusString(launch_status), cudaGetErrorString(cudaGetLastError()));
        exit(1);
    }
}

template <bool Is_causal>
void run_flash_fwd_fp8f_d256(Flash_fwd_fp8_params &params, cudaStream_t stream) {
    run_flash_fwd_fp8f_impl<Is_causal, false, 256>(params, stream);
}

template <bool Is_causal>
void run_flash_fwd_fp8f_varlen_d256(Flash_fwd_fp8_params &params, cudaStream_t stream) {
    run_flash_fwd_fp8f_impl<Is_causal, true, 256>(params, stream);
}

// ## Delta-Matching-128
template <bool Is_causal>
void run_flash_fwd_fp8f_d128(Flash_fwd_fp8_params &params, cudaStream_t stream) {
    run_flash_fwd_fp8f_impl<Is_causal, false, 128>(params, stream);
}

template <bool Is_causal>
void run_flash_fwd_fp8f_varlen_d128(Flash_fwd_fp8_params &params, cudaStream_t stream) {
    run_flash_fwd_fp8f_impl<Is_causal, true, 128>(params, stream);
}
