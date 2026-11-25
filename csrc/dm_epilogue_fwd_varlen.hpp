/******************************************************************************
 * Route-A Block-Aligned Varlen Forward Epilogue (Hopper SM90)
 *
 * Retains aligned TMA Store into (TotalPaddedTokens, H, D) for every tile
 * satisfying (m_block + 1) * kBlockM <= padded_len[b]:
 *   - D=256 (kBlockM=128, ALIGN=128): 100% of tiles use aligned TMA Store.
 *   - D=128 (kBlockM=192, ALIGN=128): interior tiles with
 *     (m_block + 1) * 192 <= padded_len[b] use aligned TMA Store; a tail tile
 *     crossing padded_len[b] (including every padded_len=128 sequence, whose
 *     first 192-row tile overruns by 64) falls back to predicated vectorized
 *     store so sequence b+1 is never overwritten.
 * Pad rows [seqlens_actual[b], padded_len[b]) are zeroed in registers before
 * storing so O stays strictly 0.0 in every pad slot.
 ******************************************************************************/
#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"

#include "seqlen.h"
#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <
    class TileShape_MNK_PV_,
    class ClusterShape_,
    class Element_,
    class ArchTag_,
    int NumEpilogueThreads_,
    bool FP8PermuteCol = true
>
struct CollectiveEpilogueFwdVarlen {
    using TileShape_MNK_PV = TileShape_MNK_PV_;
    using ClusterShape = ClusterShape_;
    using Element = Element_;
    using ElementPartial = float;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Varlen = true;
    static constexpr bool PackGQA = false;
    static constexpr bool Split = false;
    static constexpr bool Use_smem = true;
    static constexpr bool Use_TMA_O = true;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(CUTE_STATIC_V(size(ClusterShape{})) == 1);
    static_assert(sizeof(Element) == 2);

    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});
    static constexpr bool LargeHeadDimV = kHeadDimV > 256;

    using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimV % kGmemElemsPerStore == 0);
    static constexpr int kBytePerRow = kHeadDimV * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom = Layout<
        Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);
    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));

    using SmemLayoutAtomOTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_PV{})),
        decltype(cute::get<1>(TileShape_MNK_PV{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));

    using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
    using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
    using StrideLSE = cute::Stride<_1, int64_t, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t, int32_t>;

    using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK_PV{}));
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<
        StrideO, Element, EpilogueTile_MN>());
    using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    using TMA_O = decltype(make_tma_copy(
        GmemTiledCopyOTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
        SmemLayoutO{},
        select<0, 1>(TileShape_MNK_PV{}),
        _1{}));

    struct Arguments {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        ElementPartial* ptr_O_partial;
        StrideO const stride_O_partial;
        float* ptr_LSE;
        StrideLSE const stride_LSE;
        float* ptr_LSE_partial;
        StrideLSE const stride_LSE_partial;
        int32_t const nheads_kv;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
    };

    struct Params {
        Element* ptr_O;
        ShapeO const shape_O;
        StrideO const stride_O;
        float* ptr_LSE;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE;
        TMA_O tma_store_O;
        int const* cu_seqlens;
        int const* seqused;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
        TMA_O tma_store_O = make_tma_copy(
            GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, select<0, 1>(TileShape_MNK_PV{}), _1{});
        return {
            args.ptr_O, args.shape_O, args.stride_O,
            args.ptr_LSE, select<0, 2, 3, 4>(args.shape_O), args.stride_LSE,
            tma_store_O, args.cu_seqlens, args.seqused
        };
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO& tOrO,
          FrgTensorLSE const& lse,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int const offset_o = params.cu_seqlens[bidb];
        int const padded_len = params.cu_seqlens[bidb + 1] - offset_o;
        int const seqlen_o = params.seqused[bidb];
        bool const can_tma_store = (m_block + 1) * kBlockM <= padded_len;

        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});

        // Convert O to Element and strictly zero pad rows (>= seqlen_o) in registers
        // before writing to smem / gmem so aligned TMA stores keep pad slots at 0.0.
        Tensor tOrO_out = make_tensor_like<Element>(tOrO);
        flash::convert_type_out(tOrO, tOrO_out);
        if ((m_block + 1) * kBlockM > seqlen_o) {
            Tensor tOrO_out_rowcol = make_tensor(tOrO_out.data(), flash::convert_layout_acc_rowcol(tOrO_out.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tOrO_out_rowcol); ++mi) {
                if (m_block * kBlockM + get<0>(taccOcO_row(mi)) >= seqlen_o) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrO_out_rowcol); ++ni) {
                        tOrO_out_rowcol(mi, ni) = Element(0);
                    }
                }
            }
        }
        if constexpr (FP8PermuteCol) {
            flash::permute_output_fp8_Vcolmajor(tOrO_out);
        }

        // Wait until all WGs have finished reading smem_v before reusing smem_o.
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        cute::copy(smem_tiled_copy_O, smem_thr_copy_O.retile_S(tOrO_out), smem_thr_copy_O.partition_D(sO));

        // Write LSE for valid rows only (pad rows remain 0.0f from zero-init).
        Tensor mLSE = make_tensor(
            make_gmem_ptr(params.ptr_LSE + int64_t(offset_o) * get<0>(params.stride_LSE)),
            params.shape_LSE, params.stride_LSE)(_, bidh, _0{}, _0{});
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
            if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) {
                mLSE(row) = lse(mi);
            }
        }

        // Fence smem writes and sync all epilogue threads before reading sO.
        cutlass::arch::fence_view_async_shared();
        flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        if (can_tma_store) {
            // Aligned TMA store inside sequence b's padded region (cannot overwrite seq b+1).
            if (thread_idx == 0) {
                Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh, _0{}, _0{});
                Tensor gO = local_tile(
                    domain_offset(make_coord(offset_o, _0{}), mO),
                    select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
                auto block_tma_O = params.tma_store_O.get_slice(_0{});
                cute::copy(params.tma_store_O, block_tma_O.partition_S(sO), block_tma_O.partition_D(gO));
                tma_store_arrive();
                tma_store_wait<0>();
                shared_storage.pipelines.barrier_O.arrive(uint32_t(0));
            }
        } else {
            // Tail tile where (m_block + 1) * kBlockM > padded_len[b] (only occurs at D=128 kBlockM=192):
            // copy sO -> rmem, release smem_o via barrier_O (count=1), then predicated store to rows < seqlen_o.
            Tensor mO = make_tensor(
                make_gmem_ptr(params.ptr_O + int64_t(offset_o) * get<0>(params.stride_O)),
                params.shape_O, params.stride_O)(_, _, bidh, _0{}, _0{});
            Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
            GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
            Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
            Tensor tOrO_g = make_fragment_like(tOsO);
            cute::copy(gmem_tiled_copy_O, tOsO, tOrO_g);
            cutlass::arch::fence_view_async_shared();
            flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            if (thread_idx == 0) {
                shared_storage.pipelines.barrier_O.arrive(uint32_t(0));
            }
            Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O);
            }
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO_g, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM);
        }
    }

    CUTLASS_DEVICE void store_tail() {}

    CUTLASS_DEVICE void
    store_zero(Params const& params,
               int thread_idx,
               cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        int const offset_o = params.cu_seqlens[bidb];
        int const seqlen_o = params.seqused[bidb];
        Tensor mLSE = make_tensor(
            make_gmem_ptr(params.ptr_LSE + int64_t(offset_o) * get<0>(params.stride_LSE)),
            params.shape_LSE, params.stride_LSE)(_, bidh, _0{}, _0{});
        if (thread_idx < kBlockM) {
            int const row = m_block * kBlockM + thread_idx;
            if (row < seqlen_o) { mLSE(row) = -INFINITY; }
        }
        // O is zero-initialized on the host for varlen, so pad and empty tiles are already 0.0.
    }
};

} // namespace flash
