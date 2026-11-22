"""Hopper (SM90) benchmark comparing dense (padded to max_len) vs Route-A packed varlen attention
for forward, backward, end-to-end latency, and peak CUDA memory."""
import argparse
import math

import torch

from delta_matching import (
    dm_attn_func,
    dm_attn_varlen_func,
    pack_varlen_sequences,
    varlen_align,
)


def _bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _measure_peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return max(0, peak - base) / (1024.0 * 1024.0)


def run_case(name: str, seqlens: list[int], H: int, Hk: int, D: int, causal: bool, mode: str, warmup: int, iters: int):
    dev = "cuda"
    BF = torch.bfloat16
    align = varlen_align(D)
    B = len(seqlens)
    max_padded = ((max(seqlens) + align - 1) // align) * align

    g = torch.Generator(device="cpu").manual_seed(42)
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in seqlens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in seqlens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in seqlens]
    do_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in seqlens]

    # Dense padded to max_padded: (B, max_padded, H, D)
    qd = torch.zeros((B, max_padded, H, D), device=dev, dtype=BF)
    kd = torch.zeros((B, max_padded, Hk, D), device=dev, dtype=BF)
    vd = torch.zeros((B, max_padded, Hk, D), device=dev, dtype=BF)
    dod = torch.zeros((B, max_padded, H, D), device=dev, dtype=BF)
    for b, n in enumerate(seqlens):
        qd[b, :n] = q_list[b]
        kd[b, :n] = k_list[b]
        vd[b, :n] = v_list[b]
        dod[b, :n] = do_list[b]

    # Route-A packed varlen
    qv, kv, vv, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dov = torch.zeros_like(qv)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), seqlens)):
        dov[start:start + n] = do_list[b]
    total_padded = int(qv.shape[0])
    total_sq = int(sq[-1].item())

    # 1. Forward only
    with torch.no_grad():
        dense_fwd = _bench_ms(lambda: dm_attn_func(qd, kd, vd, causal=causal, mode=mode), warmup, iters)
        varlen_fwd = _bench_ms(
            lambda: dm_attn_varlen_func(qv, kv, vv, cu, actual, cu_padded_sq=sq, causal=causal, mode=mode, total_padded_sq=total_sq),
            warmup,
            iters,
        )

    # 2. Forward + Backward (with autograd)
    def dense_fb():
        q = qd.detach().requires_grad_(True)
        k = kd.detach().requires_grad_(True)
        v = vd.detach().requires_grad_(True)
        o = dm_attn_func(q, k, v, causal=causal, mode=mode)
        o.backward(dod)

    def varlen_fb():
        q = qv.detach().requires_grad_(True)
        k = kv.detach().requires_grad_(True)
        v = vv.detach().requires_grad_(True)
        o = dm_attn_varlen_func(q, k, v, cu, actual, cu_padded_sq=sq, causal=causal, mode=mode, total_padded_sq=total_sq)
        o.backward(dov)

    dense_tot = _bench_ms(dense_fb, warmup, iters)
    varlen_tot = _bench_ms(varlen_fb, warmup, iters)
    dense_bwd = max(0.0, dense_tot - dense_fwd)
    varlen_bwd = max(0.0, varlen_tot - varlen_fwd)

    dense_mem = _measure_peak_mb(dense_fb)
    varlen_mem = _measure_peak_mb(varlen_fb)

    # Effective attention FLOPs (sum_i 4 * H * N_i^2 * D * (0.5 if causal else 1.0) for fwd, 2.5x for bwd)
    actual_sq = sum(n * n for n in seqlens)
    dense_sq = B * max_padded * max_padded
    work_ratio = dense_sq / max(1, total_sq)

    print(
        f"[{name}] D={D} H={H}/{Hk} mode={mode} causal={causal} "
        f"tokens={sum(seqlens)} packed={total_padded} dense={B * max_padded} sq_ratio={work_ratio:.2f}x\n"
        f"  FWD: dense={dense_fwd:7.3f} ms | varlen={varlen_fwd:7.3f} ms | speedup={dense_fwd / varlen_fwd:5.2f}x\n"
        f"  BWD: dense={dense_bwd:7.3f} ms | varlen={varlen_bwd:7.3f} ms | speedup={dense_bwd / varlen_bwd:5.2f}x\n"
        f"  TOT: dense={dense_tot:7.3f} ms | varlen={varlen_tot:7.3f} ms | speedup={dense_tot / varlen_tot:5.2f}x\n"
        f"  MEM: dense={dense_mem:7.1f} MB | varlen={varlen_mem:7.1f} MB | saving={dense_mem / max(1e-6, varlen_mem):5.2f}x"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--mode", choices=("matched", "stale"), default="matched")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    print(f"GPU: {torch.cuda.get_device_name(0)} | mode={args.mode} | warmup={args.warmup} | iters={args.iters}\n")

    # 1. Equal aligned lengths (overhead / parity check)
    run_case("equal-aligned", [2048, 2048, 2048, 2048], H=32, Hk=8, D=128, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)
    run_case("equal-aligned", [2048, 2048, 2048, 2048], H=16, Hk=4, D=256, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)

    # 2. Skewed long-tail LLM batch (1 long + 7 short sequences)
    skewed_4k = [4096, 512, 384, 256, 200, 175, 130, 97]
    run_case("skewed-4k", skewed_4k, H=32, Hk=8, D=128, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)
    run_case("skewed-4k", skewed_4k, H=16, Hk=4, D=256, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)

    # 3. Heavy skewed 8k batch (1 very long + 7 medium/short)
    skewed_8k = [8192, 1024, 768, 512, 384, 256, 256, 128]
    run_case("skewed-8k", skewed_8k, H=32, Hk=8, D=128, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)
    run_case("skewed-8k", skewed_8k, H=16, Hk=4, D=256, causal=True, mode=args.mode, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
