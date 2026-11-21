"""Qwen3-4B-shaped packed Varlen benchmark.

The benchmark uses batch 32 and 8192 total tokens.  It compares the native
packed Varlen path with FA3 when a compiled FA3 extension is available, and
always reports PyTorch SDPA as a numerical reference.
"""
import argparse
import statistics
import time

import torch

from delta_matching import flash_attn_varlen_func


def lengths(kind):
    if kind == "uniform":
        return [256] * 32
    if kind == "skewed":
        raw = [64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304,
               320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 544, 576, 608, 640]
        scale = 8192 / sum(raw)
        vals = [max(1, round(x * scale)) for x in raw]
        vals[-1] += 8192 - sum(vals)
        return vals
    raise ValueError(kind)


def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distribution", choices=("uniform", "skewed"), default="uniform")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    lens_q = lengths(args.distribution)
    lens_k = lens_q[:]
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(8192, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(8192, 8, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    max_len = max(lens_q)

    def dm():
        return flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_len, max_len, causal=True)

    def sdpa():
        outs = []
        for b, n in enumerate(lens_q):
            q_b = q[cu_q[b]:cu_q[b + 1]].transpose(0, 1).unsqueeze(0)
            k_b = k[cu_k[b]:cu_k[b + 1]].transpose(0, 1).unsqueeze(0)
            v_b = v[cu_k[b]:cu_k[b + 1]].transpose(0, 1).unsqueeze(0)
            o = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True, enable_gqa=True)
            outs.append(o.squeeze(0).transpose(0, 1))
        return torch.cat(outs)

    arms = [("delta_matching", dm), ("torch_sdpa_reference", sdpa)]
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3

        def fa3_call():
            out = fa3(q, k, v, cu_q, cu_k, max_len, max_len, causal=True)
            return out[0] if isinstance(out, tuple) else out

        arms.insert(1, ("fa3", fa3_call))
    except Exception as exc:
        print(f"fa3: unavailable ({type(exc).__name__}: {exc})")

    print(f"distribution={args.distribution} batch=32 total_tokens=8192 q_heads=32 kv_heads=8 head_dim=128")
    outputs = {}
    for name, fn in arms:
        ms = median_ms(fn, args.warmup, args.iters)
        outputs[name] = fn()
        print(f"{name}: forward_ms={ms:.3f}")
    ref = outputs["torch_sdpa_reference"].float()
    for name, out in outputs.items():
        if name != "torch_sdpa_reference":
            err = (out.float() - ref).abs()
            print(f"{name}: max_abs={err.max().item():.6g} mean_abs={err.mean().item():.6g} finite={torch.isfinite(out).all().item()}")


if __name__ == "__main__":
    main()
