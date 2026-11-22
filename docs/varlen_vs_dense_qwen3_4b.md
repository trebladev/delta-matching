# Varlen vs dense: Qwen3-4B, B=32, seqlen=8192

Measured on 2026-11-22 against `block-align` (`code_sha=05272f975244`, Delta-Matching-128 v3).

This is an **equal-length parity / overhead** comparison, not a padding-savings case: all 32 sequences are length 8192, which is already a multiple of the Route-A align (256 at head dim 128), so packed tokens equal dense tokens.

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| kernel | `dm_attn_func` (dense) vs `dm_attn_varlen_func` (Route-A packed) |
| mode | `matched`, causal |
| model shape | Qwen3-4B GQA: `H=32`, `Hk=8`, `D=128` |
| input | `B=32`, `seqlen=8192` (262144 tokens either layout) |
| timing | CUDA events, warmup=5, iters=10 |
| FLOPs | causal fwd `2·H·D·B·N²`, bwd `5·H·D·B·N²` |
| backward split | `DM_BWD_HEAD_GROUPS=auto`, `DM_BWD_WS_LIMIT_GB=8`, `DM_BWD_STREAMS=1` → **4 query heads per call** (one KV group). Full `dS'` workspace would be `B·H·N²` ≈ 64 GiB. |

## Results

| | FWD | BWD | FWD+BWD | Peak mem |
|---|---:|---:|---:|---:|
| **dense** | 24.645 ms (713.8 TF/s) | 110.444 ms (398.2 TF/s) | 135.089 ms (455.8 TF/s) | 21.4 GB |
| **varlen** | 28.118 ms (625.7 TF/s) | 102.270 ms (430.0 TF/s) | 130.388 ms (472.2 TF/s) | 20.8 GB |
| **varlen vs dense** | 0.88× (14% slower) | 1.08× (8% faster) | 1.04× (3.6% faster) | ~1.03× saving |

## Conclusion

On this equal 8k shape the two paths are close. Varlen pays a small forward overhead (~3.5 ms) from the packed scheduler / metadata, is slightly faster in the backward, and ends ~4% ahead end-to-end (130 ms vs 135 ms) with essentially the same peak memory.

Varlen is not expected to save compute or memory here: there is no padding. The interesting regime is a skewed batch, where dense pads every sequence to `max_len` and varlen only pays per-sequence aligned squares.
