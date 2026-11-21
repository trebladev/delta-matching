# Varlen 性能排查记录

更新时间：2026-11-21

## 测试环境

- GPU：NVIDIA H200
- Qwen3-4B attention shape
  - Q：`[32, 256, 32, 128]`
  - K/V：`[32, 256, 8, 128]`
  - GQA ratio：`32 / 8 = 4`
  - causal attention
- packed varlen 总 token 数：8192
- dense 对照使用等价的 `B=32, N=256`

## 当前端到端结果

最近一次 native backward 测量如下，数值输出和 dQ/dK/dV 均为 finite。

| 路径 | Forward | Forward + Backward |
|---|---:|---:|
| native varlen | 2.67 ms | 5.43 ms |
| dense | 0.365 ms | 1.00 ms |

不等长 skewed varlen 分布仍明显更慢。

| 分布 | Forward |
|---|---:|
| uniform | 2.49 ms |
| skewed | 33.30 ms |

skewed case 的总 token 数同样为 8192。长度不均匀会增加 descriptor 对齐、workspace padding 和动态 tile 调度开销。

## Varlen 内部拆分

对 uniform Qwen3 shape 直接测量各阶段，结果如下。

| 阶段 | 耗时 |
|---|---:|
| Q/K/V quantization | 0.88 ms |
| dO quantization | 0.59 ms |
| Varlen forward kernel | 1.68 ms |
| Varlen native backward kernel | 1.86 ms |

dense 固定路径的端到端耗时为 0.36 ms forward 和 1.08 ms forward+backward。固定 shape 路径将量化、tile 调度和 attention 主循环结合得更紧，因此不能只用 attention FLOPs 解释两者差距。

## 已完成的优化

### Uniform packed batch 批量量化

原实现对每个 segment 独立调用量化 producer。Qwen3 的 32 个等长 segment 会产生大量小 kernel launch。

现在 uniform packed batch 一次通过 `_quant_native` 处理整个 batch，并恢复为 packed layout。推理模式还会跳过不需要的 Q/K twin 生成。

### dO quantization 批处理

原实现的 dO quantization 对 32 个 segment 逐个启动 producer，测得约 8.2 ms。现在等长 batch 走批量路径，降到约 0.59 ms。

### d128 scale descriptor 补齐

d128 producer 的 Q scale 原生 block 和 varlen descriptor 的 row contract 不同。现在会补齐额外的有限 scale slot。K/V scale 也会按 d128 的 256-token descriptor 要求补齐，解决 skewed case 的 `scale tensors are too short for max lengths` 错误。

### backward NaN 修复

Varlen backward 的 pass 2 会遍历规则 workspace 中的 tile。原先 `ds8` 和 `psi` 使用 `torch::empty`，未被 pass 1 覆盖的 padding 或 fully-masked tile 可能读取 allocator 的随机值。随机 finite `psi` 会进入 dQ 累加并产生 NaN。

现在这两个 workspace 使用零初始化。零值对应 pass 2 已定义的 sentinel tile，native causal backward 已重新启用。

H200 上已验证：

- d128 GQA causal：dQ、dK、dV 全部 finite
- d256 GQA causal：dQ、dK、dV 全部 finite
- Qwen3 d128 shape 连续重复测试：dQ、dK、dV 全部 finite

## 剩余性能瓶颈

### Varlen quantization 固定开销

Q/K/V 和 dO 需要单独生成 FP8 code、scale 和 twin。即使批量化后，uniform case 仍有约 1.47 ms，这部分在 dense 固定路径中被更紧密地融合。

### Forward dynamic scheduler

Varlen forward kernel 约 1.68 ms，而 dense forward 约 0.36 ms。额外成本来自 packed segment offset、动态 tile dispatch、TMA descriptor 边界和 GQA 的 varlen head 映射。

### Backward workspace 和 dQ accumulation

native backward kernel 约 1.86 ms。主要包含 dS' tile 写入、`psi` metadata、dQ pass 2 的 workspace accumulation，以及 GQA dK/dV fold。固定 dense 路径可以使用连续 tile 和固定 batch stride，利用率更高。

### Skewed segment

skewed case 需要按共同的最大 descriptor extent 运行，短 segment 的 padding tile 不产生有效计算，却仍带来 descriptor 和 scheduler 工作。要让 skewed 性能接近 dense，需要进一步实现按 segment 的真实 tile dispatch 和更细粒度的 TMA descriptor，而不是共同最大长度 workspace。

## 复现命令

使用工作区构建出的 CUDA 扩展：

```bash
export DM_SO=$PWD/build/lib.linux-x86_64-cpython-312/delta_matching/_C.cpython-312-x86_64-linux-gnu.so
export PYTHONPATH=$PWD

python benchmarks/bench_varlen_qwen3.py --distribution uniform --warmup 10 --iters 30
python benchmarks/bench_varlen_qwen3.py --distribution skewed --warmup 5 --iters 10
python /tmp/bench_fb.py
python /tmp/profile_parts.py
```

`/tmp/bench_fb.py` 和 `/tmp/profile_parts.py` 是本次排查使用的临时 profiling 脚本。正式 benchmark 当前只覆盖 forward，训练性能需要同时观察 forward+backward。

## 当前结论

最大的逐 segment quantization 瓶颈已经消除，native backward 的 dQ NaN 也已修复。uniform varlen forward 已接近可用范围，训练路径仍比 dense 慢约 5 倍。剩余工作集中在动态 scheduler、descriptor/workspace padding 和 native backward 的 dQ accumulation 利用率。
