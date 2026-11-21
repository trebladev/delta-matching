# Native packed Varlen handoff

更新日期：2026-11-20

## 当前状态

仓库已经包含一条原生 CUDA packed Varlen FP8 路径，API 形状接近 FlashAttention 3 的 `flash_attn_varlen_func`。当前实现适合继续调试和扩展，causal 场景下的非整块长度仍有阻塞性数值问题，因此当前状态不应作为完整 FA3 等价实现发布。

本地 `main` 在以下提交之后继续包含未提交的非整块长度改动：

```text
5bd2b0a Support seqused prefixes in native varlen
0cede35 Keep padded varlen FP8 scales finite
8d434ec Serialize varlen scheduler metadata preparation
2384b20 Refine varlen scheduling and empty segments
8ab8a28 Add packed varlen FP8 attention path
```

## 已实现的能力

- packed BF16 Q/K/V 输入，输出 packed BF16 attention 结果
- d256，以及编译了 `FA3FP11_HD128=1` 时的 d128
- MHA、GQA、MQA
- causal 和 non-causal 前向
- native FP8 backward，包含 packed dQ、dK、dV
- `cu_seqlens_q` 和 `cu_seqlens_k`
- Q/K 长度不相等
- 空 batch segment 和 zero-token 输入
- packed LSE 的内部保存与输出回收
- `seqused_q` 和 `seqused_k` 的逻辑前缀
- 非 block-aligned 物理长度的内部对齐 workspace
- 逻辑前缀量化，物理 suffix code 置零，未使用输出置零，LSE 置为 `-inf`
- padding scale slot 使用有限值 `1`
- scheduler metadata 在当前 stream 上串行准备，避免 metadata 可见性竞态
- d256 causal Varlen backward 在 H200 上使用 FP32 reference 重算，规避 native pass2 间歇性 dQ/dV 非有限值；d128 仍使用 native backward

## 未提交改动

当前工作区涉及三个文件：

- `csrc/delta_matching_api.cpp`
  - 前向按 Q/K tile 要求计算内部 `launch_max_q/k`
  - d128 前向 K workspace 按 256-token descriptor 对齐
  - 反向按 64-token Q tile 和对应 K tile 对齐
  - 物理 packed 结果按原始 `cu_seqlens` 回收
  - 校验每个物理 segment 不超过传入的 `max_seqlen`
- `delta_matching/varlen_quant.py`
  - 任意长度先补齐到临时 quantization extent
  - d256 K/V 的 twin 临时长度至少按 64-token 对齐
  - d128 K/V scale rows 按 256-token descriptor 要求补齐
  - padding scale 使用有限值 `1`
- `delta_matching/__init__.py`
  - dO scale rows 按 `max_seqlen_q` 补齐
  - 显式 `seqused` 反向按 ceil block 数压缩 scale，并支持非整块逻辑前缀

## 已验证结果

### CPU

```text
7 passed, 108 skipped
```

GPU 测试由于当前运行环境中的 CUDA 可见性限制被跳过。

### H200 wheel 构建

```bash
CCACHE_DIR=/tmp/ccache MAX_JOBS=2 \
pip wheel --no-build-isolation --no-deps -w /tmp/dm-wheel-check .
```

最终 wheel 构建成功。最近一次 wheel SHA256：

```text
9e0841e02222216d49dbbe2f01dc3505ffaf01aa5282e86ef5bab409b2cdceb4
```

### H200 packed Varlen

使用 Q 长度 `[100, 130]`、K 长度 `[70, 190]`、Q heads=2、K heads=1 验证了非因果 d256 和 d128。显式 `seqused_q=[90,127]`、`seqused_k=[55,129]` 的前向与反向均通过有限值检查，dQ、dK、dV 均为 finite。

已验证空 segment、非等长 causal/non-causal、GQA 以及显式逻辑前缀的多组路径。

## 当前阻塞问题

### 非整块 causal 混合 tile 会产生 NaN

在 H200 上使用单段 Q=100、K=70、`causal=True` 时，输出的部分 causal 行产生 NaN。NaN 行集中在一个 tile 内部的部分 mask 区域，例如行 30 到 93。给物理 workspace 补齐到 128 或 192，或者改用显式 `seqused`，仍可复现该现象。

这说明问题位于 causal mixed-mask tile 的 kernel 路径，单纯的 host padding 没有解决它。优先检查：

H200 上的最小矩阵已经进一步缩小为 Q 长度大于 K 长度时稳定触发。`Q=100,K=70,causal=True` 和 `Q=128,K=70,causal=True` 都产生 NaN，NaN 行固定为每个 64-row tile 中的混合区域。`Q=100,K=70,causal=False`、`Q=100,K=128,causal=True` 和 `Q=128,K=128,causal=True` 均为 finite。显式传入 `seqused_q=[Q]`、`seqused_k=[K]` 仍会触发，因此问题不在 host 端逻辑前缀元数据。

1. `third_party/flash-attention/hopper/mask.h` 中 `causal_row_offset` 与 `SeqlenInfoQKNewK` 的实际长度组合
2. `mainloop_fwd_sm90_fp8f.hpp` 中 mixed-mask tile 的 `fq8_descale`、softmax、PV promote 顺序
3. fully masked 行与部分有效行共存时的 `tOrO` 初始化和 epilogue store

### dQ 间歇性 NaN

此前 H200 测试中，d128 causal GQA 的 dQ 偶发 NaN，d256 GQA 也出现过间歇性反向非有限值。当前没有使用 `nan_to_num` 掩盖问题。需要继续检查：

- backward pass1 的 `psi` 写入
- `smem_dq` 初始化和 fragment ownership
- pass2 的 fragment 写入归属
- varlen scheduler 是否重复分派 tile

Compute Sanitizer racecheck 曾报告 `FlashAttnFwdSm90Fq8F` 与 `VarlenDynamicPersistentTileScheduler` 相关的 shared-memory RAW hazard。该报告尚未判定为工具误报或真实竞态。

## 尚未实现的 FA3 选项

当前 API 会显式拒绝以下选项：`qv`、FP8 descale inputs、sliding window、`attention_chunk`、`softcap`、`num_splits`、`pack_gqa`、`return_attn_probs` 和 `sm_margin`。matched-delta mode 也没有纳入该 packed Varlen API。

## 推荐的后续顺序

1. 为 causal mixed-mask tile 加入最小复现测试并修复 NaN
2. 重复运行 GQA d128/d256 backward，定位 dQ/dK 的竞态或未初始化读
3. 在 H200 上重新跑非整块 d128/d256、causal/non-causal、GQA、空 segment、显式 `seqused` 全套测试
4. 将非整块改动和修复拆成可审阅提交
5. 再评估是否宣称 FA3 行为兼容，并补充尚未实现的 API 选项

## 参考资料

- [FlashAttention ragged PDL/TMA workspace issue](https://github.com/Dao-AILab/flash-attention/issues/3198)
- [Ragged K/V per-CTA TMA descriptor rewrite](https://github.com/Dao-AILab/flash-attention/pull/3071)
- [Q/dO unused tail clearing](https://github.com/Dao-AILab/flash-attention/pull/3177)
- [K/V unused tail clearing](https://github.com/Dao-AILab/flash-attention/pull/3174)
- [zero-token TMA descriptor guard](https://github.com/vllm-project/flash-attention/pull/197)
- [zero-length TMA descriptor issue](https://github.com/Dao-AILab/flash-attention/issues/1343)
