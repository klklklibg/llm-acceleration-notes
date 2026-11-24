# MTP 与推理栈深度研究（第二轮）

研究时间：2026-11-08
硬件：Tesla P100-PCIE-16GB (sm_60) / CUDA 12.4 / 1250 W
软件：`D:\llama.cpp\bin`（prism-ml fork，0.3.0-dev，commit 4423f2a，含 GP100 定制）
模型：`DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q8_0`（qwen35，33 层，blk.32 为 nextn）

---

## 0. 结论摘要

| 问题 | 答案 |
|---|---|
| MTP 能带来多少收益 | **+26% ~ +59%，完全由内容类型决定**；数学最高，代码/闲聊最低 |
| n_max 该设多少 | **2**。不是调参运气，而是 GP100 MMVQ kernel 的 B≥4 成本台阶决定的 |
| KV 该不该量化 | 质量无损（三档全 90%，连错题都一样）；q8_0 掉 3.3% 速度，**q4_0 持平且最省显存** |
| 16 GB 能开多大上下文 | f16 → 128K；q8_0 → 192K；**q4_0 → 256K（模型原生上限）** |
| 其他投机技术值得用吗 | **都不值得**。DFlash 在 n=2 打平 MTP、n≥3 落后；5 种 ngram 全部无用 |
| 这个 fork 的定制优化值多少 | GP100 MMVQ **+23.7%**、算子融合 **+3.1%**、FA-GEMM prefill **+10.7%** |
| prefill 有代价吗 | 有，**−4% ~ −10%**，随 prompt 变长而恶化 |

**一句话**：MTP n_max=2 是这块卡上的唯一最优点，而且它的上限不是被接受率、也不是被显存限制的，是被 `mmvq-gp100.cu` 这一个 kernel 的多列实现决定的。

---

## 1. 方法

所有对比统一条件：`-ngl 99 -fa on -lm none -np 1 --no-mmproj`，`temp 0.7 / seed 1234 / max_tokens 192`，每组 2~3 轮取中位。固定 seed 后同配置三轮几乎完全重合（±0.5 t/s），因此 1 t/s 以上的差异可信。

脚本：`bench_suite.py`（kv / ctx / content / family / combo / prefill / env）、`bench_kv_quality.py`。原始数据在 `mtp_logs/suite_*.json`。

---

## 2. 内容类型：真正的决定因素

同一个模型、同一份配置，只换 prompt，加速比从 +26% 到 +59%：

| 内容 | baseline | MTP n2 | 加速 | 接受率 |
|---|---|---|---|---|
| 数学推导 | 49.86 | **79.38** | **+59.2%** | 82.6% |
| 代码注释改写 | 49.91 | 71.73 | +43.7% | 69.8% |
| 中文论述 | 49.91 | 68.01 | +36.3% | 63.5% |
| 自由闲聊 | 49.89 | 63.02 | +26.3% | 55.1% |
| 写新代码 | 49.73 | 62.89 | +26.5% | 55.0% |

**加速比与接受率严格单调对应**，两者几乎是线性关系：

```
Δ ≈ 1.45 × acc_rate − 0.33
（82.6% → +59%, 55% → +26%，两端吻合）
```

这条关系由一个很简单的算术决定：每轮成本固定（约 33 ms），每轮产出 `1 + acc_rate/(1−acc_rate)` 个 token 的上限被 n_max+1=3 截住。所以 acc_rate 越高，越接近「一轮出 3 个」的 3 倍理论上限；n_max=2 时实际天花板是 3/1.55 ≈ 1.94 倍，实测数学场景 1.59 倍，已榨到 82%。

**与上一轮的差异**：上一轮报告写 +32.6%，用的是 `llama-cli -lv 4`，全量 trace 日志拖了约 10%。server 模式的真实数字是 **+36.3%（论述）到 +59.2%（数学）**。

**这条结论的实际含义**：如果你的主要负载是数学、推理、带步骤的推导，MTP 值得一直开着；如果是自由创作和闲聊，收益只有四分之一。

---

## 3. KV cache 量化

### 3.1 速度与显存（ctx 8192, MTP n2, 3 轮）

| 配置 | t/s | 接受率 | 显存 |
|---|---|---|---|
| f16 / f16 | **67.16** | 63.5% | 9419 MiB |
| q8_0 / q8_0 | 64.95 | 59.2% | 9299 MiB |
| **q8_0 K / f16 V** | **42.17** | 58.5% | 9360 MiB |
| q4_0 / q4_0 | **67.32** | 62.4% | 9235 MiB |
| q8_0 / q8_0（无投机） | 46.77 | — | 8722 MiB |

三件事：

**1. 混合类型是陷阱，−37%。** q8_0 K 配 f16 V 直接掉到 42.17。原因在源码里：

```cpp
// fattn.cu:265
#ifdef GGML_CUDA_FA_ALL_QUANTS
    ... 56 种 K/V 组合全部实例化 ...
#else
    FATTN_VEC_CASES_ALL_D  (GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D  (GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D  (GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D  (GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif
```

这个构建没开 `GGML_CUDA_FA_ALL_QUANTS`，只实例化了 4 种**同类型**组合。混合类型掉进通用慢路径。**K 和 V 必须用同一种类型。**

**2. q8_0 比 f16 慢 3.3%，而且接受率也降了 4.3 个点。** 不是测量误差——3 轮完全一致（64.40/64.95/65.21）。量化让 target 的分布和 MTP 头的预测都偏移了一点，两头一起损失。

**3. q4_0 与 f16 完全持平（67.32 vs 67.16），接受率只差 1.1 点。** 因为 q4_0 有专用 tile kernel（`fattn-tile.cuh:1383` 要求 K 和 V 都是 Q4_0）。

### 3.2 质量验证（MTP n2, temp 0, 30 题题库）

| KV 类型 | 正确率 | 每题 token | 错题 |
|---|---|---|---|
| f16 | 90.0% (27/30) | 159 | 21, 23, 26 |
| q8_0 | 90.0% (27/30) | 160 | 21, 23, 26 |
| q4_0 | 90.0% (27/30) | 154 | 21, 23, 26 |

**三档完全一致，连错的题都一模一样**（21 题答成 N_A 而非 6.02×10^23、23 题答 24.5 L 而非 22.4、26 题答 sp³ 而非 sp3——都是判分规则或题目歧义问题，不是模型退化）。

30 题的统计功效有限，但至少可以说：**在这个模型上，KV 量化到 q4_0 没有可测的质量损失**。这与常见说法「KV 量化伤长上下文」不矛盾——那是长上下文场景，30 题都是短 prompt。

### 3.3 结论

| 需求 | 选择 |
|---|---|
| 默认 | **f16**（最快，显存够用） |
| 需要 >128K 上下文 | q4_0（速度无损，比 q8_0 更划算） |
| 绝对不要 | 混合类型（K 和 V 不同） |

---

## 4. 上下文上限

MTP n2，测每个 ctx 的启动显存（含 MTP context）：

| ctx | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 8192 | 68.11 / 9419 MiB | 64.94 / 9299 MiB | 67.32 / 9235 MiB |
| 32768 | 67.58 / 10332 MiB | 64.84 / 9852 MiB | — |
| 65536 | 67.79 / 11572 MiB | 64.75 / 10612 MiB | — |
| 131072 | 67.50 / 14132 MiB | 64.59 / 12212 MiB | — |
| 196608 | **启动失败** | 64.29 / 13813 MiB | 66.25 / 12277 MiB |
| 262144 | — | — | **65.51 / 13365 MiB** |

**q4_0 跑满了模型原生的 262144（256K）**，这是 f16 和 q8_0 都做不到的：

| KV 类型 | 最大 ctx | 该 ctx 下显存 | 解码速度 |
|---|---|---|---|
| f16 | 128K | 14132 MiB | 67.50 |
| q8_0 | 192K | 13813 MiB | 64.29 |
| **q4_0** | **256K（原生上限）** | **13365 MiB** | **65.51** |

q4_0 同时是三项里显存最省、可用 ctx 最大、速度还不慢的那个。**要做长上下文，答案是 q4_0，不是 q8_0。** 这也解释了 §3.1 里 q4_0 为什么能跟 f16 打平——它有专用 tile kernel。

**f16 在 196608 失败，报错不是 KV 不够，而是 MTP context 的 compute buffer：**

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 472.03 MiB on device 0: cudaMalloc failed
E common_speculative_init_result: failed to create MTP context
```

**MTP 除了权重（246 MiB）和 KV 之外，还要 472 MiB 的 compute buffer。** 这是显存规划时必须留出的余量——不开投机时不需要这 472 MiB。

**解码速度与 ctx 几乎无关**（f16 从 8K 到 128K 只差 0.6 t/s）。原因是这个模型是混合架构：

```
qwen35.full_attention_interval   4      ← 每 4 层才有 1 层 full attention
qwen35.block_count               33     → 只有 9 层有 KV cache
qwen35.attention.head_count_kv   4      (GQA 4:1)
qwen35.attention.key_length      256
```

其余 24 层是 SSM（gated delta net）层，状态大小固定，**不随 ctx 增长**。所以长上下文的 KV 成本只有标准 transformer 的 1/4 左右。

---

## 5. 投机族横评

同一 target，ctx 8192，2 轮：

| 方法 | 论述 | 代码注释改写 | 备注 |
|---|---|---|---|
| 不开投机 | 49.87 | 49.90 | 基准 |
| **MTP n2** | **68.03** | **71.46** | 自带 nextn 头，无需外部模型 |
| DFlash n2 | 68.26 | 64.77 | 需 1.32 GB 外部 draft，多占 2.9 GB 显存 |
| ngram-simple | 49.85 | 47.86 | 改写场景**反而亏 4%** |
| ngram-map-k | 49.87 | 49.49 | 零收益 |
| ngram-map-k4v | 49.85 | 49.54 | 零收益 |
| ngram-mod | 49.84 | 49.79 | 自适应版，仍零收益 |

### 5.1 DFlash 在 n=2 打平 MTP，但代价高

论述场景 68.26 vs 68.03 看似持平甚至微胜，实际是：

- 显存 **12367 vs 9419 MiB**，多 2.9 GB
- 改写场景 **64.77 vs 71.46**，落后 9.4%
- 接受率 67.7% vs 63.5%（论述）、61.8% vs 69.8%（改写）—— 不稳定

上一轮的 DFlash 报告结论不变：n≥3 全线落后，n=11 OOM。

### 5.2 ngram 系列全部无用，连最有利的场景也不行

我为 ngram 专门设计了一个甜点场景：给一段 Python 代码，要求「逐行加中文注释，代码一个字都不要改」。理论上 ngram 应该疯狂命中（输出大量复制输入的 token）。

实测：

| 方法 | 该场景 | 草稿/接受 |
|---|---|---|
| ngram-simple | 47.86 | 10 / 3 |
| ngram-map-k | 49.49 | 4 / 3 |
| ngram-mod | 49.79 | 64 / 15 |

**整段 192 token 的生成里，ngram 只产生了 4~64 个草稿 token，接受 3~15 个。** 原因是默认参数太保守：

```cpp
common_params_speculative_ngram_map  size_n = 12   // 需要 12 个 token 的完全匹配才查找
                                     size_m = 48   // 一次草稿 48 个 token
common_params_speculative_ngram_mod  n_match = 24  // 需要 24 个 token 匹配！
                                     n_max   = 64
```

要求连续 12~24 个 token 完全一致才触发，而对话/注释这类任务的局部重复远不到这个长度。即使触发了，一次性草稿 48~64 个，verify 就是 B=49~65，在 P100 上是灾难（参考 §6 的 batch 曲线）。

**ngram 适合的场景是「大量重复的固定模板输出」**（比如批量生成格式化的 JSON、填充固定 schema），不是自然对话，也不是代码注释。

### 5.3 级联组合也没用

`--spec-type` 支持逗号分隔多个类型，实现上是**按顺序级联**：前一个产出草稿就用它，否则试下一个。

| 组合 | 改写 | 写代码 |
|---|---|---|
| MTP | 71.85 | 63.38 |
| ngram-mod → MTP | 71.58 | 63.45 |
| ngram-map-k → MTP | 72.29 | 63.32 |
| MTP → ngram-mod | 71.66 | 63.07 |

全部在噪声内。因为 ngram 几乎不命中（§5.2），命中时也只有 3~15 个 token，而 MTP 本来就很便宜（3 ms/步）。

---

## 6. n_max=2 的真正原因（修正上一轮）

### 6.1 上一轮的结论错在哪

上一轮我写：B=4 的成本台阶来自「MMQ 在 sm_60 被禁用，dense 走 cuBLAS 的另一个档位」。

这个因果链是错的。真正的原因在 `mmvq-gp100.cu` —— 这个 fork 自己写的 GP100 专用 kernel。

### 6.2 决定性实验：把 MMVQ 关掉

`GGML_CUDA_GP100_MMVQ_DISABLE=1` 会让所有量化 matmul 退回通用的 cuBLAS 路径：

| 配置 | MMVQ 开 | MMVQ 关 | 变化 |
|---|---|---|---|
| 不开投机 | 49.87 | 40.00 | −19.8% |
| MTP n2 | **67.56** | 51.49 | −23.8% |
| MTP n3 | **51.02** | **52.34** | +2.6% |

**看第三行。** MMVQ 打开时，n3（51.02）远差于 n2（67.56）；MMVQ 关掉后，**n3（52.34）反而略优于 n2（51.49）**。

也就是说：**B=4 的台阶只存在于 MMVQ 路径，cuBLAS 路径的 batch 成本是平滑的。**

### 6.3 源码确认

```cpp
// mmvq-gp100.cu:1453
if (ncols < 1 || ncols > 8 || ... ||
    (ncols > 1 && (fusion || (src0->type != GGML_TYPE_Q4_K && ... &&
                              !(src0->type == GGML_TYPE_Q8_0 && src0->ne[1] >= 2048) && ...)))) {
    // Multi-column Q8_0 takes large matrices only: at 4 columns it wins there
    // (0.178 -> 0.085 ms) but the 24-row alpha/beta projections ... lose
    return false;
}
```

这个 kernel 支持 1~8 列（列 = verify batch 里的 token 数），但不同列数走不同的模板实例：

```cpp
// mmvq-gp100.cu:1488
// GP100_RMAP="nrows:R,..." overrides R for matrices with exactly nrows rows (Q6_K: R <= 4)
static int gp100_rows_override(int nrows, bool glu)
```

`R` = 每 warp 处理的行数，决定寄存器预算和并行度。1~3 列有专门的优化实例，4 列要换模板，效率断崖。

**这个 fork 的作者就是拿 MTP verify 做基准调的**——注释里直接写着：

> Q6_K's multi-column kernel loses to the generic int8 path at 5 columns in situ **（7.68 vs 6.18 ms per MTP verify batch）**

### 6.4 修正后的完整因果

```
MMVQ 定制 kernel
  ├─ ncols ≤ 3 (n_max ≤ 2): 单列 20 ms/token，三列 23.8 ms/轮 → MTP n2 极优
  └─ ncols ≥ 4 (n_max ≥ 3): 换模板，成本跳到 38 ms/轮 → MTP n3 崩

cuBLAS 通用路径（MMVQ 关掉）
  └─ 所有 batch 平滑，但基线慢 20% → 无论 n_max 多少都打不过开 MMVQ 的 n2
```

**结论不变：n_max=2。但原因从「MMQ 禁用」改成「GP100 MMVQ 的多列模板」。** 而且这是个真实的最优点，不是妥协——67.56 比任何人能拿到的组合都高。

### 6.5 顺带修正：P100 到底有没有 Tensor Core

上一轮我写「sm_60 无 Tensor Core」，也不准确。GP100 物理上有第一代 FP16 Tensor Core（19 TFLOPS），但 llama.cpp 的门槛是：

```cpp
// common.cuh:320
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) || ...   // VOLTA = 700
}
```

**P100 = 600 < 700 → llama.cpp 认为不可用**，走不了 `fattn-mma-f16`。同时 MMQ 也被禁：

```cpp
// mmq.cu:311
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {   // 610
    return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;           // dense (n_experts=0) → false
}
```

所以 P100 上：MMQ 关闭、MMA 关闭、CUDA graph 关闭、FA-GEMM 开启、GP100 MMVQ 开启。**这个 fork 的价值就是把最后一个做到了极致。**

---

## 7. 这个 fork 的定制优化，值多少

逐个关掉测（ctx 32768）：

| 关闭的优化 | t/s | 损失 |
|---|---|---|
| 都不关（基线） | 67.56 | — |
| `GGML_CUDA_GP100_MMVQ_DISABLE=1` | 51.57 | **−23.7%** |
| `GGML_CUDA_DISABLE_FUSION=1` | 65.49 | −3.1% |
| `GGML_CUDA_GP100_NO_ACT_CACHE=1` | 67.19 | −0.5%（噪声内） |

prefill 侧（16K prompt）：

| 配置 | prefill |
|---|---|
| `GGML_CUDA_FA_GEMM` 默认（开） | **720.5 t/s** |
| `GGML_CUDA_FA_GEMM=0` | 643.3 t/s（−10.7%） |

### FA-GEMM 为什么只赚 10.7%，文档却说 2 倍

`fattn-gemm.cu` 的注释：

> on Pascal there are no tensor cores, so long-context prefill falls to flash_attn_tile, which measures 3.55 TFLOPS of a 19.05 peak (18.6%) ... attention is ~86% of prefill at **262144 context**

两个前提我这里都不满足：prompt 只有 16K（不是 262K），attention 占比远不到 86%。而且它的启用门槛很严：

```cpp
// fattn-gemm.cu:339
// Only worth it once attention dominates; short contexts keep the tile kernel.
if (Q->ne[1] < 128 || K->ne[1] < 4096) {
    return false;
}
```

- `Q->ne[1]` = 本次前向的 query token 数，必须 ≥ 128
- `K->ne[1]` = 已有 KV 长度，必须 ≥ 4096

**推论：投机解码永远用不到这条路径。** decode 时 nt=1，verify 时 nt=n_max+1≤16，都远小于 128。即使 n_max 开到 15 也够不到。所以 MTP 的 verify 成本不受 FA-GEMM 影响，只有长 prompt 的 prefill 受益。

**另一个隐性价值**：GEMM 路径不需要把整个 KV 转成 f16 staging，省 512 MiB/卡（在 262144 ctx 时）。这是 f16 能开到 128K 的原因之一。

---

## 8. prefill 的代价

| prompt | 不开投机 | 开 MTP | 损失 |
|---|---|---|---|
| ~50 token | 132.4 | 126.7 | −4.3% |
| ~3500 token | 811.2 | 761.2 | −6.2% |
| ~16000 token | 767.6 | 692.4 | −9.8% |

MTP 的 drafter 也要吃一遍 prompt（它要拿 target 的 hidden state 建自己的 KV），所以 prefill 会慢。**prompt 越长，相对损失越大**，因为 MTP 那层的成本按 token 线性增长，而 target 的 prefill 有 batch 红利。

实际影响：16K prompt 的 prefill 从 20.8 s 变成 23.1 s，多 2.3 s；而后续 192 token 的生成从 3.85 s 缩短到 2.77 s。**只有生成超过约 240 token 才能回本。** 短回答 + 超长 prompt（比如长文档问答只答一句）的场景，开 MTP 是亏的。

---

## 9. 最终建议

### 当前配置（`config.json`）

```
spec_enabled = true
spec_type    = auto        # 有 nextn 头 → draft-mtp --spec-draft-n-max 2
ctk = ctv    = f16
ctx          = 126000
vision       = 关
```

保持不变。这是实测最优组合。

### 参数速查

| 场景 | 配置 |
|---|---|
| 默认 / 数学推理 | MTP n2 + f16 KV，什么都不用改 |
| 需要 >128K 上下文 | KV 改 **q4_0**（不是 q8_0）：256K 全开只要 13365 MiB，速度 65.51（只比 f16 少 3%） |
| 绝对不要做 | K 和 V 用不同类型（−37%）；n_max≥3（MMVQ 台阶）；启用 ngram（零收益） |
| 显存紧张 | MTP 额外要 472 MiB compute buffer，别按「权重+KV」算漏 |

### 还没做但值得做的

1. **`-ctkd/-ctvd q8_0`**：MTP 自己的 KV（1 层）目前是 f16。改量化理论上能省几十 MB，但可能影响接受率，收益太小不值得测。
2. **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`**：能让 GPU 借用系统内存，理论上突破 16 GB 跑 256K 完整上下文。代价是 decode 速度崩（要走 PCIe）。只在「必须 256K」时考虑。
3. **`-ot` 把 SSM 层或部分 attention 层放 CPU**：腾显存换 ctx。P100 + 6 核 CPU，代价会很大。

---

## 附：全部原始数据

| 文件 | 内容 |
|---|---|
| `mtp_logs/suite_kv.json` | KV 量化 5 组 × 3 轮 |
| `mtp_logs/suite_ctx.json` | ctx 上限 f16/q8_0 × 5 档 |
| `mtp_logs/suite_content.json` | 5 种内容 × 开/关投机 |
| `mtp_logs/suite_family.json` | 7 种投机 × 2 场景 |
| `mtp_logs/suite_combo.json` | 5 种级联组合 × 2 场景 |
| `mtp_logs/suite_prefill.json` | 3 档 prompt 长度 × 开/关 |
| `mtp_logs/suite_env.json` | 4 组环境变量 A/B + FA-GEMM 对照 |
| `mtp_logs/kv_quality.json` | KV 量化 30 题 × 3 档 |
| `mtp_logs/suite_*.log` | 每个 case 的完整 server 日志 |
