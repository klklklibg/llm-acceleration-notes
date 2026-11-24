# MTP（Multi-Token Prediction）技术研究

实测环境：Tesla P100 PCIe 16GB（sm_60, ECC on）、`D:\llama.cpp\bin`（0.3.0-dev, commit 4423f2a）、
模型 `DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q8_0`（9.11 GB, qwen35 架构, 32 主干层 + 1 个 nextn 层）。
所有数字为本次实测，非引用。

---

## 0. 结论先行

| 问题 | 答案 | 证据强度 |
|---|---|---|
| P100 上 MTP 值不值得开 | **值得，但只在 dense 量化（Q8_0）上** | 实测 +32.6% |
| n_max 该设多少 | **2**，不是官方默认的 3 | 实测，n=3 反而 -5.6% |
| 上限由什么决定 | **verify 阶段 B=4 的成本台阶**，不是接受率 | 源码 + 实测交叉验证 |
| 接受率的天花板 | pos1 0.68–0.88，取决于内容，不取决于温度 | 实测 4 组 |
| 剩余优化空间 | 几乎全在 CUDA graph（P100 被硬禁用） | 源码级 |

---

## 1. MTP 是什么

### 1.1 训练目标

在主模型最后一层隐藏态 `h_i` 之上，追加 D 个串行的预测模块。第 k 个模块：

```
h'_k = TransformerBlock_k( eh_proj_k( [ RMSNorm(emb(x_{i+k})), RMSNorm(h'_{k-1}) ] ) )
p_k   = shared_head( h'_k )        # 预测 x_{i+k+1}
```

- 输入是**上一个深度的表示** + **第 i+k 个 token 的真实 embedding**（教师强制）
- 输出头与主模型共享（`shared_head_head` 复用 `output.weight`），embedding 也共享
- 损失：`L = L_main + λ · mean_k(CE_k)`，DeepSeek-V3 用 λ=0.3（前 10T token）/ 0.1（后 4.8T）
- 关键点：**MTP 不只是推理加速手段，它首先是训练目标的密集化**。每个隐藏态被迫同时支撑 1 步和 k 步的预测，等价于让模型"提前规划"。DeepSeek-V3 的消融显示这本身能提升主模型分数（小 MoE 基线 39.0 → 41.4 BBH）。

### 1.2 推理时的两条路

1. **丢弃**：MTP 模块不参与前向，成本为零，得到一个普通模型。
2. **当 drafter**：用 MTP 模块做自投机（self-speculative）。因为 draft 与 target 共享 embedding/输出头/词表，**不需要单独训练 drafter，也不需要 `-md` 指定草稿模型**。这是 MTP 相对 EAGLE/Medusa 的核心工程优势。

DeepSeek-V3 官方数据（D=1，即只多预测 1 个 token）：第 2 个 token 接受率 **85–90%**，端到端 **1.8x TPS**。

### 1.3 与其它投机方案的定位

| 方案 | drafter 来源 | 额外训练 | 显存开销 | 适用前提 |
|---|---|---|---|---|
| draft-model | 独立小模型 | 不需要 | 一整个模型 | 有同词表的快模型 |
| ngram / lookup | 上下文自身 | 不需要 | 0 | 输出高度重复（改代码、长文档回抄） |
| EAGLE-3 | 外挂草稿头 | **要**，几小时单机 | 小 | 通用，当前生产主流 |
| **MTP** | 模型自带 nextn 层 | **要**，但训练时顺带完成 | 1 层权重（本模型 258 MB） | 模型本身带 nextn 层 |

MTP 的独特性：它是**唯一一个在预训练阶段就内建、推理时零额外训练成本**的 drafter。

---

## 2. 实现解剖（本 fork，非上游 llama.cpp）

本仓库对 MTP 的支持远比上游完整，有专用代码路径：

| 位置 | 作用 |
|---|---|
| `src/llama-context.cpp:30` | `LLAMA_CONTEXT_TYPE_MTP` → `LLM_GRAPH_TYPE_DECODER_MTP` |
| `src/models/qwen35.cpp:489` | `graph_mtp` 构造器，MTP 前向图 |
| `src/llama-model.cpp:2496-2620` | MTP context 用**独立 KV cache**，只持 nextn 层 |
| `common/speculative.cpp:1348` | `common_speculative_impl_draft_mtp`，投机驱动 |
| `src/llama-ext.h:92-99` | `llama_set_embeddings_nextn` / `llama_set_nextn_layer_offset` |

### 2.1 MTP 前向图（`qwen35.cpp`）

```cpp
h_norm   = norm(h_embd,      layer.nextn.hnorm)     // 上一深度隐藏态
e_norm   = norm(tok_embd,    layer.nextn.enorm)     // 当前 token embedding
concat   = ggml_concat(e_norm, h_norm, dim=0)       // [2*n_embd]
cur      = eh_proj @ concat                          // [n_embd]
... 一个完整 decoder block: attn(Q/K/V/norm/rope) + FFN(gate/up/down) ...
logits   = shared_head_head @ norm(cur, shared_head_norm)
```

注意三点：

1. **MTP 层是完整的 decoder block**，不是线性头。它有 Q/K/V、有 FFN、**有自己的 KV cache**。
2. 本模型的 `blk.32` 结构与 `blk.31` 完全同构（attn_q 35.65 MB、ffn_* 各 53.5 MB），只多一个 `eh_proj`（35.65 MB）。总重 **258.5 MB**（Q8_0）。
3. 之前记录的"投机 +330 MiB 显存"主要就是这层权重（246 MiB）+ 它自己的 KV，**不是上下文 KV 膨胀**。

### 2.2 三种驱动模式（同一份代码，按模型分派）

```cpp
// speculative.cpp:1378-1385
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;   // gemma4: 共享 target KV
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;        // step35: K 个 head 链式
// 都不是 → qwen35/qwen35moe: 单个 head，自回归外推
```

| 模式 | 代表模型 | n_max 上限 | 行为 |
|---|---|---|---|
| `is_mem_shared` | Gemma4 assistant | 不限 | 所有 draft token 用**同一位置**，共享 target 的 KV |
| `chain_heads` | Step3.5 | **= n_mtp_layers**（代码强制 clamp） | 第 i 个 token 用第 i 个 head，每步重建自己层的 KV |
| 单 head 外推 | **Qwen3.5 / 本模型** | 代码不限 | 同一 head 反复喂自己的输出 |

**这是理解本模型行为的关键**：`n_mtp_layers == 1`，所以 n_max>1 的每一步都是纯外推——
训练时该 head 只见过去 1 步，第 2 步之后是分布外。Step3.5 有 K 个 head 时，代码直接把
`n_max` clamp 到 K，避免外推；Qwen3.5 没这个保护。

```cpp
// speculative.cpp:1387-1391  chain_heads 才有上限
if (chain_heads) {
    this->params.n_max = std::min(this->params.n_max, n_mtp_layers);
}
```

外推的递推式（`speculative.cpp:1740-1743`）：

```cpp
common_batch_add(batch, id, dp.n_past + i + 1, {seq_id}, true);
std::memcpy(batch.embd + ..., h_row, row_bytes);   // 上一步 MTP 输出的 h 当下一步输入
```

### 2.3 一个默认值陷阱

```cpp
// common/common.h:326-330
int32_t n_max = 3;      // 官方默认 3
int32_t n_min = 0;
float   p_min = 0.0f;   // 默认不提前停止
```

`p_min = 0` 意味着"只收集高置信度 draft token"这道闸门**默认完全打开**：即使第 4、5、6 个
draft 的 top-1 概率只有 5%，也照送 verify。下面的实测会显示这有多贵。

---

## 3. 实测：n_max 扫描

固定 prompt（中文论述，256 token，`--ignore-eos`，temp 0.7，ctx 2048，`-np 1`，`-fa on`，`-lm none`）。
每次单独起进程，无缓存干扰。

| tag | tg t/s | 相对基线 | mean acc len | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 |
|---|---|---|---|---|---|---|---|---|---|
| baseline（无 MTP） | 49.64 | 1.000 | — | — | — | — | — | — | — |
| n_max=1 | 63.47 | **+27.9%** | 1.70 | 0.700 | | | | | |
| **n_max=2** | **65.80** | **+32.6%** | 2.17 | 0.684 | 0.487 | | | | |
| n_max=3 | 46.86 | −5.6% | 2.35 | 0.694 | 0.426 | 0.231 | | | |
| n_max=4 | 43.74 | −11.9% | 2.73 | 0.720 | 0.484 | 0.290 | 0.237 | | |
| n_max=6 | 31.35 | −36.8% | 2.71 | 0.606 | 0.394 | 0.287 | 0.191 | 0.138 | 0.096 |

接受率曲线是干净的双段结构：**pos1 ≈ 0.68–0.72，pos2 直接掉到 0.43–0.49，之后每步再腰斩**。
pos1 与 DeepSeek 报告的 85–90% 有差距（原因见 §6），但 pos2 的崩塌是单 head 外推的必然结果。

### 3.1 p_min 能不能救回来

`p_min` 让 draft 在低置信度处提前停止，从而缩短 verify batch：

| tag | tg t/s | 相对基线 | acc len | 实际 draft/轮 | draft 总耗时 |
|---|---|---|---|---|---|
| n6, p_min=0 | 31.35 | −36.8% | 2.71 | 5.93 | 1666 ms |
| n6, p_min=0.2 | 31.26 | −37.0% | 2.52 | 5.40 | 1677 ms |
| n6, p_min=0.4 | 50.61 | +2.0% | 2.90 | 3.28 | 1042 ms |
| n6, p_min=0.6 | 53.31 | **+7.4%** | 2.99 | 2.86 | 908 ms |
| n8, p_min=0.5 | 44.87 | −9.6% | 3.66 | 3.76 | 953 ms |

**p_min 确实有效**：0 → 0.6 把 n_max=6 从 −36.8% 拉回 +7.4%，acc len 反而更高（2.71→2.99）。
但它**打不过 n_max=2 的 +32.6%**——因为长 draft 的每一步都要付固定成本（见 §4）。
换句话说：在 P100 上，"短 draft 跑得快" 优于 "长 draft 早停"。

---

## 4. 为什么 n_max=2 是硬上限（成本模型）

### 4.1 draft 单步成本恒定 3.0 ms

从 `dur(b,g,a)` 的 `g`（draft 总耗时）除以"轮数 × n_max"：

| n_max | 总 draft ms | 轮数 | ms/步 |
|---|---|---|---|
| 1 | 476.2 | 150 | 3.17 |
| 2 | 710.7 | 117 | 3.04 |
| 3 | 967.3 | 108 | 2.99 |
| 4 | 1107.6 | 93 | 2.98 |
| 6 | 1666.1 | 94 | 2.95 |

**完全线性，恒定 ~3.0 ms/步**。

这 3.0 ms 里有多少是真实计算？MTP 层权重 258.5 MB，P100 实测带宽约 430 GB/s（ECC on），
读一遍 ≈ **0.60 ms**。也就是说 **2.4 ms（80%）是固定的 kernel launch / graph dispatch 开销**。

根因在源码：

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:4840
if (ggml_cuda_info().devices[...].cc < GGML_CUDA_CC_VOLTA) {   // VOLTA = 700
    graph->disable_due_to_gpu_arch = true;                      // P100 = 600 → 命中
}
```

P100（cc 600）**CUDA graph 被硬禁用**。每次 `llama_decode` 都要逐个 launch kernel，无法录制成图复用。
target 的 32 层前向（20 ms/step）里这点开销只占 12%，几乎无感；MTP 只有 1 层，开销占比直接 80%。

### 4.2 verify 的 B=4 台阶

用 `llama-batched-bench` 单独测 target 每步成本随 batch 的变化（B = 每步 token 数）：

| B | S_TG t/s | ms/step | ms/token |
|---|---|---|---|
| 1 | 50.06 | 19.98 | 19.98 |
| 2 | 96.73 | 20.68 | 10.34 |
| **3** | **126.20** | **23.77** | **7.92** |
| 4 | 104.48 | 38.28 | 9.57 |
| 5 | 106.80 | 46.82 | 9.36 |
| 6 | 105.62 | 56.80 | 9.47 |
| 7 | 106.50 | 65.73 | 9.39 |
| 8 | 115.76 | 69.11 | 8.64 |

**B=3 是峰值，B=4 每步成本 +61%（23.77 → 38.28 ms），吞吐反而从 126 掉到 104。**

从 MTP 实测反推的 verify 成本与这张表高度吻合（差 5–14%，来自 KV 长度差异），交叉验证成立：

| n_max | verify B | 反推 verify ms | batched-bench ms | 偏差 |
|---|---|---|---|---|
| 1 | 2 | 23.60 | 20.68 | +14% |
| 2 | 3 | 26.91 | 23.77 | +13% |
| 3 | 4 | 41.19 | 38.28 | +8% |
| 4 | 5 | 50.50 | 46.82 | +8% |
| 6 | 7 | 68.70 | 65.73 | +5% |

### 4.3 合并成每轮账本

| n_max | draft | verify | 轮总 | 产出 token | ms/token | t/s |
|---|---|---|---|---|---|---|
| 1 | 3.2 | 23.6 | 26.8 | 1.70 | 15.8 | 63.5 |
| **2** | **6.1** | **26.9** | **33.0** | **2.17** | **15.2** | **65.8** |
| 3 | 9.0 | **41.2** | 50.2 | 2.35 | 21.4 | 46.9 |
| 4 | 11.9 | 50.5 | 62.4 | 2.73 | 22.9 | 43.7 |
| 6 | 17.7 | 68.7 | 86.4 | 2.71 | 31.9 | 31.4 |

看清了：n_max 从 2 到 3，**产出只多 0.18 个 token，成本多 17.2 ms**。边际成本 95 ms/token，
荒谬。而 n_max=1→2 的边际是 6.2 ms 的 draft 换 0.47 个 token，值。

**n_max=2 不是调参运气，是"verify batch = 3 = P100 上 cuBLAS 的效率峰值"这件事的直接映射。**
B = n_max + 1，所以最优 n_max 必然是 2。

### 4.4 B=4 台阶的根因（分级）

- **确定（源码）**：P100 上 MMQ 被禁用，dense 矩阵乘法全部走 cuBLAS。

```cpp
// ggml/src/ggml-cuda/mmq.cu:311-316
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {   // P100: 600 < 610
    // for MoE, mmq is faster even without native dp4a
    return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;            // dense: n_experts=0 → false
}
```

- **确定（实测）**：B=4 处 ms/step 有 +61% 的硬台阶，且在 MTP 侧独立复现。
- **推断（未用 ncu 验证）**：台阶来自 cuBLAS 在 M=3 → M=4 之间切换 GEMV/窄矩阵核到通用 HGEMM tile，
  tile 利用率从满到 4/8。P100 有 FP16 Tensor Core（19 TFLOPS），B=3 时 126 t/s ≈ 2.52x 基线，
  这个倍率符合 TC 参与的特征。

### 4.5 这条模型解释了之前 27B ternary 的全线亏损

| 模型 | 量化路径 | batch 红利 | MTP 结果 |
|---|---|---|---|
| 9B Q8_0（本文） | cuBLAS FP16 TC，MMQ 禁用 | **B=3 达 2.52x** | **+32.6%** |
| 27B PTQ1_0 ternary | 定制 `mmvq-gp100.cu`，解包计算主导 | 无（100→261→478→716） | −70% |

同一套投机框架，一个赚一个亏，差别不在接受率（27B 的 pos1 也有 34%），而在
**verify 阶段有没有 batch 红利可供收割**。

---

## 5. MTP 的收益取决于内容，不取决于温度

同一 prompt 结构，三种内容，n_max=2，256 token：

| 内容类型 | tg t/s | 相对基线 | pos1 | pos2 | acc len |
|---|---|---|---|---|---|
| 数学推导 | **76.94** | **+55.0%** | 0.879 | 0.697 | 2.58 |
| 代码生成（Python） | 66.67 | +34.3% | 0.722 | 0.496 | 2.22 |
| 中文论述 | 65.80 | +32.6% | 0.684 | 0.487 | 2.17 |

温度（同一论述 prompt）：0.0 → 0.759 / 0.518，0.3 → 0.806 / 0.556，0.7 → 0.684 / 0.487。
**波动 ±0.06，在重复测量噪声内，温度不是主变量。**

结论：**结构越强、越可预测的输出，MTP 越赚**。数学推导的 pos1 到 0.88，已经贴到 DeepSeek
报告的下沿。早先 128 token 实测 pos1=0.863 与此一致，说明 §3 里 0.684 那组是"该段中文论述恰好
不好猜"，不是实现问题。

---

## 6. 为什么本模型 pos1 只有 0.68–0.88，而 DeepSeek 报 85–90%

1. **模型规模**。9B vs 671B（37B 激活）。MTP head 的表达能力随主干缩放，小模型的 nextn 头本身更弱。
2. **训练配比未知**。DeepSeek 用 λ=0.3 训了 10T token 的 MTP 目标；这个社区模型（Jackrong 合并版）
   的 nextn 层训练量无从验证——GGUF 里只有权重，没有训练元信息。
3. **单 head 的天然上限**。DeepSeek 的 85–90% 是"D=1 的第二个 token"，对应本模型的 **pos1**。
   本模型数学场景 pos1=0.879 已在区间内，中文论述 0.684 低于区间——差异是内容，不是架构。
4. **采样温度**。DeepSeek 报告未说明评测温度；greedy 下接受率天然更高。本文 temp=0 时 pos1=0.759。

值得注意：**pos2 的 0.49 是外推，不是训练目标**。DeepSeek 从未宣称第 3 个 token 的接受率。
本模型 pos2=0.49 已经是"外推仍有正收益"的运气，不是能力。

---

## 7. 实践结论

### 7.1 参数

| 参数 | 建议 | 理由 |
|---|---|---|
| `--spec-type` | `draft-mtp` | 模型自带 nextn 层，不需要 `-md` |
| `--spec-draft-n-max` | **2** | verify B=3 命中 cuBLAS 峰值；3 起亏 |
| `--spec-draft-p-min` | 0.4–0.6（仅在 n_max≥4 时有意义） | 但 n_max=2 时用不上，n_max 大时也救不回 |
| `-fa on` | 必须 | 之前实测 decode +4.9% |
| `-lm none` | 建议 | 省 6 GB 映射页，加载快一倍，速度不变 |
| `-np 1` | 必须显式 | 新版默认 auto 会切 4 slot |
| 视觉 | 纯文本时关掉 | `--mmproj-auto` 会挂 mmproj，decode −18% |

### 7.2 什么时候别开

- **非 dense 量化路径**（PTQ1_0 / ternary / 任何无 batch 红利的定制 kernel）——必亏。
- **输出高度重复的场景**：ngram 自投机更便宜（零 draft 成本），MTP 的 3 ms/步是纯浪费。
- **显存紧到临界**：多 330 MiB 意味着 16 GB 上双实例必 OOM。

### 7.3 剩余优化空间（按性价比排序）

1. **CUDA graph（唯一的大头）**。P100 被 `cc < 700` 硬禁。若能打开，draft 3.0 → ~0.8 ms/步：
   - n_max=2：33.0 → 26.5 ms/轮 → 82 t/s（**+24%**）
   - n_max=4：62.4 → 46.4 ms/轮 → 59 t/s（仍然不如 n_max=2）
   - 结论：即使开了 graph，最优仍是 n_max=2，但整体水位从 65.8 抬到 ~82。
   - 风险：要改 `ggml-cuda.cu:4840` 并重编译 sm_60 CUDA 单元（20–40 min），且 P100 的
     graph 捕获有已知问题（上游当年禁用大概率有实测依据）。**不建议在不动计算栈的前提下碰。**
2. **B=4 台阶**。除非让 n_max=3 变得划算，否则无收益。
3. **内容路由**。数学/代码场景收益 +34~55%，自由文本 +33%。如果按请求类型动态开关 MTP，
   能在低收益场景省下 330 MiB。

---

## 8. 复现

```bash
# n_max 扫描 + p_min 扫描（11 组，约 3 分钟）
python D:\llamacpp-launcher\bench_mtp_study.py

# target 的 batch 成本曲线
D:\llama.cpp\bin\llama-batched-bench.exe -m <model> -c 16384 -b 4096 -ub 1024 \
    -npp 0 -ntg 128 -npl 1,2,3,4,5,6,7,8 -fa on -lm none -ngl 99

# 单次带逐位置接受率（-lv 4 才会输出 SPC_TRC 统计）
D:\llama.cpp\bin\llama-cli.exe -m <model> -p "..." -n 256 -st --ignore-eos \
    --no-mmproj -c 2048 -np 1 -fa on -lm none -ngl 99 \
    --spec-type draft-mtp --spec-draft-n-max 2 --temp 0.3 -lv 4
```

原始数据：`D:\llamacpp-launcher\mtp_study.json`，完整日志：`D:\llamacpp-launcher\mtp_logs\*.log`
