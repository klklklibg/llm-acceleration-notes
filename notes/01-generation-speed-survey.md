# 提高 token 生成速度：技术盘点 + P100 实测

> 2026-11-07 深夜。目标模型 DeepSeek-V4-Pro-Qwen3.5-9B-MTP，Tesla P100 PCIe 16GB（sm_60），
> llama.cpp 定制分支 `D:\llama.cpp`。全部数据为本机实测，脚本在 `D:\llamacpp-launcher`。

## 0. 一句话结论

**Q8_0 的解码已经吃满 95% 的显存带宽，投机解码只是补 launch 开销；真正还没动过的杠杆是权重位宽。**

换成同仓库的 **Q4_K_M** 后：MTP n2 从 78.0 → **87.2 t/s（数学）/ 67.4 → 77.3（论述）**，
30 题准确率 **完全一致（27/30，连分科都一样）**，显存省 3.4 GB。

## 1. 先算带宽账：为什么优化空间比想象小

解码一个 token 必须把全部权重从 HBM 读一遍。于是：

| 权重 | 体积 | 基线解码 | 每 token 读取带宽 | 占 P100 峰值(480 GB/s) |
|---|---|---|---|---|
| Q8_0 | 9.11 GB | 49.87 t/s | **454 GB/s** | **95%** |
| Q4_K_M | 5.38 GB | 60.00 t/s | 323 GB/s | 67% |

**Q8_0 在基线下已经贴着硬件极限**，所以任何"不减少权重量"的优化（更聪明的调度、
更好的采样、更多线程）都不可能有大收益——这解释了为什么前面一轮里
`-ub/-t/op-offload` 全在噪声内。

而 Q4_K_M 只做到 67%，说明它撞的不是带宽墙，而是 **K-quant 的反量化计算**。

## 2. 技术全景：哪些能用，哪些不能用

| 技术 | 本机可用 | 结论 |
|---|---|---|
| **权重量化 Q4_K_M** | ✅ | **+11.8%，质量无损（本文新增，最大收益）** |
| MTP（模型自带 nextn 头） | ✅ | +30~59%，n_max=2 最优 |
| KV cache 量化 q4_0 | ✅ | 速度持平、显存 −50%、质量无损 |
| **EAGLE3** | ✅（本文新增） | 头需与 target 完全同源；本例不匹配 → 输给 MTP 20% |
| DFlash / DSpark | ⚠️ | DFlash 已测更差；DSpark 无 9B 草案（只有 0.8B~4B） |
| ngram / prompt-lookup 5 种 | ✅ | 对话/代码改写全部零命中 |
| 后端采样 `-bs` | ⚠️ | +3.7%，但**随机 CUDA illegal memory access** → 不可用 |
| Medusa | ❌ | llama.cpp 无实现，且需要重训解码头 |
| Lookahead decoding | ❌ | llama.cpp 无实现；本身也只有 1.5~2×，且吃计算 |
| CUDA graph | ❌ | `ggml-cuda.cu:4840` 对 sm_60 硬禁用，需重编译 |
| vLLM / SGLang / TensorRT-LLM | ❌ | 要求 CC ≥ 7.5（P100 = 6.0），装不上 |
| ExLlamaV2 / Marlin / AWQ / 2:4 稀疏 | ❌ | 均要求 Turing / Ampere 以上 |

一句话：**这张卡上能动的只有 llama.cpp 里那几项，其中最大的还没被用过的是权重位宽。**

## 3. 新实测数据

测试条件：ctx 8192、temp 0.7、seed 1234、192 token、2~3 轮取中位；
同配置重复轮次重合在 ±0.5 t/s 内，故 >1 t/s 的差异可信。

### 3.1 权重位宽矩阵

| 权重 | 体积 | 基线 math | MTP n2 math | MTP n2 essay | MTP n3 math | 显存@8K |
|---|---|---|---|---|---|---|
| Q8_0 | 9.11 GB | 49.87 | 78.04 (82.6%) | 67.36 (63.5%) | 51.02 | 9419 MiB |
| **Q4_K_M** | **5.38 GB** | **60.00** | **87.23 (74.5%)** | **77.34 (59.8%)** | 84.37 (71.0%) | **6062 MiB** |

三个值得注意的点：

1. **基线上 Q4 快 20%（49.9→60.0），但开了 MTP 只快 11.8%。** 因为 MTP 把
   "每 token 读一遍权重"摊成"每轮读一遍、产出 2.2 个 token"，权重占比下降，
   省下来的带宽红利被摊薄。
2. **Q4 的接受率反而更低**（74.5% vs 82.6%）。量化同时扰动了 target 和 MTP 头的分布，
   两者的一致性变差。但这不影响结论——仍快 11.8%。
3. **Q4 下 n3 的惩罚从 −35% 缩到 −3%**（84.37 vs 87.23）。Q4_K 走的是和 Q8_0 不同的
   MMVQ 模板，B=4 台阶变浅。但 n2 仍是最优。

### 3.2 EAGLE3：机制很强，但需要"同源"的头

从 `yuyijiong/Qwen3.5-9B-Eagle3`（apache-2.0，0.88 GB）转换：
`LlamaForCausalLM EAGLE3`，1 层、hidden 4096、draft_vocab 50000 带 d2t 映射，
`target_layers=[2,16,29]`，用 `--target-model-dir` 指向 Qwen3.5-9B 的 config/tokenizer 即可。
转换与运行都成功（`eagle3.target_layers` 等元数据齐全，14 个张量）。

| 配置 | math | essay | 接受率 |
|---|---|---|---|
| 不开投机 | 60.00 | 60.03 | — |
| EAGLE3 n1 | 69.33 | 69.47 | 57.9% / 58.3% |
| EAGLE3 n2 | 67.57 | 64.34 | 46.7% / 41.8% |
| EAGLE3 n3 | 55.91 | 51.77 | 36.9% / 31.7% |
| EAGLE3 n6 | 34.36 | 27.65 | 24.0% / 16.0% |
| EAGLE3 n6 + p_min 0.5 | 62.50 | 53.59 | 67.6% / 49.1% |
| **MTP n2** | **87.23** | **77.34** | **74.5% / 59.8%** |

**原因**：这个头是在 Qwen3.5-9B **base** 上训练的，而 target 是 Jackrong 的
DeepSeek-V4-Pro 微调版——权重不同，隐状态分布就不同。EAGLE3 吃的是 target 的
第 2/16/29 层隐状态，一旦 target 换了，特征映射就失配，pos1 接受率从该有的 85% 掉到 57.9%。

另外 EAGLE3 多占 2.9 GB 显存（8556 vs 6062 MiB），因为它是一个独立 draft 模型，
不像 MTP 头就长在 target GGUF 里。

**结论**：EAGLE3 不是"更强的技术"，而是"更强的技术 + 必须配对的头"。
换模型时如果官方发布同源 Eagle3 头，值得重测；当前不值得用。

### 3.3 后端采样 `-bs`：+3.7%，但会崩

```
math：79.51 → 82.47 t/s（两轮重合）
```

但两处崩溃，都可复现：

| 组合 | 结果 |
|---|---|
| `-bs` 单独，math prompt | 正常 |
| `-bs` 单独，code prompt | **CUDA illegal memory access** |
| `-bs` + `--spec-draft-p-min` | **CUDA illegal memory access** |

`ggml-cuda.cu:108` 直接报非法访存，服务进程死掉。这是实验性路径的 bug，
**不要开**。相对收益 3.7%，风险是整轮生成失败。

### 3.4 p_min：能用，但救不回 n_max

MTP 不支持"低置信提前截断"的收益——n3 配 p_min=0.5 把接受率从 51% 拉到 82.8%
（Q4 下），但速度 83.56 仍低于 n2 的 87.23。原因还是同一处：B=4 的 MMVQ 模板台阶
是**单轮成本**问题，截断只能让它少触发几次，不能消除。

### 3.5 显存与上下文（Q4_K_M 的红利）

| 权重 | ctx | KV | 显存 | 速度（论述） |
|---|---|---|---|---|
| Q8_0 | 8192 | f16 | 9419 MiB | 67.4 |
| Q8_0 | 126976 | f16 | 14308 MiB | 68-71 |
| Q4_K_M | 8192 | f16 | 6062 MiB | 77.3 |
| Q4_K_M | 65536 | f16 | 8223 MiB | 79.6 |
| Q4_K_M | 131072 | f16 | 10783 MiB | 79.4 |
| Q4_K_M | 196608 | f16 | 13344 MiB | 78.5 |
| Q4_K_M | 262144 | f16 | 启动后请求即失败（15904 MiB 已贴顶） | — |

**Q4_K_M 让 16 GB 卡在 f16 KV 下就能开 196K 上下文**，比 Q8_0 的 128K 翻半，
而且仍保留 2.5 GB 余量给系统/其它进程。

### 3.6 质量验证

同一批 30 题（数学/物理/化学各 10）、贪心解码、MTP n2：

| 权重 | 准确率 | 分科 | 每题 token |
|---|---|---|---|
| Q8_0 | 90.0%（27/30） | 100 / 100 / 70 | 159 |
| Q4_K_M | **90.0%（27/30）** | **100 / 100 / 70** | 176 |

**错的三道题完全相同**，都是判分规则或题目歧义（`N_A` 写法、22.4 L 与 24.5 L、`sp³` 上标）。
Q4_K_M 每题多用约 17 个 token（思考略啰嗦），但正确数与分布一致。

> 诚实边界：30 题是小样本，Q4_K_M 是有损量化。它在这套题库上看不出差异，
> 不代表所有任务都无损。若要绝对保守，留在 Q8_0 只损失 11.8% 速度。

## 4. 为什么别的推理框架帮不上

| 框架 | 门槛 | P100 |
|---|---|---|
| vLLM | CC ≥ 7.5（官方文档） | ❌ 6.0 |
| SGLang | 同 vLLM | ❌ |
| TensorRT-LLM | Ampere 及以上 | ❌ |
| ExLlamaV2 | Turing 及以上 | ❌ |
| MLC-LLM | 支持 Pascal | ⚠️ 能跑，但无 Pascal 专用 kernel，通常更慢 |

也就是说在 P100 上 **llama.cpp 是唯一能打的运行时**，已启用/未启用的硬件特性：

```
MMQ       关（sm_60 < DP4A 的 sm_61）
FP16 MMA  关（common.cuh:320 要求 cc≥700）
CUDA graph 关（ggml-cuda.cu:4840 要求 cc≥700）
FA        开
FA-GEMM   开（只对 prefill 的 Q->ne[1]≥128 生效，decode/verify 够不到）
GP100 MMVQ 开（价值 23.7%）
```

## 5. 最终配置

```
-m DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf
-ngl 99 -c <ctx≤196608> -b 2048 -ub 1024 -t 6 -fa on -lm none
--jinja --metrics -np 1 --no-mmproj
--spec-type draft-mtp --spec-draft-n-max 2
```

**不要加**：`-bs`（会崩）、`--spec-draft-p-min`（配 -bs 必崩，单独用也不如 n2）、
`-ctk q8_0`（比 f16 慢 3%）、K/V 混类型（−37%）。

## 6. 还剩什么

唯一剩下的大杠杆是 **CUDA graph**。Q4_K_M 下每轮 25.9 ms 里，
两次 draft 各 3.0 ms、其中约 2.4 ms 是 kernel launch 空转。打开以后
draft 可降到 0.8 ms/步，估算 **+20%（87 → 105 t/s）**。
代价是改 `ggml-cuda.cu:4840` 重编译 sm_60 CUDA 单元 20~40 分钟，
且上游当年禁用大概率有实测依据（Pascal 上 graph 捕获可能不稳）。

另一条不动软件的路：**关 ECC**（`nvidia-smi -e 0`，需重启），对带宽密集的解码约 +10%。

## 7. 复现

```bash
python bench_quant.py Q4_K_M            # 量化矩阵
python bench_quant_quality.py Q4_K_M Q8_0   # 质量对比
python bench_eagle3.py                  # EAGLE3 全 n_max 扫描
python bench_suite.py kv/content/prefill/family/combo/env/ctx
```

原始数据：`mtp_logs/bench_quant.json`、`eagle3_vs_mtp.json`、`quant_quality.json`。
EAGLE3 转换产物：`D:\lmstudio\eagle3\Qwen3.5-9B-Eagle3\eagle3-F16.gguf`（947 MB）。
