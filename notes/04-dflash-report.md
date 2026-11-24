# DFlash 在 P100 上的实测：全面劣于 MTP

测量日期 2026-11-08　目标 P100 16GB (sm_60)，llama.cpp 0.3.0-dev (4423f2a)

## 社区对照：我的结论站得住，但有重要前提

先说结论：**我的数字与三份独立测试一致，但社区里确实存在 DFlash 大幅领先的报告——
差别不在硬件新老，而在 target 本身有多慢。**

### 印证我的三份

| 来源 | 硬件 | 结论 |
|---|---|---|
| 本报告 | P100 sm_60 16G | MTP 68.36 > DFlash 65.87；n≥4 全部跌破基线 |
| `thc1006/qwen3.6-speculative-decoding-v100` | 2×V100 sm_70 32G | DFlash 增速峰值在 **n≈2–4**，随 n 增大而侵蚀；**MTP > DFlash**（n4: MTP 1.60× vs DFlash 1.46×） |
| `sergiiob` Intel Arc Pro B70 | Arc B70 | 峰值 n=2（27.6 t/s），接受率 0.91→0.30；**n=8 启动即崩** |
| `asf0/Qwen3.8-27B-DFlash-GGUF` | 未注明 | **MTP 19.07 > DFlash 18.99**，作者自述"Not faster than DFlash2 or native MTP" |

V100 那篇还引了别人两句话，和我的观察完全同构：
> @jschmied "3–4 was the peak"，@HH1162 "n=16 no speedup"

Intel B70 的作者写的总结是 **"Small draft windows win"**——与我在 P100 上得到的
"n_max 越大越差、n11 直接 OOM" 是同一条曲线。

### 反例：为什么别人能拿到 2.69× / 4.44×

| 来源 | 硬件 | 数字 |
|---|---|---|
| PR #22105 自报 | DGX Spark | Qwen3.6-27B Q4_K_M：12.57 → 33.76 t/s，**2.69×** |
| 第三方复测 | RTX 6000 PRO | Qwen3.6-27B 36K ctx，**4.44× vs MTP** |

关键在 **baseline 有多慢**：

- DGX Spark 上 27B Q4 的基线只有 **12.57 t/s**（79.5 ms/token，LPDDR5X 带宽受限）
- 我的 P100 上 9B Q8_0 基线是 **49.89 t/s**（20 ms/token）——本来就快 4 倍

投机解码的收益是"每轮产出 token 数 ÷ 每轮成本"。target 越慢，draft 那点固定开销
占预算的比例越小，批次越容易摊薄；target 本身已经 50 t/s，投机的空间就被压没了。

V100 那篇把这条写成公式并验证了：

```
C(n) = τ / speedup ≈ d + v·n
固定项 d 两者近似相等，但边际成本 v 随 baseline 速度上升（dense 0.16 / MoE 0.26）
```

**所以判据不是"卡新不新"，而是"target 慢不慢"。** 我的配置恰好落在 DFlash 最不利的一侧。

### 我漏掉、补测后仍然不变的一项：p_min

社区主推 `--spec-draft-p-min 0.3`（asf0 的推荐值）。我第一轮完全没测，
因为怀疑它只对 DFlash2 的 selector 生效。查代码后确认**普通 DFlash 分支也会用它截断**：

```cpp
// speculative.cpp:1320
const llama_token id = cur_p->data[0].id;
if (cur_p->data[0].p < params.p_min) { break; }   // 置信度不足就砍掉尾部
```

补测（n_max=7）：

| p_min | t/s | draft/接受 | 接受率 |
|---|---|---|---|
| 0（原始） | 30.17 | 121/480 | 25.2% |
| 0.2 | 33.19 | 124/387 | 32.0% |
| 0.3（社区推荐） | 40.38 | 129/291 | 44.3% |
| 0.5 | 49.20 | 125/170 | 73.5% |

**p_min 确实把 n7 从 30.17 救到 49.20（+63%），但天花板正好是不开投机的 49.89。**
它做的事是把无效的长草稿砍短——等价于退回 n_max≈1~2，代价是每轮产出也变少。

MTP 也带了 p_min 对照：n4 + p0.3 = 50.23，与不加 p_min 的 49.96 无显著差异，
说明这个参数在 MTP 上也没能把 n4 救回 n2 的水平。

### 还有两件社区提到、我没覆盖的事

**1. 内容类型对接受率的影响远大于参数**

| 来源 | 内容 | 接受率 |
|---|---|---|
| Nemotron + RTX 5080 | 数学 / 逐步推理 | 55–63%，均长 5，**2.25×** |
| 同上 | 自由文本解释 | 19–26%，+14~25% |
| 同上 | 简短闲聊 | 7–17%，**慢 15%** |
| Ling-3.0-tiny | Python 代码 | 53%，+61% |
| Ling-3.0-tiny | 散文 | 28%，+8% |

我这轮用的是中文论述（自由文本），属于 DFlash 最不利的场景之一。
如果主要跑代码或数学，两头都会移动——但 P100 的成本劣势是结构性的，不会因此翻盘。

**2. 长 prompt 会让 prefill 明显变慢**

Nemotron 那篇实测：prompt processing 从 **8714 掉到 4841 tok/s**，因为 drafter 也要读 prompt。
12.6K prompt 的 TTFT 从 1.45 s 涨到 2.6 s。

我固定用短 prompt（22 token）测试，没暴露这一项。**在 126K 上下文场景下这是实打实的额外损失**，
本机没测，但方向上只会让 DFlash 更差。

**3. thinking 模式未测**

V100 那篇报告"thinking traces draft better than concise answer tokens"，
而 asf0 的基准是 thinking disabled。两者不矛盾（一个说可草稿性，一个说测试设置），
但对 DeepSeek-V4-Pro 这种 reasoning 模型值得知道。本轮未测。

## 我的实测结论

**DFlash 在这张卡上不可用。** 最优配置（n_max=2）68.36 → 65.87 t/s，比 MTP 低 3.6%；
n_max≥4 后跌破不开投机的基线 49.89；n_max=11 直接 CUDA OOM。

DFlash 的设计前提是"用大块摊薄草稿成本"，而 P100 恰好在大 batch 时**反向劣化**
（MMQ 被 sm_60 硬禁用转 cuBLAS，CUDA graph 被 `ggml-cuda.cu:4840` 禁用），
两头不讨好。

## 实验设置

| 项 | 值 |
|---|---|
| target | `DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q8_0` (qwen35, 33 层, Q8_0 9.11 GB) |
| draft | `Qwen3.5-9B-DFlash.Q8_0` (dflash 架构, 1.3B, 6 层, 1.32 GB) |
| DFlash 关键元数据 | `block_size=16`, `target_layers=[2,6,10,14,18,22,26,30]`, `n_extract=8`, **无 selector → DFlash 1 而非 2** |
| 上下文 | 8192（126K 时 14.3 GB + 1.4 GB draft 必 OOM） |
| prompt | 同一段 200 字中文论述，`max_tokens=192`, `temp=0.7` |
| 每配置 | 2 轮取中位，server 模式，`-fa on -lm none -np 1 --no-mmproj` |

## 数据

| 方法 | n_max | t/s | 相对基线 | draft/接受 | 接受率 | 轮数 | 每轮 ms | 每轮出 token |
|---|---|---|---|---|---|---|---|---|
| 基线 | — | 49.89 | 1.00× | — | — | 192 | 20.0 | 1.00 |
| **MTP** | **2** | **68.36** | **1.37×** | 107/167 | 64.1% | 85 | 33.0 | 2.26 |
| MTP | 3 | 51.02 | 1.02× | 116/226 | 51.3% | 76 | 49.5 | 2.53 |
| MTP | 4 | 49.96 | 1.00× | 126/253 | 49.8% | 66 | 58.2 | 2.91 |
| DFlash | 2 | 65.87 | 1.32× | 106/167 | 63.5% | 86 | 33.9 | 2.23 |
| DFlash | 3 | 50.67 | 1.02× | 118/214 | 55.1% | 74 | 51.2 | 2.59 |
| DFlash | 4 | 45.03 | 0.90× | 124/264 | 47.0% | 68 | 62.7 | 2.82 |
| DFlash | 5 | 36.45 | 0.73× | 122/338 | 36.1% | 70 | 75.2 | 2.74 |
| DFlash | 7 | 30.17 | 0.60× | 121/480 | 25.2% | 71 | 89.6 | 2.70 |
| DFlash | 11 | OOM | — | — | — | — | — | — |

轮数 = 总生成 token − 接受数（每轮至少产出 1 个 bonus token）。

## 三个根因

**1. 每轮多 1~4 ms 固定开销，来自 target 的特征外吐**

DFlash 要求 target 输出 8 个中间层的输入 embedding（`n_extract=8`，
拼成 `8 × 4096 = 32768` 维注入 draft 的 KV cache）。MTP 完全不需要这一步。

```
n_max=2：MTP 33.0 ms/轮　DFlash 33.9 ms/轮　(+0.9)
n_max=4：MTP 58.2 ms/轮　DFlash 62.7 ms/轮　(+4.5)
```

**2. 块内接受率衰减比 MTP 更陡**

| n_max | MTP 接受率 | DFlash 接受率 |
|---|---|---|
| 2 | 64.1% | 63.5% |
| 3 | 51.3% | 55.1% |
| 4 | 49.8% | 47.0% |
| 5 | — | 36.1% |
| 7 | — | 25.2% |

短块时两者持平（DFlash 在 n=3 甚至略高），一过 n=4 就断崖。
并行去噪缺左侧上下文，位置越靠后越不准——这正是 DFlash2 要用 selector lattice
和 two-tap 动态卷积去修的问题，而本地这个是 DFlash **1**，没有这些机制。

**3. 大块直接撞显存和 verify 掉档**

- n_max=11 → `CUDA error: out of memory`（非因果 attention 的块内激活 + 8 层特征 buffer）
- n_max=7 时每轮 89.6 ms 只出 2.70 个 token，等于花 7 个草稿的钱买 2.7 个

## 为什么 DFlash 的"并行"优势在 P100 上归零

DFlash 的核心卖点是"16 个草稿一次前向，省掉 16 次串行解码"。

但在 MTP 侧，串行的代价本来就只有 **3.0 ms/步**，其中 2.4 ms 是 kernel launch
（P100 无 CUDA graph）。n_max=2 时总共才 6 ms。

一旦把块加大到能体现并行优势的规模（n≥8），verify 侧就要 B≥9。
而 P100 的 verify 成本在 B=4 就跳档（MMQ 禁用，转 cuBLAS 另一个 kernel 档位），
实测 B=1/2/3/4/8 吞吐 50.1/96.7/126.2/104.5/115.8 t/s。

**draft 侧省下的几毫秒，被 verify 侧多付的几十毫秒吃掉。**

## 附带发现

- DFlash 草案是为**原版 Qwen3.5-9B** 训练的，本实验 target 是 Jackrong 的
  DeepSeek-V4-Pro 微调版。接受率打了折是预期的，但即便完全匹配，
  上面三条成本结构也不会变。
- n_max=2 时 DFlash 的 draft 计数（167/106）与 MTP（167/107）几乎一致，
  说明两者在这个场景下"能猜对多少"没区别，差距纯粹在成本。
- 显存：ctx=8192 时 target+draft 合计 11952 MiB；ctx=126000 时单 target 就 14.3 GB。

## 建议

1. **保持 MTP n_max=2**，这是 P100 + 9B Q8_0 的唯一最优点。
2. DFlash draft 模型（1.32 GB，`D:\lmstudio\AtomicChat\Qwen3.5-9B-DFlash-GGUF\`）
   留作对照即可，不影响启动器任何逻辑；要省空间可以删。
3. 换卡时的判据不是"卡多新"，而是 **target 有多慢**：

   | target 基线 | DFlash 预期 |
   |---|---|
   | < 20 t/s（大模型 / 低带宽 / MoE） | 可能 1.5~2.7×，值得开 |
   | 20~40 t/s | 边际，需按内容类型实测 |
   | > 45 t/s（本机 9B Q8_0 @ P100） | 结构上亏，除非同时满足 MMQ + CUDA graph 可用 |

4. 若要重测，优先改这三处而不是换硬件：`p_min`（社区推荐 0.3~0.5）、
   内容类型（代码/数学 vs 自由文本）、`--cache-type-k/v q8_0`（asf0 的标准配置，我未用）。

## 复现

```bash
python bench_dflash.py 2          # 全部配置各 2 轮
python -u -c "import bench_dflash as B; B.run('mtp_n3', B.build('mtp',3), 2)"
```

原始数据 `mtp_logs/dflash_vs_mtp.json`、`mtp_logs/dflash_scan.txt`，逐配置日志 `mtp_logs/df_*.log`。
