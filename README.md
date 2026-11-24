> **已合并到 [P100 LLM Toolkit](https://github.com/klklklibg/p100-llm-toolkit)** — 本仓库保留作为历史存档，后续更新在 monorepo 的 `acceleration-notes/` 目录。

# LLM 推理加速笔记：Tesla P100 (sm_60) 实测

一份**只记录实测数字和负结果**的本地大模型推理加速笔记。所有结论来自同一台机器、同一套对照方法，不是从 benchmark 榜单抄来的。

**硬件**：Tesla P100 PCIe 16GB (sm_60, GP100, 无 Tensor Core) + Ryzen 5 1600X (6C/12T) + 16GB RAM，Windows，CUDA 12.4（固定不动）。
**软件**：llama.cpp 多个构建（上游 master / Kmic P100 fork / 本地 Pascal 优化分支）+ PXA 4.0 引擎。

## 核心结论（先看这个）

| 技术 | 对 P100 是否有效 | 实测数字 |
|---|---|---|
| P100 专用 fork（Kmic） | **有效**，已日常使用 | 35B-A3B prefill 568 vs 459 t/s（+24%）；decode 79.2 vs 77.8（+1.9%） |
| `-fa on` | **有效**，已采纳 | prefill +1.9%，decode 持平 |
| KV cache `q8_0` | **负收益** | decode -15%；显存不紧张时用 f16 |
| 投机解码 draft-simple / ngram | **负收益** | ngram -15%，fork 上 MTP -21% → 默认关闭 |
| DFlash | **负收益** | 全面劣于 MTP，见 `notes/04` |
| PXA 4.0 + PXQN 量化 | **有效，但只对其私有格式** | 27B decode 26.0 vs 17.8 t/s（+52%）；标准 GGUF 反而 -42% |
| PXA MTP 投机 | **有效**（与 fork 相反） | 35B Fusion4 decode 88.1 vs 65.7 t/s（+34%） |
| sm_60 fp16 carve-out | 35B +1.7%，Inkling2 **-6.6%** | 因负例弃用，仅留档 |
| EAGLE3 | 机制成立但需同源 draft 头 | 见 `notes/01` §3.2 |

**一句话**：换后端/换量化的收益有，但"通用加速 trick"在这个架构上大多翻车。所有正收益都经过了 3 轮以上交错对照。

## 方法论

1. **先算物理上限**：带宽/位宽算出理论 t/s，再看实测差多少（见 `notes/01`）。
2. **逐改动隔离**：一次只改一个变量，base/patch 交错跑，避免热状态和显存碎片带偏。
3. **保留负结果**：-42%、-15%、-6.6% 这些数字和正收益一样入库，防止后来者重复踩坑。
4. **原始证据**：JSON 结果和 llama-server 原始日志都在 `evidence/`，不是只留结论表。

## 目录

- `notes/01-generation-speed-survey.md` — 带宽账 + 技术全景（量化/投机/后端/采样）
- `notes/02-mtp-research.md` — MTP 原理与实现解剖
- `notes/03-mtp-deep-dive.md` — 内容类型、KV 量化等第二轮研究
- `notes/04-dflash-report.md` — DFlash 实测，含社区反例对照
- `notes/05-pxa4-benchmark.md` — PXA 4.0 与 PXQN 量化实测（27B/35B）
- `notes/06-backend-and-kv-matrix.md` — 后端路由、KV 位宽、carve-out 负对照矩阵
- `evidence/` — 原始 JSON 结果与 llama-server 日志
- `scripts/` — 基准脚本

## 说明

- 原始日志保留了本机绝对路径（如 `D:/lmstudio/...`），用于实验溯源；不含密钥或凭证。
- 相关代码补丁见 [p100-patches](https://github.com/klklklibg/p100-patches)，启动器见 [llamacpp-launcher](https://github.com/klklklibg/llamacpp-launcher)。
- 笔记正文为中文；本文件为索引。

## 许可

MIT — 见 [LICENSE](LICENSE)。实测数据可自由引用，请注明来源硬件配置。
