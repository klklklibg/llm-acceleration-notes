# 后端路由、KV 位宽与 carve-out 对照矩阵

## 后端选择（35B-A3B，P100 sm_60）

| 后端 | prefill (t/s) | decode (t/s) | 备注 |
|---|---|---|---|
| 本地 Pascal 优化分支（旧） | 459 | 77.8 | SenseNova mHC GPU 路径只在此分支 |
| Kmic P100 fork (2026-10-28) | **568** | **79.2** | prefill +24%，decode +1.9%；日常默认 |

架构特例：
- `inkling2_loop` 只有上游 master 构建支持，其他后端直接 unknown arch。
- SenseNova-Loop 的 mHC router GPU 路径只在旧 Pascal 分支，fork 缺这段 CUDA 实现。

## 推理参数矩阵

| 改动 | 结果 | 决策 |
|---|---|---|
| `-fa on` vs auto | prefill +1.9%，decode 持平 | 采用 |
| `-ctk/-ctv q8_0` | decode **-15%** | 显存不紧张时用 f16 |
| `-ctk/-ctv f16` | 基准 | 默认 |
| 投机 draft-simple | 无收益 | 关闭 |
| 投机 ngram (`ngram-simple`) | **-15%**，且 fork 上曾崩 `GGML_ASSERT(n_accepted==0)` | 关闭 |
| fork 上 MTP | **-21%** | 关闭 |
| 采样 `-bs` | +3.7%，但长跑会崩 | 不采用 |

## sm_60 fp16 carve-out 负对照

改 `ggml/src/ggml-cuda/common.cuh`：在 3 处排除 sm_60 的 fp16 分支。原始 JSON 在 `evidence/carveout-*.json`。

**35B-A3B（3 轮交错，prompt 1902 tok，KV=f16）**：

| 构建 | 各轮 decode (t/s) | 均值 |
|---|---|---|
| base | 78.02 / 78.51 / 78.47 | 78.00 |
| patch | 78.51 / 79.71 / 79.73 | **79.32（+1.7%）** |

**Inkling2**：base 33.38 vs carve 31.18，**-6.6%**。

**决策**：尽管 35B 上有 +1.7%，因 Inkling2 明确倒退，补丁未合入日常构建，只在 `p100-patches` 仓库留档。这是"正收益不一定值得合入"的例子——收益小于噪声、且跨架构不稳定。

## 配套工具

`scripts/bench_pxa.py` 可直接改参数跑任意后端；启动器仓库的 `benchmarks/` 里有 KV 质量、MTP、DFlash、上下文曲线等更多脚本。
