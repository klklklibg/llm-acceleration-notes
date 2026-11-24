# PXA 4.0 / PXQN 量化实测（2026-11-23）

## 环境

- 引擎：`poisonxa16/pxa` v4.0.4，Windows 原生构建 `e37f292792`，MIT
- 主机：Tesla P100 16GB，`-ngl 99 -c 8192 -b 1024 -ub 512 -t 6 -fa on -np 1`
- 对照组：`D:\llama-p100\build-cuda\bin`（Kmic fork）
- 原始日志：`evidence/logs/pxa_*.log`

## 27B：PXQN 有效，标准 GGUF 大幅倒退

| 配置 | prefill (t/s) | decode (t/s) | 相对 fork |
|---|---|---|---|
| fork + IQ3_M 标准 GGUF | 197.4 | 17.83 | 基准 |
| PXA + 同一 IQ3_M GGUF | 108.7 | **10.27** | **-42%** |
| PXA + PXQN | 223.4 | **25.87** | **+45%**（重复跑 26.32，+48%） |
| PXA + PXQN + MTP (`n_max=4`) | 222.5 | **27.17** | +52%（对 plain +5%） |

**结论**：PXA 的加速完全依赖它自己的 PXQ/PXQN 张量类型。标准 GGUF 走通用路径，速度只有 fork 的一半多。**别把 PXA 当通用后端用。**

## 35B：Fusion4-PXQ2 装得下，MTP 有效

| 配置 | 模型体积 | prefill (t/s, 1314 tok) | decode (t/s, 128 tok) |
|---|---|---|---|
| fork + Ornith GSQ-RCO 3bit | — | — | 79.26 |
| PXA + Fusion4-PXQ2 plain | 11.2 GB | 901.2 | 65.66 |
| PXA + Fusion4-PXQ2 + MTP | 11.2 GB | **959.8** | **88.11** |
| 同上，重复跑 | — | — | 88.10 |
| 同上，32K 上下文 | — | — | 86.87 |

MTP 在 PXA 上 **+34% decode**（65.66 → 88.11）；同一 MTP 在 fork 上是 **-21%**。这是本次调研里最反直觉的结果：**投机解码的收益取决于后端实现，不取决于模型**。

Ornith 的 PXQ4 版本是 19.6GB，单卡 16GB 装不下；Fusion4-PXQ2 是唯一能塞进去的 35B PXQ 档位。

## 长提示 prefill

6605 token 提示，PXA + PXQN 27B：248.6 / 247.1 t/s（两轮），与短提示 223 t/s 同量级，没有观察到 prefill 随长度崩塌。

## 复现

```bash
python scripts/bench_pxa.py <tag> <引擎bin目录> <模型gguf路径> <端口> [额外参数]
# 例：python scripts/bench_pxa.py pxqn_mtp D:\pxa\engine\bin <model> 8096 --spec-type mtp:n_max=4
```

脚本自动等待 `/health`，跑固定长度生成，从 server 日志抓 `prompt eval time` / `eval time`。模型路径请填你自己的。
