# 原始实验证据

这里放未经修饰的原始结果，供复核笔记中的数字。

## JSON 结果

| 文件 | 内容 |
|---|---|
| `carveout-base.json` / `carveout-patch.json` | fp16 carve-out 的 3 轮交错 decode/prefill + 问答抽样 |
| `qual2-base.json` / `qual2-patch.json` | 同一对照的质量检查输出 |

## 原始日志（`logs/`）

均为 llama-server 原始 stdout，保留了 `prompt eval time` / `eval time` 行。

| 文件 | 对照组 |
|---|---|
| `pxa_fork27b.log` | 27B fork + IQ3_M（decode 17.83 t/s） |
| `pxa_pxa27b.log` | 27B PXA + 同一标准 GGUF（10.27 t/s，-42%） |
| `pxa_pxqn_plain.log` / `pxa_pxqn_mtp.log` | 27B PXQN，无/有 MTP（25.87 / 27.17 t/s） |
| `pxa_fork35b.log` | 35B fork + Ornith（79.26 t/s） |
| `pxa_fus4_plain.log` / `pxa_fus4_mtp.log` / `pxa_fus4_32k.log` | 35B Fusion4-PXQ2，plain / MTP / 32K（65.66 / 88.11 / 86.87 t/s） |
| `pp4k_pxa4k.log` | 27B PXQN 长提示 6605 tok prefill（248.6 t/s） |

**说明**：日志保留了当时的本机绝对路径（如 `D:/lmstudio/pfeifferj/...`），这是为了让实验可溯源——模型文件名本身就是实验配置的一部分。不含任何密钥或凭证。
