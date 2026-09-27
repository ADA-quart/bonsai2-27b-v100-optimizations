# Ternary Bonsai 2 27B on a V100 16 GB：把「跑不动」变成「够快」

一套在 **Tesla V100-SXM2-16GB（sm_70，Windows/WDDM）** 上把 27B 三值量化模型
（1.75 bpw PTQ1_0 + 131072 上下文 + q8_0 KV + MTP 投机解码）从「长提示直接塌掉」
优化到「可用且接近上限」的移植与调优记录，含补丁、内核源码、控制台和全部实测数据。

## 结果（同机、同模型族、ctx 131072 + q8_0 KV）

| 指标 | 优化前（上游运行时） | 优化后 | 变化 |
|---|---|---|---|
| 预填充 12K 提示 | 175.8 t/s | **850.6 t/s** | **4.8×** |
| 预填充 55.5K 提示 | 13.55 t/s | 681.1 t/s | **50×** |
| 预填充 100K 提示 | 不可用 | 610~740 t/s | — |
| 解码 12K | 44.15 t/s | 45.75 → **53~60 t/s**（开 MTP） | +20~35% |
| 解码 100K | — | 26~35 t/s | — |
| 显存（131072 ctx） | 15,847 MiB（无 MTP） | **14.84 GB（含 MTP）** | −1.9 GiB |
| PPL（50K 语料 / q8_0） | 4.6649 | **4.6649**（逐位一致） | 无损 |

「优化前」不是模型慢：FlashAttention 内核会先把整段会话的 q8_0 KV 反量化成 f16
（131072 上下文时约 512 MiB，**每个 op 重做一次**），显存压爆后触发 WDDM 分页。

## 内容

```
patches/   相对上游 842b18804 的完整补丁（含 oripoin 9 个提交与本地适配）
code/      我们自己新增/重写的内核（D256 prefill FA、PTQ1_0 planar mat-vec、解码内核原型）
panel/     自建控制台（实时 tok/s、首 token、显存拆分、GPU 温度/功耗、一键接入等）
tools/     探针与基准脚本（prefill/probe、agent 轮次基准、阶段埋点、spec-bench、部署脚本）
docs/      研究记录：逐项 A/B、被否证的方向、性能账本、优化前后对比
THIRD-PARTY-NOTICES.md   上游与第三方许可清单
```

## 怎么用

1. **底座**：`llama.cpp` 上游 `842b18804`（MIT）。
2. 打补丁：`git apply patches/oripoin-picks-20260926-full.patch`
   （该补丁落在 `work/picks-llama` 分支上，对应 `RESULTS.md` 的提交清单）。
3. 需要 sm70 D256 prefill FA 时，把 `code/ggml-cuda/` 下的文件放进
   `ggml/src/ggml-cuda/`；它们依赖 vendored CuTe/CUTLASS 与 flash-attention 头文件
   （见 `THIRD-PARTY-NOTICES.md`，需自行 `git clone` 对应 commit）。
4. 控制台：`python panel/bonsai-dashboard.py --port 8090 --server-port 8080`。
5. 复现实验：`tools/` 里的脚本（`prefill-probe.py`、`agent-turn-bench.py`、
   `decode-phase-prof.ps1` 等），具体调用见 `docs/` 各节的"脚本"行。

## 注意事项

* **模型权重不在本仓库**：`Ternary-Bonsai-2-27B` 权重由其发布方授权，请从原始出处获取；
  本仓库只包含代码与实测记录。
* 数值口径：PTQ1_0 与 fp16/cuBLAS 路径的舍入不同，端到端**贪心输出逐字节一致**（见
  `docs/RESEARCH.md` 第 24 节），但 PPL 这类指标请在换路径后自行复核。
* 本文档里的路径已脱敏（`<REPO>` / `~`），部分脚本需要按你的目录改名。

## 许可

本仓库的新增代码与文档：**MIT**（见 `LICENSE`）。上游与第三方许可见
`THIRD-PARTY-NOTICES.md`，全部为 MIT / BSD-3-Clause 等宽松许可。
