# Ternary Bonsai 2 27B on a V100 16 GB：把「跑不动」变成「够快」

[English](README.md) | **简体中文**

一套在 **Tesla V100-SXM2-16GB（sm_70，Windows/WDDM）** 上把 27B 三值量化模型
（1.75 bpw PTQ1_0 + 131072 上下文 + q8_0 KV + MTP 投机解码）从「长提示直接塌掉」
优化到「可用且接近上限」的移植与调优记录，含补丁、内核源码、控制台和全部实测数据。

## 结果（同机、同模型族，ctx 131072 + q8_0 KV）

| 指标 | 优化前（上游运行时） | 优化后 | 变化 |
|---|---|---|---|
| 预填充 12K 提示 | 175.8 t/s | **854.1 t/s**（12K 探针、ub=1024 生产档；复测 848~873） | **4.9×** |
| 预填充 55.5K 提示 | 13.55 t/s | 681.1 t/s | **50×** |
| 预填充 100K 提示 | 不可用 | 610~740 t/s | — |
| 解码 12K | 44.15 t/s | 45.75 → **53~60 t/s**（开 MTP） | +26~38%（相对 44.1 t/s 基准，随负载） |
| 解码 100K | — | 26~35 t/s | — |
| 显存（131072 ctx） | 15,847 MiB（上游运行时） | 13,979 MiB / 含 MTP（三档 ub 实测 13,952~13,980） | −1.9 GiB |
| PPL（2-chunk 语料 / q8_0，PQ2_0 → PTQ1_0） | 7.1131 | **7.1131**（逐位一致） | 无损 |

> 另有一组 50K 语料的 PPL 4.6649 对照，属于 Volta FA 移植与 q8_0-vs-q4_0 KV 的 A/B
> （见 `docs/RESEARCH.md` §17），不是 PTQ1_0 的等价性证据；PTQ1_0 的逐位一致证据是
> §7.2 的 2-chunk 语料 7.1131。

「优化前」不是模型慢：FlashAttention 内核会先把整段会话的 q8_0 KV 反量化成 f16
（131072 上下文时约 512 MiB，**每个 op 重做一次**），显存压爆后触发 WDDM 分页。

## 内容

```
patches/   1) oripoin-picks-20260926-full.patch   上游 842b18804 → 9b98d9dfd（oripoin 提交 + 本地适配）
           2) local-patch-v100-20260927.patch    9b98d9dfd → 生产工作树（PTQ1_0/m70 D256/注册/MTP 等）
           附：*-commits.txt 提交清单；single-feature/ 单点摘录（仅参考，勿叠加 apply）
code/      新增/重写的内核（D256 prefill FA、解码原型、PTQ1_0 planar mat-vec）——与补丁 2 内容一致，供阅读
           sm70-vendor/ 两个 BSD-3 许可文本 + 拉取说明；flash/ 的 7 个头文件随仓库分发（cute/cutlass 需自行拉取）
panel/     自建控制台（实时 tok/s、首 token、显存拆分、GPU 温度/功耗、一键接入等）
tools/     探针与基准脚本（prefill-probe、agent 轮次基准、阶段埋点、spec-bench、部署脚本）
docs/      研究记录：逐项 A/B、被否证的方向、性能账本、优化前后对比（一页纸有英文版 BEFORE-AFTER.en.md）
THIRD-PARTY-NOTICES.md + THIRD-PARTY-LICENSES/   上游与第三方许可清单与许可正文
```

## 怎么用

1. **底座**：`llama.cpp` 上游 `842b18804`（MIT）。
2. 依次打两个补丁（顺序不能反）：

   ```bash
   git apply patches/oripoin-picks-20260926-full.patch
   git apply patches/local-patch-v100-20260927.patch
   ```

   补丁 2 已包含三个内核源码及其注册代码（`fattn.cu` 的 D256 选择分支、`mmvq.cu` 的 PTQ1_0
   派发、`CMakeLists.txt` 的 include 守卫），所以不需要再手工拷贝 `code/`。
3. **vendored 头文件**（只有 D256 prefill 内核需要）：按 `code/sm70-vendor/README.md`
   ①把本仓库的 `code/sm70-vendor/flash/` 整个拷到 `ggml/src/ggml-cuda/sm70-vendor/flash/`
   （其中 2 个文件带我们的移植补丁、1 个是上游 CMake 生成文件的替身，上游拿不到）；
   ②再从 CUTLASS@`62750a2b` 取 `include/cute` 与 `include/cutlass` 放到同目录下。
   不想要这个内核就删掉 `fattn-sm70-d256*.cu`（或运行时设 `LLAMA_SM70_D256=0`）。
4. 构建（参考我们用的 sm_70 配置见 `tools/` 与 `docs/RESEARCH.md` §8）。
5. 控制台：`python panel/bonsai-dashboard.py --port 8090 --server-port 8080 --demo-dir <你的 model 目录>`。
6. 复现实验：`tools/` 里的脚本（`prefill-probe.py`、`agent-turn-bench.py`、
   `decode-phase-prof.ps1` 等），具体调用见 `docs/` 各节的"脚本"行；
   脚本↔文档对照见 `docs/TOOLS.md`。

## 注意事项

* **模型权重不在本仓库**：`Ternary-Bonsai-2-27B` 权重由其发布方授权（Apache-2.0），请从原始
  出处获取；本仓库只包含代码与实测记录。
* 数值口径：PTQ1_0 与 fp16/cuBLAS 路径的舍入不同，端到端**贪心输出逐字节一致**（见
  `docs/RESEARCH.md` §7.2）；但 **sm70 D256 prefill 路径的贪心输出会与 stock 分叉**（§20.3，
  预期行为，验收请看 PPL/日志分歧），PPL 这类指标换路径后请自行复核。
* 补丁侧重放：`patches/single-feature/*.patch` 是补丁 1 的单点摘录（fastmtp d2t、graph shape
  cache），**只作参考，不要再叠加 apply**。
* 文档里的 `work\...` 路径是作者工作树的内部路径，保留作溯源用；脚本里的目录请按你的环境改。
* 仓库里默认关闭的实验内核（`GGML_SM70_D256_DECODE`、`GGML_PTQ1_0_MULTI_CHUNK_MAX`、
  `SPC_DECODE_PROF` 等）只作记录，生产路径不启用。
* 详细的实验记录（`docs/`）目前是中文；本 README 与其结果表是英文，关键结论两边一致。

## 许可

本仓库的新增代码与文档：**MIT**（见 `LICENSE`）。上游与第三方许可见
`THIRD-PARTY-NOTICES.md`（MIT / BSD-3-Clause / Apache-2.0 混合，正在使用的全部为宽松许可）。
