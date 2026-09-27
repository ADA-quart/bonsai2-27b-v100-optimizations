# V100 / llama.cpp 相关项目与论文调研（2026-09-26）

目的：为「单卡 Tesla V100-SXM2-16GB + Ternary Bonsai 2 27B（PQ2_0）+ llama.cpp 分叉」
找可借鉴的内核实现与理论依据。检索途径：arXiv abs 页逐条核验、GitHub 搜索/PR 拉取
（arXiv/OpenAlex 的 API 在本机被网关限流，改用网页与 GitHub 连接器）。

## 一、与我们的瓶颈一一对应的项目

| 项目 | 是什么 | 对我们哪一块 | 关键数据 |
|---|---|---|---|
| [fishlikeX/sm70-attn](https://github.com/fishlikeX/sm70-attn) | **llama.cpp 分叉 + SM70 原生 FlashAttention 插件**，目标模型与我们同构（Qwen3.5/3.6/3.8-27B、head_dim=256、GQA 6:1、16 层全注意力） | 长上下文 FA（pp16384 的 15.6%） | 176k prefill 372.94 → **521.93 tok/s（+39.9%）**；SplitKV3 再 +3.7%；q4_0 核内直读；decode 持平（<0.1%） |
| [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | 在 Volta 上用 `mma.sync.m8n8k4` 跑 NVFP4/FP8 的 QPN2/QPN8 内核（1Cat-vLLM 之上） | decode 的 MMVQ（68%） | M=1~8 达 read roofline 的 **71–82%**；核心结论：*先反量化成 FP16 再 GEMM 等于白花 HBM 带宽*，packed 权重应直接解到 mma 寄存器操作数；MT=2 两 tile 共享一次权重读取 |
| [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) | 从零写的 C++/CUDA V100 引擎（Qwen3.6/3.8-27B、软件 NVFP4、MTP/DFlash2、int8 KV、前缀检查点、OpenAI/Anthropic 兼容接口） | 整套栈的工程结论 | pp2048 ≈ **1100 tok/s**；decode **219 tok/s**（K=1、接受率 99.2%）；每加一个 draft 行 +0.383 ms vs 串行一步 +0.817 ms；KV 声明 262,144 免费、填充才收费 |
| [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75) | Qwen 官方 [FlashQLA](https://github.com/QwenLM/FlashQLA)（TileLang 写的 GDN chunked 内核）的 **SM70/SM75 分叉** | GDN（pp16384 的 11.6%）——正是我们手写 chunked 失败的那块 | RTX 2080Ti(SM75)：GDN 阶段 1.126 → **0.520 ms（2.1×）**，整请求 prefill **+7.17%**；SM70 只做到编译通过，**V100 运行时验证仍是空缺** |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | 让现代 vLLM 跑在 Volta 的发行版 | 基础设施 | 内含 `flash_attn_v100`（15MB sm_70 设备码：paged KV、split-KV prefill、XQA/WMMA decode、量化 KV） |
| [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100) | FA2 base layer 的 V100 移植（sm70-attn 的 vendor 来源） | FA 内核素材 | BSD-3，可作二次开发基础 |
| [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) + [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) | 我们模型的上游：低比特内核 + 官方白皮书/已知问题 | 上游同步 | 白皮书 `bonsai-2-27b-whitepaper.pdf`；`KNOWN_ISSUES.md` 里直接列了待合并的性能 PR |
| [Quitetall/tritium](https://github.com/Quitetall/tritium) | 三值（多平面 additive-ternary）量化+推理基础设施，PyTorch PTQ/QAT + 多后端 | 三值表示与评测方法 | 把 recipe/校准证据/字节数/质量当作一个可审计产物；含 BitNet b1.58 2B 的 CPU/CUDA 证据 |

## 二、上游 `prism` 分支比我们本地基点新出的东西（可直接 cherry-pick 的候选）

本地 clone 的 `origin/prism` 引用是旧的（HEAD 领先 14、落后 0 是"没 fetch"的假象）。
上游实际已新增约 15 个提交，其中与性能相关的：

| 提交/PR | 内容 | 我们的对应关系 |
|---|---|---|
| `25092e7d5` (#216) | **GDN 内核 4 列/warp 铺到所有 Ampere+**（Ada pp2048 1220 → 1297 t/s，+6%） | **就是我们这一轮自己做的改动**（Volta +2.7%）；建议直接对齐上游版本，避免冲突 |
| #218 commit 5 | bf16 且行数 <64 的矩阵在 2~8 列时改走 mat-vec（mmvf.cu） | 同一个病灶（48 行的 ssm 门控投影）——我们用「bf16→fp16 转换 +5.9%」修掉，上游走 kernel 选择，两条路可并存 |
| #214 / #215 / #221 | 无分支 MMQ tile loader；PTQ1_0/PQ2_0 的 **planar-transposed 激活布局** 与 2/4 列混合路径 | Ampere 上 decode 1.5×；思路（激活读取才是多列的成本）对 PQ2_0 同样成立 |
| #248 / `842b18804` | x86 SSE2/SSSE3 vec_dot（PTQ1_0 与 PQ2_0），CPU decode 3.3–4.5× | 我们 mmproj 跑在 CPU 上 |
| `279df6644` | **DFlash2 投机解码进上游** | 面板里的 `--spec-type draft-dflash` 可以换成上游实现 |
| 新增旗标 `GGML_CUDA_BATCH_INVARIANT=1` | 1~4 列时让 logits 与单 token 逐位一致 | 正是「开 MTP 后 greedy 输出变了」的解法 |

另外我们构建里 **`GGML_CUDA_FA_ALL_QUANTS=OFF`**，而官方 KNOWN_ISSUES 明确写着
量化 KV 配 FA 需要它——值得做一次 `=ON` 的 A/B（我们对 q8_0 KV 的 FA 目前靠 oripoin 的
直读补丁撑着）。

## 三、论文（arXiv ID 已逐条核验）

**模型/架构**
- Gated DeltaNet：`2412.06464` *Gated Delta Networks: Improving Mamba2 with Delta Rule*
- DeltaNet 并行化：`2406.06484` *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
- Mamba-2：`2405.21060` *Transformers are SSMs*
- 我们的模型：PrismML **Bonsai 2 27B 白皮书**（仓库内 PDF）——PQ2_0/PTQ1_0 的量化方案与评测口径都在这

**低位/三值量化**
- `2402.17764` *The Era of 1-bit LLMs: All LLMs are in 1.58 Bits*
- `2404.00456` *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*（我们这套 Hadamard 旋转的源头）
- `2405.16406` *SpinQuant: LLM quantization with learned rotations*

**量化推理系统**
- `2408.11743` *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference*（sm80+，但调度思路通用）
- `2405.04532` *QServe: W4A8KV4 Quantization and System Co-design*
- `2407.00088` *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM*

**注意力**
- `2205.14135` *FlashAttention*（唯一支持 Volta 的一代）
- `2307.08691` *FlashAttention-2*（Ampere+，V100 不可用）

**Volta 硬件本身**
- `1804.06826` *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*
  （`mma.sync.m8n8k4`、HMMA 吞吐、L2/带宽的权威数据）

**投机解码**
- `2211.17192` *Fast Inference from Transformers via Speculative Decoding*
- `2401.10774` *Medusa*
- `2412.19437` *DeepSeek-V3 Technical Report*（MTP 的原始描述）

**KV 量化**
- `2402.02750` *KIVI*、`2401.18079` *KVQuant*

**系统评估 / "llama.cpp 论文"**
- `2211.05102` *Efficiently Scaling Transformer Inference*（V100 时代的 scaling 分析）
- `2601.14277` *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization*（2026-01）
- 说明：**llama.cpp 没有官方论文**。它的"论文"实际是三类东西：上游 PR/commit 里的实测报告
  （质量最高，见第二节）、模型卡/白皮书、以及围绕它的评测论文（如 `2601.14277`）。

## 四、按 ROI 的下一步建议

1. `git fetch` 上游 `prism` → cherry-pick #216（GDN 列布局，与我们已有改动合并）、#214
   （MMQ tile loader）、#218 的 mmvf 修正；再做一次 `GGML_CUDA_FA_ALL_QUANTS=ON` 的 A/B。
2. 移植/借鉴 **sm70-attn 的 D256 Split-D FA 内核 + SplitKV3**：目标是 pp16384 那 15.6% 的 FA
   （参考值 +39.9%@176k，同模型族）。
3. **FlashQLA-SM70** 作为 chunked GDN 的第二条路（我们手写 WY 版失败，但那条路在 SM75 上
   已被证明有 2.1×）；核对它的数值稳定性处理与我们的门控约定。
4. 若要动 decode 的 68%（MMVQ），**v100-skinny 的 QPN 思路**（packed 权重直接解到
   `mma.sync.m8n8k4` 寄存器操作数 + 激活的 planar-transposed 布局）是天花板最高的方向，
   属于周级工程。

---

## 五、llama.cpp 生态里的项目（2026-09-26 补充）

### 5.1 分叉链：我们这条线

```
ggml-org/llama.cpp（上游）
  └─ PrismML-Eng/llama.cpp（prism 分支：PQ2_0/PTQ1_0 三值内核 + Hadamard 旋转 + MTP/DFlash）
       ├─ git.oripoin.me/oripoin/llama.cpp（Turing 专项：tile q8_0 KV 直读、MMQ 窄 tile、benches/turing）
       └─ 我们：work/picks-llama（= prism + oripoin 的挑选 + 本地 GDN cols_per_warp 等）
  └─ fishlikeX/sm70-attn（llama.cpp + SM70 原生 FlashAttention 插件）
```

### 5.2 同一个模型的社区项目（最能直接抄）——Ternary Bonsai 2 27B

| 项目 | 是什么 | 可抄的东西 |
|---|---|---|
| [professorpalmer/bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery) | **23 个 `git am` 补丁**打在 PrismML `prism@9a9394a` 上，12GB 卡跑 262k 窗口 + 服务配方；分支 `llama.cpp-ada-ternary#bonsai-combo` | 补丁 0008/0009（无分支 MMQ tile loader + 4 列 GDN 布局）、0007（**GDN 的 recurrent-state gather 折进内核**，他们的 #220）、0012（**FA 的 MMA 直接原地读 q4_0/q8_0 K/V，不再拷成 F16 scratch**）、0016（**Hadamard 变换给自己输出做量化，每步少 ~390 次 launch**）、0020（`--spec-draft-depth-max`：深上下文停止草稿）、0004-0006（`GGML_CUDA_BATCH_INVARIANT` + bf16 小行 mat-vec） |
| [sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu) | PTQ1_0 小批量 mat-vec 内核（3060 decode 26.3 → 40.5 tok/s）+ **MTP 头嫁接工具**（能读写 PrismML 自定义量化类型的 byte-range GGUF 读写器）+ 各显存档位的 serve 脚本 | 内核 `pr-ptq1-mmv`（上游 #218）；Hadamard-MTP 修复 `pr-hadamard-mtp`（#217）；`graft/` 工具（把 MTP 头并进 GGUF，含 byte-exact strip 校验） |
| [ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP) | 这个模型**官方权重之外的 on-policy MTP 头**（被 bonsai-ada-surgery 用作默认嫁接源，接受率 70.6% vs 教师头 66.3%） | 我们现在用独立的 FastMTP 草稿模型；换成嫁接进主 GGUF 的 MTP 头可能更省显存/更快 |
| 质量与配方结论（`docs/QUALITY.md`） | 同一模型的实测：**q4_0 KV 在深度上每 48 个位置翻 1 次 top-1，q8_0 是 1/160**；`reasoning_effort=xhigh` 会加一条"想仔细点"的系统行导致"reasoning madness"；工具调用走原生 XML+grammar 9/9 通过，写成 JSON-in-content 只有 1/9 | 直接印证我们之前那些 thinking/reasoning 的调整；KV 精度取舍有数据了 |

### 5.3 通用生态（挑与我们相关的）

| 项目 | 用途 | 对 V100 |
|---|---|---|
| [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | 性能向大分叉：Trellis/IQK 量化、**融合 delta-net（GDN）PR 1315/1333/1362/1373**、**Qwen 3.5/3.6/3.8 的 MTP（含独立 MTP 头 `-md`）**、DFlash/DSpark、**KV 的 Hadamard 旋转（CPU #1033 / CUDA #1034, V #1527）**、递归模型检查点、自动按显存摆放张量 | ⚠️ README 明确"只有 CPU(AVX2+) 与 CUDA(Turing+) 是完整可用后端"，**Volta 不在支持范围**；但 PR 里的算法级工作（GDN 融合、MTP、KV 旋转）可以借鉴 |
| [eugr/llama-benchy](https://github.com/eugr/llama-benchy) | 面向 OpenAI 端点的基准工具：按 **上下文深度** 扫 pp/tg、测 prefix-cache 命中路径、**正确处理 MTP 多 token chunk**、并发、导出 JSON/CSV | 硬件无关，正好补上我们现在用 ad-hoc 探针的短板（v100-skinny 也用它测深度曲线） |
| [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap) | 在多个 llama-server 配置间热切换的代理 | 硬件无关；可替代面板里的部分流程 |
| [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) / [ollama/ollama](https://github.com/ollama/ollama) / [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) | llama.cpp 之上的发行版（自带后端打包 / 调度 / 单文件分发） | 一般不针对 Volta 做内核优化；llamafile 的 tinyBLAS 是自研 GEMM，可当参考 |
| [alexziskind1/llama-throughput-lab](https://github.com/alexziskind1/llama-throughput-lab) | llama-benchy 的实时可视化（进度流） | 硬件无关 |

### 5.4 从 5.2 里挑出来、对我们 V100 最可能有效的补丁

1. **0012 / #221：FA 的 MMA 原地读 q4_0/q8_0 K/V** —— 我们现在靠 oripoin 的 q8_0 直读补丁，
   上游已把它做进 MMA 路径，直接对齐可以拿到同一份收益 + 少一次 F16 拷贝。
2. **0007 / #220：GDN 的 recurrent-state gather 折进内核** —— 我们 pp16384 里 GDN 占 11.6%，
   且我们此前只动了列布局。
3. **0016：Hadamard 自量化，每步少 ~390 次 launch** —— 我们的 kernel 账本里 `fwht_cuda_block` +
   `convert_unary` + `quantize_q8_1` 加起来在 decode 超 10%。
4. **0020：`--spec-draft-depth-max`（深上下文停止草稿）** —— 与我们实测"长上下文开 MTP 反而慢"
   的结论一致，直接用现成开关。
5. **0004-0006：`GGML_CUDA_BATCH_INVARIANT` + bf16 小行 mat-vec** —— 前者解决 MTP 开关导致
   greedy 输出变化，后者是我们在 PQ2_0 上已用 bf16→fp16 转换修掉的同一个病灶。

⚠️ 但注意：这批补丁的验证环境是 Ada（sm_89）/Ampere（sm_86），**没有一条在 sm_70 上验证过**；
其中 planar PTQ1_0 mat-vec、MT=2、mma 路径都依赖 Turing+ 的指令。移植时需要逐条判断
"算法级"（可移植）还是"指令级"（不可移植）。

---

## 六、执行记录：逐条落地（2026-09-26 19:xx）

### 6.1 逐条 sm_70 可用性判定（`work/tmp/classify-patches.py` 输出）

| 补丁 | 内容 | sm_70 判定 |
|---|---|---|
| 0001/0002/0003/0010/0011/0013/0021 | PTQ1_0 planar 激活布局 + 专用 mat-vec（含 SoA/混合列） | ✗ **PTQ1_0 专用**（我们的模型是 PQ2_0）；指令本身是 dp4a/`__byte_perm`（sm_61+ 有），**若要跑 PTQ1_0 模型则可移植** |
| 0008 | 无分支 PTQ1_0 MMQ tile loader + Ampere tile 表 | ✗ PTQ1_0 + Ampere 配置表 |
| 0009 | GDN 4 列/warp 铺到 Ampere+ | ✅ 已在我们的基点里（我们还自己扩到了 Volta + 加 `n_tokens>=8` 门限） |
| 0012 | FA 的 **MMA** 路径原地读 q4_0/q8_0 K/V | ✗ MMA 路径是 Turing+；sm_70 上 FA 走非 mma 分支，**等价物（oripoin 的 q8_0 直读补丁）我们已有** |
| 0023 | planar PTQ1_0 的 PDL 依赖等待 | ✗ Hopper/Blackwell（Volta 无 PDL） |
| 0004/0005/0006/0007/0014/0015/0016/0017/0018/0019/0020/0022 | 见下 | ✅ 多数可移植 |

### 6.2 已落地并验证的

**批一（0004 + 0014 + 0015 + 0017 + 0018 + 0020）—— MTP/投机解码链**（提交 `d5b463e13`）

* 0015 把「catch-up 行」从 `process()` 推迟到下一步 `draft()` 的同一个 `llama_decode`（省一次图启动），
  0014 让 MTP 图只发布输出行配合它，0018 丢弃过期行，0017 拒绝 OOV id，
  0020 新增 `--spec-draft-depth-max`（深上下文停止草稿），0004 新增 `GGML_CUDA_BATCH_INVARIANT`（默认关）。
* 交替 A/B（`work\tmp\mtp-ab2.ps1`，2 轮 × 4 任务 = 各 8 次，服务端 `draft-mtp n-max 2`）：

| 任务 | pre（部署版） | post（打补丁） | Δ |
|---|---|---|---|
| math | 50.0 | 53.2 | +6.5% |
| code-csv | 41.8 | 44.0 | +5.2% |
| code-bs | 47.8 | 50.8 | +6.3% |
| chat | 47.9 | 50.2 | +4.8% |
| **均值** | **46.87** | **49.55** | **+5.7%** |

接受率完全一致（同一批草稿）；PPL 7.1131 不变；pp2048/pp16384/tg128 不变。

**批二（0007）—— GDN 的 recurrent-state gather 折进内核**（提交 `4d0a06e3c`）

* `build_rs` 的 `GET_ROWS(cache, ids)` 临时张量只被 GDN 的 state 输入消费，于是图求值时跳过该节点，
  内核直接用 `s_ids[seq]` 索引 cache 行；省掉每层一个 gather 内核（27B 上 3 MB 临时）及其 L2 污染。
* V100 A/B（`GGML_CUDA_GDN_GATHER_FUSION=0` 对照）：tg128 **50.08 ± 0.38 → 50.94 ± 0.23（+1.7%）**，
  pp2048/pp16384 不变，**PPL 两臂逐位相同（5.4426 @ `--chunks 4`）**。
* 移植时要做两件事：`common.cuh` 补 `<unordered_map>`；launch hunk 与我们自己的 WIDE（4 列/warp）
  分发手工合并（补丁按单实例写，我们有 `if (wide)` 双实例）。

**已在上游/基点里，无需移植**：0006（MTP 图的 token embedding Hadamard 逆）——我们的 `qwen35.cpp`
`graph_mtp` 里已有等价代码。

**对我们无实际收益**：0005（bf16 小行矩阵走 mat-vec）——我们的模型已经把这 96 个 bf16 张量
转成 fp16（第十八节），树上不再有 bf16 权重。

### 6.3 待做

* 0016（Hadamard 变换自量化，省 ~390 次 launch/步）+ 0019（FWHT q8 池的 LIFO 释放，
  依赖 0016）：需要写一个 **PQ2_0 版**——原补丁是给 PTQ1_0 的 planar/SoA 布局写的，
  我们只需要 AoS（标准 `block_q8_1`）那一路。
* 0001/0002/0010/0011/0013（PTQ1_0 专用内核栈）：若哪天改用 PTQ1_0 模型（同权重、1.75 bpw、
  5.95 GB，生态优化都在这边），这批就是主线。

### 6.4 0016 的 PQ2_0 版：实现完成、但 **V100 上净亏 2.5%** → 否决（留档）

补丁留档：`work/patches/local-fwht-q8-fusion-REJECTED-20260926.patch`（约 320 行）。

实现内容（只保留 AoS 一路）：
* `quantize.cuh`：把 `quantize_q8_1` 的逐块数学抽成 `quantize_q8_1_lane()`；
* `fwht.cu`：把 block 蝶形抽成共享内联 `fwht_block_row()`，新增
  `fwht_cuda_block_q8_1`（变换后直接写 q8_1 块，布局与独立量化内核逐位一致）；
* `common.cuh`：`ggml_cuda_fwht_q8` 注册表 + pool 块持有（输出与输入别名时用）；
* `ggml-cuda.cu`：pass 挂在**真正热的那条路上**——`{MUL(signs), RESHAPE, MUL_MAT(hint)}`
  三节点融合点（`ggml_cuda_op_fwht_signed`，259 次/解码步），而不是节点循环（那里看不到）；
* `mmvq.cu`：命中注册表时跳过 `quantize_row_q8_1_cuda`。

实测（V100，f16ssm 模型）：

| 指标 | 融合 ON | OFF |
|---|---|---|
| PPL（`--chunks 4`） | 5.4426 | 5.4426（**逐位一致** ✓ 实现正确） |
| 内核 launch 总数 | 207,178 | 217,815（−4.9%） |
| `quantize_q8_1` | 41,347 次 / 97.6 ms | 43,473 次 / 98.7 ms |
| `fwht` | **31,654 次 / 145.2 ms（4.59 µs/launch）** | 33,282 次 / 108.1 ms（3.25 µs/launch） |
| `mul_mat_vec_q` | 1,492.5 ms | 1,552.8 ms |
| 内核时间合计 | 2.20 s | 2.25 s |
| **tg128 墙钟（3 轮交替 × r=8）** | **48.44 t/s** | **49.69 t/s（−2.5%）** |

结论：**在 Volta 上否决**。原因是内联量化每 32 元素子块要两次串行 warp 归约
（`max` 与 `sum`），把这个本来就偏延迟受限的 fwht 内核从 3.25 µs 抬到 4.59 µs/launch
（+41%），超过省下的独立量化内核；内核时间账本看似下降，墙钟却一致变慢。上游在 Ada 上
能赢（SM 更多、延迟更容易被掩盖），我们这台不行。别名只占 19%（192/1028），不是分配开销的问题。

### 6.5 本轮结束时的状态

* 已落地（本地提交）：`e60cd6fa7`（GDN 4 列/warp + Volta 门限）、`d5b463e13`（MTP 链 + 标志）、
  `4d0a06e3c`（GDN gather 折叠）。
* 已部署：`work/bin-backup/20260926-204606-full` 为部署前快照；服务端 12k prompt 探针
  **960.4 t/s**（本轮开始时 947~957），tg128 `llama-bench` **51.20 ± 0.16**（r=6），
  PPL 7.1131（`--chunks 2`）/ 5.4426（`--chunks 4`）不变。
* 未采用：0001-0003/0008/0010-0013/0021（PTQ1_0/Ampere/mma 专用）、0012（sm_70 无 mma 分支）、
  0016+0019（见 6.4）、0022（依赖 PTQ1_0 布局）、0005（我们已无 bf16 权重）、
  0006（基点里已有）。

---

## 七、PTQ1_0 路线：整栈移植 + 切换生产模型（2026-09-26 22:xx）——本轮最大收益

### 7.1 为什么是 PTQ1_0

官方同时发两种打包，**同一批三值权重**：

| 打包 | 位宽 | 文件 | 说明 |
|---|---|---|---|
| PQ2_0 | 2.13 bpw | 6.71 GiB | 每个 trit 占 2 bit 槽 |
| **PTQ1_0** | **1.75 bpw** | **5.54 GiB** | 稠密 base-3 打包（省 1.17 GiB） |

而整个社区（ninfer、v100-skinny、bonsai2-small-gpu、ada-surgery）的优化都在 PTQ1_0 这条线上。
关键判断：PTQ1_0 的解包只用 `__byte_perm` + dp4a（sm_61+ 就有）**不需要 Turing 的 mma**，
所以理论上能编到 sm_70 —— 这一点本轮验证成功。

### 7.2 移植结果

套用：0001（planar-transposed 激活布局）、0002（专用 1~4 列 mat-vec，全 lane 利用）、
0003（测试形状）、0010（SoA q8 激活 + 精确整数和）、0011（混合单/多列分发）、
0013（多列 raw digits + per-pair epilogue）、0021、0022。
**0008 与配置表本来就已经在树里**（branch-free loader + 16 条 PTQ1_0 tile 表 ✓，
且 Volta 走 `mmq_get_config_ampere`）。

正确性：`test-backend-ops test -o MUL_MAT -p type_a=ptq1_0` → **157/157 通过**。
模型侧同样做了一次无损 bf16→fp16（96 个 ssm 张量，8016/23.6M 次正规数舍入，与 PQ2_0 同一脚本）。

同构建、同权重 A/B：

| 指标 | PQ2_0 | **PTQ1_0** | Δ |
|---|---|---|---|
| PPL (`--chunks 2`) | 7.1131 | **7.1131** | 逐位一致 |
| pp2048 (`-ub 1024`) | 1084.91 ± 18.83 | **1133.63 ± 3.69** | **+4.5%** |
| pp16384 | 945.43 ± 2.27 | **977.53 ± 0.52** | **+3.4%** |
| tg128 | 50.13 ± 1.73 | **53.36 ± 0.70** | **+6.4%** |
| 权重显存 | 6.70 GiB | **5.53 GiB** | **−1.17 GiB** |
| 服务端 MTP 解码（spec-bench 4 任务均值） | 49.55 | **53.03** | **+7.0%**（接受率与草稿完全一致） |

服务端实测（12k prompt 探针）：**960.4 → 975.8 t/s**；显存占用 **14452 → 12904 MiB（−1.5 GB）**。

### 7.3 已切换

面板 `dashboard-config.json` 的 `model` 指向 `Ternary-Bonsai-2-27B-PTQ1_0-f16ssm.gguf`
（PQ2_0 两个文件都还在磁盘上；配置备份 `dashboard-config.backup-ptq10-*.json`）。
本地提交 `bea69867f`（+935 行）。二进制已部署（快照 `bin-backup/20260926-225634-full`）。

### 7.4 剩下的方向

* **sm70-attn 的 D256 Split-D FA 内核**（同模型几何，176k prefill +39.9%）——对应我们 pp16384 里
  仍占 15.6% 的 FA；这是目前最大的一块。
* ProCreations 的 on-policy MTP 头嫁接进主 GGUF（省掉独立草稿模型的显存/切换成本）。
* 上游 `prism` 的 8 个新提交（CPU AVX-512 / Metal / server 显示）对本机无收益。

## 八、sm70-attn 的 D256 Split-D FlashAttention：集成完成并上线（2026-09-27 01:xx）

本地提交 **`5e36a7d40`**（分支 `local/picks-20260926`，未推送）。二进制已部署到
`work/bonsai-demo/bin/cuda`（SHA256 校验，快照 `work/bin-backup/20260927-010123-full`），
生产 server 已重启（PID 61128，显存 **12921 MiB**，与部署前 12904 基本一致——镜像缓冲本来
就由 stock 路径分配，没有额外占用）。

### 8.1 集成方式

新增 `ggml/src/ggml-cuda/fattn-sm70-d256.{cu,cuh}` + `sm70-vendor/`（cute/cutlass，13 MB，
MIT）。`fattn.cu` 里注册 `BEST_FATTN_KERNEL_SM70_D256 = 500`：

* 选择条件：`cc == sm_70`、head_dim 256、`q_len >= 256`（只接管 prefill）、有显式 mask、
  GQA 比例整除、KV 类型可转 f16（f16 直读 / q4_0 内核内直读 / 其余走 f16 镜像）。
* `get_alloc_size` 必须提前 return 本内核的 scratch（output + 镜像 + Qs/Os + SplitKV3 部分和），
  否则越界写（早期症状：PPL 498）。
* `CMakeLists.txt`：`sm70-vendor` include + `--std=c++17` + `/Zc:__cplusplus` + `M_LOG2E`。

### 8.2 踩到的两个真 bug（都有实测数据）

1. **多 stream ubatch（`k->ne[3] = n_stream > 1`）**：launcher 传的 `k/v_outer_stride = 0`，
   于是每个 stream 都去读 stream 0 的 KV。`llama-perplexity --chunks 2` 的 ubatch 正好是
   2 个序列各 256 token（`Q=(256,256,24,2)`），所以在 chunks=1 时"看起来是对的"、chunks=2 时
   **PPL 3.679 vs 7.113**。修法：f16/q4-direct 用 `nb[3]`（元素/字节），staged 镜像用
   `ne0*ne1*ne2`。q4-direct 分支的内核 head base 也补了 batch 项。
2. **staged f16 镜像的布局**：本 fork 的 `ggml_is_contiguously_allocated()` 只比较"总跨度 vs
   紧凑大小"，KV 缓存视图（`nb[1]=hkv*row, nb[2]=row`）因此被判为"连续"，走 **flat 反量化**
   分支——镜像保持**缓存内存顺序**，步长必须按 `nb*blck/type_size` 缩放（stock 代码正是这么做的）。
   我先前假设成 packed `[ne0][ne1][ne2]` 布局 → **PPL 5.67 vs 12.29**（q8_0 KV，chunks=1）。
   用新加的 `SM70_DUMP_RAW` 把原始缓存和镜像落盘逐元素对比后定位：f16 现在 max diff = 0，
   q8_0 差 2e-4（就是 f16 舍入）。

### 8.3 验证结果（q8_0 KV，`-fa on`，同一二进制 A/B）

| 场景 | OFF | ON | Δ |
|---|---|---|---|
| PPL q8_0 `--chunks 1` | 12.2854 | **12.2796** | −0.005（噪声级） |
| PPL q8_0 `--chunks 2` | 7.1260 | **7.1242** | −0.002 |
| PPL f16 `--chunks 2` | 7.1131 | **7.1113** | −0.002 |
| pp512 / pp1024（-ub 512，交错 r=8×3 次） | 969.7 / 965.7 | **974.3 / 973.0** | +0.5% / +0.8% |
| pp2048（-ub 1024） | 1097.3 | **1107.4** | +0.9% |
| pp8192（-ub 1024） | 1011.0 | **1050.8** | +3.9% |
| pp16384（-ub 1024） | 949.9 | **1025.9** | **+8.0%** |
| pp32768（-ub 1024） | 824.4 | **938.2** | **+13.8%** |
| tg128（解码） | 49.09 | 48.21 | 噪声（解码不走这个内核） |

生产实测（12k prompt 探针，`-ub 1024 -b 8192`）：**970–976 → 998.6 / 1008.1 t/s（+2.5~4%）**；
对话冒烟测试输出正常（`2, 3, 5`，推理链完整）。

注意短序列那一档要用高重复度交错测：`-p 512 -ub 512 -r 3` 早期测出过 −25% 的假回归，纯属
并发负载噪声（同一配置 r=8×3 次测得 +0.5%）。

### 8.4 调参决定

`LLAMA_SM70_SPLITKV3_MIN_KV` 默认 **2048 → 4096**：在 kv_len=2048 时 3 路切分 + 合并净亏 ~1%
（1077 vs 1087），8192/16384/32768 开关无差别（1051/1005/938）。

### 8.5 还没做的

* q4-direct 路径只做了 batch 项的修正，没有实测多 stream（默认关闭，生产不用）。
* 构建仍是 `GGML_CUDA_FA_ALL_QUANTS=OFF`（官方 KNOWN_ISSUES 建议量化 KV + FA 时打开）——
  A/B 未做。

### 8.6 `test-backend-ops -o FLASH_ATTN_EXT`（回归网，已跑）

ON / OFF 两个 arm 的结果**完全一致**：`not_supported = 1688`、`cuda_errors = 2`，两条 CUDA
错误都是同一条 `hsk=320, permute=[0,2,1,3]` 用例在 **stock 的 mma 路径**里
`cudaFuncSetAttribute` 失败（V100 动态 smem 上限），与本次集成无关（关闭 sm70 后逐字复现）。
也就是说**没有引入新的失败用例**。

注意：该测试的 FA 用例大多传 `mask=0`（空 mask），而我们的内核只在有显式 mask 时接管，
所以它主要起"没有回归"的作用；内核本身的正误由上面的 PPL 四组对照 + 镜像逐元素核对负责。

## 九、ProCreations 的 on-policy MTP 头嫁接进主 GGUF（2026-09-27 08:xx）——省掉独立草稿模型

### 9.1 结论先说

* 手里那份 `Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` 是**合并文件**（851 个原 base 张量逐字节未变 +
  15 个 `blk.64.*` 头张量，Q8_0 矩阵 + F32 范数）。所以没必要换用它（那是 PQ2_0，会丢掉 PTQ1_0 的
  +4.5%/+6.4%），而是把那 15 个张量**嫁接到我们的 PTQ1_0 GGUF 里**。
* 用 HTTP range 请求只取头部 16 MB 解析出张量表，再只下载 `blk.64.*` 的 430.4 MiB（全文件 7.13 GiB），
  24.8 s 写出 `Ternary-Bonsai-2-27B-PTQ1_0-f16ssm-MTP.gguf`（5.96 GiB，866 张量，`block_count=65`，
  `qwen35.nextn_predict_layers=1`）。校验：453 个源张量逐字节一致（含 token_embd / output）、
  15 个头张量与原始 blob 逐字节一致。
* 我们的运行时早就具备条件：`qwen35.cpp` 的 `graph_mtp` 已经带着 ProCreations 那个
  Hadamard 反变换修复（`src/models/qwen35.cpp:649-653`），`--spec-type draft-mtp` 也在。

### 9.2 最大的坑：`-md` 不能传

`--spec-type draft-mtp` 的草稿头**跑在目标模型自己的权重上**。如果额外传
`--spec-draft-model <同一个文件>`，运行时会**把整份 5.83 GB 权重再加载一遍**：

```
0.03.878 I load_tensors: offloaded 66/66 layers to GPU
0.03.878 I load_tensors:        CUDA0 model buffer size =  5825.74 MiB   <- 目标
0.14.683 I load_tensors: offloaded 66/66 layers to GPU
0.14.683 I load_tensors:        CUDA0 model buffer size =  5825.74 MiB   <- 草稿（重复！）
```

* ctx 8192：两份都进显存，能跑，解码 64.95 t/s。
* ctx 131072（生产档）：+q8_0 KV 直接压过 16 GB，草稿被挤到慢速路径，
  **prompt 处理从 1015 t/s 掉到 3.8 t/s**，解码 19.8 t/s。

不传 `-md` 时草稿上下文只多 **+1.04 GB**（12867 → 13906 MiB）。ada-surgery 的
`start-server.ps1` 就是这么用的（`@('--spec-type','draft-mtp','--spec-draft-n-max',…,'-ctkd',…)`，无 `-md`）。

### 9.3 实测（131072 ctx / q8_0 KV / -b 8192 -ub 1024 / V100，同一嫁接文件 A/B）

| 场景 | 基线 | +MTP（嫁接头，n_max=2） | Δ |
|---|---|---|---|
| spec-bench 四任务解码（短提示） | 44.1 t/s | **55~61 t/s**（接受率 52~88%） | **+26~38%** |
| 12K 深度解码 | 39.6 t/s | **50.3 t/s**（接受率 87%） | +27% |
| 24K 深度解码 | 38.9 t/s | **50.9 t/s**（接受率 84.8%） | +31% |
| 48K 深度解码 | 33.3 t/s | **40.5 t/s**（接受率 60.7%） | +22% |
| 冷启动 12K prefill | 1015 t/s | 558 t/s | **−45%**（≈ +0.85 ms/prompt-token） |
| 冷启动 48K prefill | 886 t/s | 506 t/s | −43% |
| 前缀缓存命中（同 prompt 第二次） | prompt 203 ms | prompt 278~401 ms | 基本持平 |
| 显存 | 12867 MiB | 13906 MiB（生产带 mmproj：14742 MiB） | +1.04 GB |

`LLAMA_MTP_EAGER_CATCHUP=1`（catch-up 改成前置）对 prefill 没有帮助（557.8 vs 559.2 t/s）。
ada-surgery 说"深度 24K 以后草稿不再划算"（4070 上），**在 V100 上不成立**——我们到 48K 还有 +22%，
所以 `--spec-draft-depth-max` 默认取 **0（不停止）**。

### 9.4 面板与生产

* `outputs/bonsai-dashboard.py`：
  - `draft-mtp` 分支**不再传 `--spec-draft-model`**（改成只发 `--spec-type/--spec-draft-n-max/
    --spec-draft-type-k/-v` + 可选 `--spec-draft-depth-max`），其它 spec 类型（dspark 等）保持原样；
  - 新增 `spec_draft_depth_max` 配置项与 UI 输入框，默认 0；
  - 默认配置与「Agent 最优」「短会话」两个预设都改成开 MTP（附实测注记）。
* 生产 `dashboard-config.json`：`model` → `Ternary-Bonsai-2-27B-PTQ1_0-f16ssm-MTP.gguf`，
  `spec_type=draft-mtp`，`spec_draft_model=""`，`spec_draft_depth_max=0`
  （备份 `dashboard-config.backup-mtp-*.json`）。
* 验证：服务端命令里没有 `-md` ✓；12K 探针 prefill 555.5 t/s、解码 **51.5 t/s**；对话冒烟
  `2, 3, 5`，解码 54.8 t/s；显存 14742 MiB（含视觉塔）。

### 9.5 还没做的

* MTP 头的 catch-up 成本看着偏高（0.85 ms/prompt-token ≈ 1 TFLOPS 量级，像是按 mat-vec 跑的）。
  头是 **Q8_0**，而我们的 sm_70 优化全在 PTQ1_0 上——把头重量化成 PTQ1_0（430 MB → ~90 MB）很可能
  同时省显存和加速 catch-up；因为草稿只影响接受率、正确性由目标模型验证，这个损失是可接受的方向。
* `-ctkd/-ctvd` 目前固定 q8_0（面板里写死），没做 KV 类型 × MTP 的 A/B。

## 十、清尾：FA_ALL_QUANTS 对照与 q4-direct 定责（2026-09-27 09:xx）

本地提交 **`b2b2c48fa`**（`fix(sm70-attn): q4-direct batch offset; hard-disable that path`）。
二进制已重新部署（快照 `work/bin-backup/20260927-091303-full`），生产服务验证：12K 探针
prefill 556.1 t/s、解码 **51.7 t/s**（MTP 生效）。

### 10.1 `GGML_CUDA_FA_ALL_QUANTS`（官方 KNOWN_ISSUES 建议量化 KV + FA 时打开）：**实测没用，保持 OFF**

用同一个构建，`-DGGML_CUDA_FA_ALL_QUANTS=ON` 重编后重跑（q8_0 KV，生产形状，GPU 空闲）：

| 指标 | OFF（部署版） | ON |
|---|---|---|
| PPL q8_0 ch1（sm70 on / stock） | 12.2796 / 12.2854 | 12.2796 / 12.2854 |
| PPL q8_0 ch2（sm70 on / stock） | 7.1242 / 7.1260 | 7.1242 / 7.1260 |
| pp2048（sm70 on / stock） | 1107.4 / 1097.3 | 1101.8 / 1093.8 |
| pp16384（sm70 on / stock） | 1025.9 / 949.9 | 1022.8 / 949.2 |

**PPL 逐位相同，速度在噪声内略负**。原因是我们这条链路本来就不依赖它：q8_0 KV 的 prefill 由
sm70 内核接管，解码走 vec/tile 内核（该 fork 的 tile 内核已经支持 q8_0 直读），stock 与 sm70 的
PPL 本来就一致。结论：不打开，构建保持 OFF（已回退并重建）。

### 10.2 q4-direct：**在我们的集成里是错的，硬关掉**

`-ctk/ctv q4_0 -fa on`，同一模型（`-c 512 --chunks 1/2`）：

| 路径 | PPL chunks=1 | PPL chunks=2 |
|---|---|---|
| stock（`LLAMA_SM70_D256=0`） | 12.5431 | 7.2164 |
| sm70 **staged f16 镜像** | 12.6327 | 7.2429 |
| sm70 **q4-direct**（`LLAMA_SM70_D256_Q4_DIRECT=1`） | **105.27** | **54.26** |

也就是说：

* **staged 镜像对 q4_0 KV 是正确的**（与 stock 的差异在 ±2.4 / ±0.88 误差内，和 q8_0/f16 的表现一致）；
* q4-direct（内核里直接读 q4_0 原始 block）在这个集成里数值全错——不只是多 stream 偏移的问题
  （单 stream 也错），说明它搬过来时依赖的上游布局/预处理与我们这棵树不一致。

处理：给 q4-direct 分支补上了 per-stream 偏移（内核），同时把 `sm70_q4_direct()` **写死返回 false**
（保留代码与注释，env 不再能打开）。验证：`LLAMA_SM70_D256_Q4_DIRECT=1` 时现在得到
7.2429 / 12.6327，即自动退回正确的镜像路径。

顺带说明：生产用的是 q8_0 KV，这条路径本来就不参与；关掉的目的是拆掉"某天有人把 env 打开就得到
乱码输出"的地雷。

## 十一、MTP 头三元化（PTQ1_0）试验：**否决**（2026-09-27 09:4x）

动机：MTP 的 catch-up 要 0.85 ms/prompt-token（≈ 1 TFLOPS 量级，像在跑 mat-vec），而我们的
sm_70 优化全压在 PTQ1_0 上；头的矩阵是 **Q8_0**。如果把它换成 PTQ1_0（430 MB → 93 MB），
理论上既省显存又让 catch-up 走快路径。

做法（工具：`work\tmp\mtp-headtool.py` 抽取/回植 + `llama-quantize`）：

1. `extract`：把 `blk.64.*`（15 个）抽成独立小 GGUF。**必须连 `output.weight` 一起抽**，
   否则 loader 会报 `a tied Hadamard output requires version 2 and tied_output=true`
   （`src/llama-model.cpp:1354`：KV 里 token_embd 绑了 Hadamard 反变换、而文件里没有
   output.weight 时就会触发）。
2. `llama-quantize --allow-requantize head.gguf head-ptq.gguf PTQ1_0`。
   注意两点：(a) 这个版本 `type` 是**位置参数**，不是 `--type`；(b) `--tensor-type` 覆盖只在
   "默认类型是量化类型" 时生效（`src/llama-quant.cpp:697` 的 `if (ggml_is_quantized(default_type))`），
   所以 `--type COPY` + 规则这条路走不通，只能走"抽出来单独量化"。
   另：PTQ1_0 的 ggml 类型号是 **143**（块 128 / 28 B）。
3. `regraft`：把量化后的 15 个张量换回主文件 → `...-f16ssm-MTP-ptqhead.gguf`（5.62 GiB）。

实测（131072 ctx / q8_0 KV / V100，两个 arm 背靠背）：

| | 显存 | 12K prefill | 12K 解码 | spec-bench 解码 | 接受率 |
|---|---|---|---|---|---|
| Q8 头（现状） | 14548 MiB | 553.5 t/s | 51.4 t/s | 53.5~66.8 t/s | 60~91% |
| **PTQ1_0 头** | 14232 MiB | 594.3 t/s | **26.1 t/s** | 30.5~32.0 t/s | **8~12%** |

**结论：否决。** 头的接受率从 ~70% 掉到 ~10%（草稿数从 212 涨到 483），解码腰斩——MTP 头是
BF16 训练 + Q8 QAT 出来的，1.75 bpw 三元化直接把它推到分布外（不像主干是原生三值训练的）。
同时 prefill 只快了 7%，说明 **catch-up 的瓶颈不是头权重的格式**（否则换格式后应该大幅提速），
更像是每行一次图/发射开销——真要治得动 speculative.cpp 里的 catch-up 批处理，属于运行时改造，
不是换量化格式能解决的。

处置：生产保持 Q8 头（`Ternary-Bonsai-2-27B-PTQ1_0-f16ssm-MTP.gguf`，已验证 command line 里
`-m ...MTP.gguf` + `--spec-type draft-mtp` ✓，显存 14.6 GB）；三个实验文件（头 Q8/PTQ + ptqhead
模型，共 7.4 GB）已移到 `work\tmp\mtp-head-exp\`，模型目录回到干净状态（面板下拉框不再被污染）。
脚本保留：`work\tmp\mtp-headtool.py`、`mtp-head-quant.ps1`、`mtp-head-ptq-ab.ps1`。

## 十二、MTP 的 prefill 代价：找到真凶（草稿 micro-batch = 4）并修掉（2026-09-27 10:xx）

本地提交 **`9a7b871a1`**，二进制已部署（快照 `work/bin-backup/20260927-101803-full`）。

### 12.1 埋点先落地（`SPC_MTP_PROF=1`）

给 `common/speculative.cpp` 的 draft-mtp 三处 `llama_decode(ctx_dft)` 加了计时/行数统计
（只在 `SPC_MTP_PROF=1` 时生效，每处理一个大 batch 打一行进度；析构时再打一行汇总）。
12K 提示的实测输出：

```
[mtp-prof] rows(process)=11039 draft decodes=2 rows=8234 5456.8 ms (0.6627 ms/row)
[mtp-prof] rows(process)=12063 draft decodes=3 rows=11039 7349.5 ms (0.6658 ms/row)
```

即：**catch-up 是一次 8192 行的大 batch**（不是逐行），但每行 0.66-0.70 ms，与 batch
大小、`-ub`（256/1024/2048）、头权重格式（Q8_0 / PTQ1_0 / F16）都无关；同一时段用
`nvidia-smi` 每秒采样，MTP 那 10 秒里有 12 个样本停在 **30-45%** 利用率（基线是全 100%），
即 GPU 约 60% 时间在空转 ⇒ **发射/依赖受限，不是算力受限**。

（中间踩的坑：用 nsys 抓那两次的 profile 其实**没启用 MTP**（解码速度=基线、日志里没有
draft 行），所以"内核表完全一致"是假象，不能当结论。）

### 12.2 真凶

`common/speculative.cpp` 的 `common_speculative_draft_n_ubatch()`：

```cpp
const int64_t step_rows   = 1 + n_max;                  // 3
const int64_t window_rows = need_n_rs_seq() + 2;        // 4
return min(params.n_ubatch, max(step_rows, window_rows) * n_parallel);   // = 4
```

草稿上下文只按"一步草稿"的宽度分配 micro-batch = **4 行**。catch-up 一次给 8192 行，
于是被拆成 **2048 次图执行**（每次 ~20 个 kernel）→ 每行 ~0.7 ms 的发射开销 ✓ 完美吻合。

### 12.3 修法与实测（12K 提示，q8_0 KV，MTP n_max=2）

```cpp
static const int64_t min_ubatch = [] { const char * e = getenv("GGML_SPEC_DRAFT_UBATCH");
                                       return e ? atoll(e) : 256; }();
const int64_t want = std::max<int64_t>(rows, min_ubatch);   // rows 仍保底 rollback 窗口
return min(params.n_ubatch, want);
```

| 草稿 ub | 4（原） | 64 | 128 | **256（新默认）** | 512 | 1024 |
|---|---|---|---|---|---|---|
| 12K prefill | 522 t/s | 757 | 781 | **802** | 811 | 695（显存压力） |
| 显存 | 14522 MiB | 14690 | 14735 | **14806** | 14940 | 15247 |
| 12K 深度解码 | 47.9 | 50.1 | 49.3 | **52.1** | 49.4 | 41.9 |
| 接受率（四任务） | 91.0/59.9/73.9/70.7% | 同 | 同 | **同** | 同 | 同 |

正确性：贪心输出 **MTP 开 / 关逐字节一致**（sha256 `A5D8F229552032FF`，160 token，草稿
142 提出 / 88 接受）；前缀缓存不受影响（cached prompt 259 ms）。

生产验证（131072 ctx + 视觉塔）：12K 探针 **prefill 802.8 t/s、冷启动 15.1 s（原 22.7 s）**、
解码 43-52 t/s、显存 14825 MiB。相对无 MTP 的 944 t/s，prefill 代价从 **−45% 降到 −15%**。
`GGML_SPEC_DRAFT_UBATCH` 可继续调（显存换 prefill）。

### 12.4 还没做的

* 剩下的 0.19 ms/row 是 MTP 头真实的前向成本（1 层 / 425M 参数）。要再快就得让
  Q8_0/PTQ1_0 的大 batch matmul 在 Volta 上走 MMQ（现在 `ggml_cuda_should_use_mmq` 里
  PTQ1_0 要求 Turing+，我们只能走 dequant→F16 GEMM）——那是 CUDA 后端的大工程。

## 十三、"给 Volta 开 MMQ"这条撤销 + MTP 配置扫描（2026-09-27 11:xx）

### 13.1 MMQ 不是剩余杠杆（结论撤销）

先查了策略代码：`ggml_cuda_should_use_mmq()` 对 NVIDIA 的最终判定是

```cpp
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;  // 64
}
```

也就是**故意**让"有 FP16 张量核心 + 大 batch"走 dequant→FP16 GEMM（dp4a 只在列数 <64 时更快）。
实测整干 prefill 的 12.5 s 里 CUTLASS FP16 GEMM 7.34 s、PTQ1_0 反量化 1.31 s —— 换算下来
**有效算力 ≈54 TFLOPS**（V100 fp16 TC 实际可用的量级），说明这条路径本来就是对的。
另外 `turing_mma_available` 那道门只影响 PTQ1_0 在 **2..63 列** 的小 batch（解码走 MMVQ、
大 batch 走 FP16 GEMM），而 MTP 头是 Q8_0（本来就走 dp4a MMQ，`ne11<64` 分支）。
所以"给 sm_70 写 PTQ1_0 MMQ"既不会被调度器选中、也不会更快 —— **撤销该方向**。

### 13.2 MTP 配置扫描（n_max × 草稿 KV 类型）：现有默认已是最优

5 个 arm，都是 131072 ctx / q8_0 KV / 草稿 ubatch=256（`work\tmp\mtp-nmax-sweep.ps1`）：

| arm | 显存 | 12K prefill | 12K 解码 | spec-bench 解码均值 | 接受率 |
|---|---|---|---|---|---|
| n_max=1, ctkv=q8_0 | 14692 | 851 | 46.9 | 58.5 | 95/72/82/82% |
| **n_max=2, ctkv=q8_0（现状）** | 14874 | 848 | **53.4** | **61.4** | 91/60/74/71% |
| n_max=3, ctkv=q8_0 | 14987 | 839 | 53.3 | 58.7 | 77/49/63/65% |
| n_max=2, ctkv=q4_0 | 14695 | 846 | 49.9 | 61.5 | 90/59/74/72% |
| n_max=3, ctkv=q4_0 | 14809 | 850 | 52.8 | 58.5 | 74/49/63/62% |

* `n_max=3` 明显更差（接受率掉到 49~77%，白跑草稿）；`n_max=1` 覆盖不足。
* 草稿 KV 换 q4_0 只省 ~180 MB 显存，12K 深度解码掉 7%（49.9 vs 53.4）→ 不值。
* 结论：**生产配置不动**（`--spec-draft-n-max 2` + `-ctkd/-ctvd q8_0`），而且 5 个档位的
  prefill 都在 838~851 t/s，再次确认第 12 节的 ubatch 修复是稳的（修复前 522）。

## 十四、第二轮检索：新项目与新论文（2026-09-27 12:xx）

检索途径：GitHub 连接器（仓库搜索 + raw README/patch 全文）、HuggingFace 论文索引 API
（arXiv 官方 API 在本机被网关拦，返回 0 字节；HF 的 `/api/papers/search` 可用）。
以下都是第一轮（第 1~7 节）**没有**的条目。

### 14.1 新项目

| 项目 | 是什么 | 能抄什么 |
|---|---|---|
| [WyvernTKC/llama.cpp-4xV100](https://github.com/WyvernTKC/llama.cpp-4xV100)（分支 `sm-tensor-4xv100`） | **4×V100-SXM2-32GB + Windows** 的 llama.cpp 分支：张量并行 `-sm tensor`（列切 + all-reduce）、以及**比显存大的 MoE**（专家权重留主机内存、按 token 流式搬运 + `GGML_META_EXPERT_CACHE` 设备侧专家缓存）。156 GiB 模型在 128 GiB 显存上 15.7 → 42.2 t/s | ① **Volta FA 调优三连**：`028c1a96b` 针对 head size 256 重调 FA MMA 配置（去掉 ~3 KiB/线程的寄存器溢出）、`ba3088a16` DV=512 拆分累加、`eca0bd8b9` 修 FA MMA combine 里的**分歧 `__syncthreads`**（compute-sanitizer 判为 UB）。② 一堆方法论：nsys 只对 `llama-server` 有效（`llama-cli` 的 CUDA 工作全在子进程里）、`--no-warmup` 会让 CUDA graph 建不起来从而把 GPU-bound 误判成 host-bound、`cmake --build` 可能 exit 0 却有编译错误、**"输出变成同一个字反复出现"时先看是哪个字**（Qwen 的 token 0 是 `!`，NaN logits 会让 argmax 恒返回 0）。③ 他们实测 `GGML_CUDA_CUBLAS_COMPUTE_TYPE` 单独就能让 gemma4 的 PPL 动 3.4%（见 14.3，我们在自己模型上验证了**没有**这个效应）。 |
| [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)（+ [Windows/V100 整合包](https://github.com/liuzhuohua/kvmem-llama.cpp-Windows-V100)） | **KV 分层**：完成的 KV 块存主机内存，按当前**用户查询**检索相关块（128-token 块）放进有上限的 GPU 常驻窗口；同名论文见 14.2。对 Qwen3.8-27B（**我们同款**）在 16 GiB 卡上把逻辑工作区做到 **256K**（他们的卡是 5060 Ti 16GB；Windows 版专门做了 CUDA 12.9 的 V100 构建） | ① 直接对应我们"16GB 跑 131072"的上限：**要么**走他们的检索窗口（论文说只留 32K 常驻、≤256K 查询近乎无损），**要么**走他们对比的 Raymond Huang 式"全量 KV 流式"（保持全注意力、按层预取、代价是 PCIe 流量）。② 他们处理了 **ReplaySSM**（混合模型的循环状态重放）——我们模型也是 GDN 混合，任何 KV 窗口方案都绕不开这一步。③ 实现方式是 `llama_memory_i` 适配器 + patches（pin `b81c99b`），工程量大但接口干净。④ 文档还点名：`GGML_CUDA_FA_ALL_QUANTS=ON` 是 `--kv-dtype q5_0` 在混合模型上的前提（我们用 q8_0，不受影响）。 |
| [jackinthebox52/qwen38-v100-serve](https://github.com/jackinthebox52/qwen38-v100-serve) | **同款模型（Qwen3.8-27B）在 V100 上的生产栈**（Linux 脚本 + 两套二进制：MTP 用 stock、非投机用打了 T2-001 的补丁）。128K 上下文：stock 16.49 → T2-001 23.83（+44.9%）→ MTP 42.22 t/s（+156%） | **T2-001 补丁**：给 `flash_attn_ext_vec` 加 **`ncols2 = 3`** 的 GQA 打包（我们 24 Q / 4 KV、gqa=6，stock 只按 2 的幂打包 ⇒ 每个 KV 头被读 3 遍）。128K 解码的 KV 流量 26.37 → 8.59 GB/token，**已逐位验证**（KLD 0、top-1 100%）。**但**：作者把打包限制在 **F16/BF16 KV**（量化 KV 路径在 D=256 已经在 252~255 寄存器，打包后主循环里溢出 30 STL + 17 LDL）⇒ 我们用 q8_0 KV **吃不到**；除非把 KV 升到 F16（128K 下 KV 从 ~4.5 GB 变 ~8.5 GB，16GB 卡装不下）。另外它的"运维陷阱"清单与我们踩过的一致（CUDA ≤12.9、驱动 ≤580、`-np 1`、时钟锁定、prompt cache 决定多轮墙钟）。 |
| [xormal/V100-SM_70-Flash-Attn-2-v1](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1)（"HOOLIGAN"） | **Volta 全套注意力内核包**（PyTorch 扩展）：prefill/decode/backward、**int8/int4 KV**、**MXFP4**（gpt-oss 的格式，stock vLLM 直接拒绝）、分页 KV、滑窗/双侧窗、softcap、ALiBi、MLA/latent attention，以及一个 **vLLM `FA2_SM70` 一等后端** | ① **"把 int8 编码进 fp16 尾数"**：`0x6400|u` 就是 `1024+u`，把常量偏移折进 rank-1 修正 —— 于是"在 sm_70 上没有 int8 张量指令"也能让 byte 内核**比 fp16 还快**（1.94~2.35× vs SDPA-eff）。这条对我们是"KV 换更低位宽"的思路来源。② **deferred rescale**：用 Cauchy–Schwarz 上界 `U` 去掉 online softmax 的串行依赖（证明 `O` 与 `U` 无关），fp16 下 bit-exact 且 1.04~1.8×。③ **split 数按 CTA 数定，不按块大小定**（六个形状里五个都要 16 splits ⇒ `B·H·splits ≈ 512`，即 ~6.4 CTAs/SM）。④ 一条正好呼应我们第 13 节教训的话：他们把 byte-KV 接进服务端后**端到端只差 0.7%**，因为 attention 只占 131K prefill 的 16.3%——"**接内核之前先量份额**"。 |
| [gilby/volta-nvfp4](https://github.com/gilby/volta-nvfp4)（v100-skinny 的 MoE 分支） | 把 **NVFP4 MoE** 搬到 Volta（QPN2-MoE：1.538× vs stock Marlin-MoE，byte-identical；Qwen3.6-35B-A3B 单卡 99.4 tok/s）。测试机是 **6×V100-PCIE-32GB、无 NVLink、200 W 限功率** | ① 一个重要发现其实是**配置 bug**：1Cat-vLLM 把 `cudagraph_capture_sizes` 钉死 `[1,2]`，导致 N≥3 的并发直接掉进 eager 路径 —— 修复后 N=4 从 39.9 → 297.7 tok/s（7.46×），功耗从 53.9 W 升到 84.7 W（"GPU 被饿着"而不是"跑满"）。对我们 llama.cpp 的启示：**并发/宽 batch 时先确认 CUDA graph 真的被复用了**。② "≥95% 的并发轮时间在 NVFP4 GEMM 之外，剩余目标是 GDN/线性注意力"——和我们 nsys 里 GDN 占 prefill 7.2% 的观察同向。③ 功率实验：200 W 对 250 W 无差别（峰值只到 83 W、SM 时钟已 1380 MHz 满频）。 |
| [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | **我们同款模型的非均匀量化**（IQ2_S/IQ2_XS/IQ3_XXS/IQ3_S，**每个都带 `-mtp` 版本**，即官方 MTP 头保留），配视觉投影 | 想换量化口味时的现成候选（我们目前是 PTQ1_0 1.75 bpw + 自己嫁接的 on-policy MTP 头）。注意它用的是标准 IQ 类型 ⇒ llama.cpp 原生支持，不需要我们打任何内核补丁。 |

### 14.2 新论文

**三值/超低位**（都能用于"再压主干或草稿头"）

| ID | 标题 | 与本项目的关系 |
|---|---|---|
| `2606.26650` | CAT-Q: Cost-efficient and Accurate Ternary Quantization | **PTQ 三值**（不靠昂贵的 QAT），正好对应我们"Q8 草稿头三元化会崩"的那类需求 |
| `2601.07892` | Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification | 1.25-bit 三值 + 细粒度稀疏；核心论点是"现有三值实现与商用硬件的 2-bit 对齐打包错配"——打包/内核视角 |
| `2509.23809` | Tequila: Trapping-free Ternary Quantization | 三值训练的"陷阱"问题（若要继续训三值） |
| `2603.27914` | ITQ3_S: High-Fidelity 3-bit via Interleaved Ternary Quantization with Rotation-Domain Smoothing | 交错三值 + 旋转域平滑（我们的权重已经带 Hadamard 旋转） |
| `2402.10076` | QUICK: Quantization-aware Interleaving and Conflict-free Kernel | 量化布局与"无冲突内核" |
| `2512.06443` | Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference | 1.58-bit 的查表推理路线（GPU 侧的 LUT 并行与它的瓶颈） |

**投机/MTP**

| ID | 标题 | 关系 |
|---|---|---|
| `2509.18362` | **FastMTP**: Accelerating LLM Inference with Enhanced Multi-Token Prediction | 我们之前用的 FastMTP 草稿模型背后的论文（现在是我们的历史，不是现状） |
| `2601.11580` | **Speculative Decoding: Performance or Illusion?** | 生产级引擎（vLLM）上对 SD 的系统性复核——给"MTP 一定更快"泼冷水的证据面 |
| `2607.25852` | AngelSpec | 生产环境投机解码 |
| `2602.06019` | Multi-Token Prediction via Self-Distillation | 自蒸馏训 MTP 头（自训草稿头的路线） |

**Volta/低资源 GPU 与 MoE**

| ID | 标题 | 关系 |
|---|---|---|
| `2410.16663` | FastAttention: Extend FlashAttention2 to NPUs and Low-resource GPUs | FA2 向"低端 GPU"扩展（我们 FA 那节的补充文献） |
| `2508.17467` | MoE-Inference-Bench | 跨硬件 MoE 推理评测（含老卡） |
| `2503.09716` | MoE-Gen | 单卡 MoE 高吞吐（模块化 batching，与 WyvernTKC 的专家缓存同向） |

**长上下文 / agent KV（第二轮最大的收获面）**

| ID | 标题 | 关系 |
|---|---|---|
| `2609.04852` | **KVMem**: Virtualizing Million-Token Agent Workspaces on a Consumer GPU | 见 14.1 的项目；论文结论："≤256K 查询下只留 **32K GPU 常驻** 近乎无损"（LongMemEval-S 85.6 vs 86.6、AgentLongBench 60.9 vs 59.5） |
| — | Fathom: Per-Query Read Depth for Sparse Decoding over **Offloaded KV Caches** | 2026-09；主机内存 KV + 按查询决定"读多深"的稀疏解码（KVMem 的下一步） |
| — | UltraQuant: 4-bit KV Caching for Context-Heavy Agents | 2026-06；agent 场景的 4-bit KV（我们目前 q8_0） |
| — | Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention | 2026-08；保留性 KV 压缩 + 混合格式分页注意力 |
| — | RedKnot / CacheWise / KVFlow / Execution-State Capsules | 长上下文服务与 agent 的 KV 复用/检查点系列 |

**混合线性注意力（我们模型的层类型）**

| ID | 标题 | 关系 |
|---|---|---|
| `2605.22791` | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | GDN 的后续（含 Kimi Delta Attention）；若未来 Bonsai 换代，这是内核要跟的架构 |
| `2608.11805` | Hybrid Gated Attention | 2026-08 |
| `2607.07953` | Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing | 综述性质，做取舍时参考 |

### 14.3 本轮顺手验证的两件事

**(a) cuBLAS compute type 对我们没有质量红利（否证）。** 我们的 prefill 走
`ggml_cuda_mul_mat_cublas`：量化权重 + 有快 FP16 ⇒ `compute_type = F16`（`ggml-cuda.cu:1627`）。
WyvernTKC 报告 gemma4 的 PPL 单靠这个开关就动 3.4%（默认 1.8139 vs 强制 f32 1.7549）。
我们同二进制 A/B（q8_0 KV、chunks 2、pp2048）：

| compute type | PPL | pp2048 |
|---|---|---|
| 默认（F16） | 7.1242 ± 0.86 | **1133.1 ± 12.1** |
| 强制 f32 | 7.1261 ± 0.86 | **266.9 ± 0.9**（**−76%**） |

⇒ **我们这条路径没有可捡的质量**，保持默认。（脚本 `work\tmp\cublas-compute-ab.ps1`）

**(b) T2-001 补丁评估**：全文已取（`ggml/src/ggml-cuda/fattn-vec.cuh`，71+/21−），
几何与我们完全一致，但作者明确把 `ncols2=3` 限制在 **F16/BF16 KV**；我们生产是 q8_0 KV，
这条吃不到（除非牺牲 KV 显存换 F16）。留档。

### 14.4 按 ROI 的下一步

1. **KVMem 评估（最高价值）**：我们要不要"16GB 上 256K"？两条路——检索窗口（省显存、但
   Codex 这类"要精读整份上下文"的负载存疑）或全量流式（保真、吃 PCIe）。需要先读他们的
   `patches/` 与 `docs/architecture.md`，评估移植到我们 prism 树的成本（他们有 ReplaySSM，
   我们有 GDN，正好对口）。
2. **Volta FA 三连移植**（成本低）：把 WyvernTKC 的 hs=256 寄存器溢出修复 / DV=512 拆分 /
   syncthreads 修复，对照我们树里 **stock FA mma** 路径取用（我们的 prefill 走 sm70-attn、
   decode 走 vec，需先确认这些改动是否落在我们实际触发的内核上）。
3. **CAT-Q / Sherry**：如果哪天想再压主干或重训草稿头（我们已知 Q8 头三元化会崩，这两篇是
   "PTQ 三值"的最新方法）。
4. 观察项：GSQ-RCO（同款模型的另一套量化）、Fathom/UltraQuant（agent KV 的最新进展）。

## 十五、KVMem 评估：结论是"不用移植"，因为 256K 今天就能跑（2026-09-27 13:xx）

### 15.1 KVMem 是什么（读了它的 `docs/architecture.md` 与补丁清单）

块稀疏 KV 工作集管理器：GPU 侧只有 `budget + gen_reserve` 个 **128-token 槽**的池，其余 KV 进主机内存
（NVMe 未实现），每个 agent step 按**最后一条 `role=user` 的消息**检索相关块塞回池子。
关键实现细节（决定了它能不能嫁接到我们树）：

* **不做 re-RoPE**：槽里存的位置就是**原始 token 位置**，恢复是"按原 pos 的 packed GPU 格式 memcpy"，
  所以 FA 与 RoPE 都不用改；检索打分用的是写缓存时捕获的 **pre-RoPE mean-K**。
* MTP：草稿上下文拿一个**同尺寸的 follower 槽池**（同 block id / slot 索引）；GDN 的循环状态**不进槽**，
  由 **ReplaySSM**（FP32 GDN Record/Fold）负责。
* 混合模型、MTP、视觉、Windows/V100 构建它们都做了（`kvmem` 的 Windows 包就是 CUDA 12.9 的 V100 版）。

**移植成本**：llama.cpp 侧 `patches/llama-kvmem-current.patch` = **776 插入 / 152 删除、30 个文件**
（`llama-kv-cache.cpp` +106、`llama-graph.cpp` +95、`llama-memory-recurrent.cpp` +83、
`models/delta-net-base.cpp` +33、`models/qwen35.cpp`、`server-common.cpp` 163 行重构…），
基点上游 `b81c99b`；外加自带适配层 ~120 KB（`llama-memory-kvmem.cpp` 单文件 157 KB）+ 34 KB CUDA stage-in
内核。我们是 prism 派生树，需要逐块重放。

**它能给什么 / 代价**：16 GB 卡上 256K 逻辑工作区（他们实测 prefill 437–463 t/s、解码 31–33 t/s @MTP3），
但那 256K 里**只有约 52K 参与注意力**（窗口 + 检索）；v1 还有硬限制：**单次生成不能超过
`--kvmem-gen-reserve`（16K）**，超过直接 `llama_decode(gen) failed rc=1`。

### 15.2 结论：不移植——我们已经能跑 262144 的**完整**窗口，而且是精确全注意力

模型训练上限就是 262144（`qwen35.context_length`），之前只跑到 131072 纯粹是 **q8_0 KV 太占显存**。
换成 **q4_0**（每 token 18.4 KB，262144 全量 4.8 GB）就全解决了：

**质量**（`work\tmp\kv4-ppl-long.ps1`，49K token 真实语料拼接，误差 ±0.07）：

| KV 配置 | PPL | Δ |
|---|---|---|
| q8_0/q8_0 | **4.6649 ± 0.069** | — |
| q4_0/q4_0 | 4.6703 ± 0.070 | **+0.12%（统计上测不出）** |
| q4_0 + 偏置(`-rot`) | 4.6738 | +0.19% |
| q4_0 + 偏置(`-norot` + `LLAMA_ATTN_ROT_DISABLE=1`) | 4.6787 | +0.30% |

⇒ **q4_0 的代价测不出来**；而**"KV 均值中心化矫正"在我们模型上没有收益**（不但没降 PPL，还略升）——
和官方 `KV-CACHE.md` 的期待相反，值得记一笔（两份校准文件都是早前会话用
`llama-kv-mean-center` 生成的，`-rot`/`-norot` 分别对应旋转开/关）。

**速度与容量**（100K token 探针，`work\tmp\kv4-deep-probe.ps1` / `k224-mtp-test.ps1`）：

| 配置 | 显存 | 100K prefill | 100K 解码 |
|---|---|---|---|
| 128K + q8_0 + **MTP** | 15123 MiB | 612.9 t/s | **25.8 t/s** |
| **256K + q4_0 + 无 MTP** | **14326 MiB** | **692.1 t/s（+13%）** | 19.6 t/s |
| 224K + q4_0 + MTP | 14875 MiB | 612.7 t/s | 20.2 t/s |
| 256K + q8_0 | — | **装不下**（`common_fit_params` 拒绝） | — |
| 256K + q4_0 + MTP | — | **装不下**（草稿 KV 也 q4_0 时仍不行） | — |

**MTP 与 q4_0 互斥的机理**：q8_0 时 MTP 让 100K 深度解码 19.8 → 25.8（+30%）；q4_0 时只有
19.6 → 20.2（**+3%**）——因为 q4_0 的 vec 内核要在线反量化，而投机验证的 4 行也是这个内核。
所以两个方向本来就该二选一，不是"叠加越好"。

**落地**：面板的「极限 256K · q4_0」预设原来**没关 MTP**（MTP 开着点下去必然加载失败），已修成
自动关 MTP 并把上面这些实测数字写进按钮提示（`outputs/bonsai-dashboard.py`）。

### 15.3 什么时候再回头看 KVMem

* 目标是 **>262144**（超出训练窗口）或"几百 K 历史 + 少量事实"的检索型 agent 时——那时它的
  block-slot 池 + ReplaySSM 是现成设计（我们已经有能跑通的 256K 全注意力打底）。
* 顺带发现：它仓库里的 `docs/prefill-harvest-optimization.md`（67 KB）是**纯 llama.cpp prefill 优化
  笔记**，与 KVMem 机制无关，值得单独翻一遍（另一条低风险的借鉴来源）。

## 十六、读完那份 67 KB 笔记的结论：**没有可抄的性能项，只有 4 条注意事项**（2026-09-27 14:xx）

先纠正我第 15 节的判断：`docs/prefill-harvest-optimization.md` **不是**通用 prefill 优化笔记，
而是 **KVMem 自己 harvest 流水线的设计文档**（把"每个 ubatch 主机同步 → D2D/D2H → pack → 256 次同步
64 KiB pwrite"改成 fence + F16 存储 + worker 合并写）。它针对的是他们特有的 NVMe 收割开销，
我们不做 harvest，所以**没有可抄的性能改动**。可迁移的是 5 条事实：

| # | 事实（他们的证据） | 对我们的含义 |
|---|---|---|
| 1 | `ggml_backend_cuda_buffer_set_tensor` 在 **`cudaStreamPerThread`** 上拷贝并只同步该流（`ggml-cuda.cu:787`）；llama.cpp 仅在 `cparams.pipeline_parallel` 为真时才在 `set_inputs` 前主机同步 ⇒ 去掉同步会让 ubatch N+1 的 `set_input` 抢跑 graph N（他们标 P0） | 我们没开 `pipeline_parallel`（默认 false）⇒ 安全；**将来若要做 ubatch 重叠必须先处理这个** |
| 2 | `ggml_backend_tensor_get[_async]` 同样只同步 PerThread 流，**不等 ggml 计算流** ⇒ 他们的 MTP D2H 一旦去掉主机同步就会读到脏数据（P0） | 我们的 MTP 也走 `ggml_backend_tensor_get_async(backend_h, t_h_nextn, ...)`（`llama-context.cpp:1773/2210`）——但**贪心一致性测试是逐字节相同的**（第 12 节，160 token），说明这条路径实际是安全的；**以后动 MTP 管线要拿这个测试当闸门** |
| 3 | ubatch 太小是悬崖：他们引用 qw3 的实测"塌到 16/32 token 的 chunk 是 ~7× prefill 悬崖"，qw3 的 `effective_prefill_chunk_size` 默认 **2048** | 与我们自己的调参结论一致（多轮会话 1024 最优、单发超长提示 2048）——面板注释里已写，无新增 |
| 4 | `GGML_CUDA_GRAPHS=ON` 是默认；graph capture 在 `graph_compute` 返回前就结束，**在 ubatch 之间插 `cudaStreamWaitEvent` 会和下一次 `BeginCapture` 互相干扰** | 我们实测关掉 CUDA graph 明显更慢（decode 50→29.5 t/s），所以保持开启；若将来加 ubatch 重叠要专门验证 capture |
| 5 | 他们的 **slot 池让 `n_kv` 恒定** ⇒ prefill 图形状稳定 ⇒ 可被 CUDA graph 复用；我们的 K/V 视图按当前长度切片（`K->ne[1] = kv_len`），形状每个 ubatch 都变 ⇒ 图形无法跨 ubatch 复用 | 理论上是我们 prefill 的一个潜在优化点，但**实测证明不需要**（见下） |

### 16.1 顺手量了：我们的 prefill 不是发射受限（`work\tmp\prefill-util-probe.ps1`）

动机：他们那次 256K prefill 只有 **SM 24% / mem 5%**，诊断是"主机空转 + 上千次 ggml 小启动"。
我们用同样的方法（`nvidia-smi` 每秒采样）量了自己的生产配置：

| 请求 | prefill | SM 平均 | ≥80% 样本 | 30–79% | <30% | 显存控制器平均 |
|---|---|---|---|---|---|---|
| 12K 提示（+16 token 解码） | 860.4 t/s / 14.0 s | 64.6% | 10 | 3 | 6（含解码尾部） | 26.8% |
| **100K 提示**（+16 token 解码） | **646.9 t/s / 154.3 s** | **89.1%** | **140 / 159** | 5 | 14 | 29.2% |

⇒ 我们**不是**发射/间隙受限（他们 24% vs 我们 89%），显存控制器也只有 ~29% 利用率，
瓶颈确实是 dequant→FP16 GEMM 本身（换算 ≈54 TFLOPS ≈ V100 实用上限，见第 13 节）。
所以第 5 条"稳定形状 → CUDA graph"这条**不值得投入**（收益空间在我们这里不存在）。

顺带一个横向对照：他们的 260012-token prefill 是 **197.5 t/s / 峰值 18122 MiB（RTX 5090）**；
我们 99821 token 是 **646.9 t/s（V100）**——老卡快 3.3×/token，差距来自他们那条 harvest 管线，不是内核。

**结论**：这条线到此为止（可抄性能项 0 条，注意事项 4 条已记）。剩下的性能空间仍在 FA/GDN 内核侧
（第 13 节的 MMQ 结论：大 batch 已走 FP16 GEMM，dp4a 只在 <64 列有意义）。

---

## 十七、Volta FA 三连移植（WyvernTKC/llama.cpp-4xV100）：用了 2 条，验证后已上线（2026-09-27 15:xx）

来源是 `work\tmp\wyvern-*.patch` 里那三个 commit。我们的模型 head_dim=256，所以**第三条（DV=512）用不到**，
只拿了前两条：

| # | 来源 | 内容 | 是否应用 |
|---|---|---|---|
| 1 | `028c1a96b` | Volta hs=256 的 MMA FA 配置重调 | ✅ 已应用（我们原来只有 512/512、576/512 四条，hs=256 落到 Ampere 兜底） |
| 2 | `eca0bd8b9` = 上游 [ggml-org/llama.cpp#27955](https://github.com/ggml-org/llama.cpp/pull/27955) | np>1 combine 块内 `__syncthreads()` 分歧（UB）修复 | ✅ 已应用 |
| 3 | 该仓库第三个 commit | DV=512 的配置 | ❌ 不适用（我们没有 DV=512 的注意力层） |

**为什么 hs=256 配置对我们仍相关**：D=256 的 prefill 由 sm70-attn 内核接管，但 **q_len 17..255 的小批**
（增量续写、草稿验证）、**mask-less 注意力**、以及非 256 head 的图仍然走 `fattn-mma-f16`，
而这些形状此前正落在 Ampere 配置上（Q 常驻寄存器 → DV=256 时 VKQ 累加器 + Q 超过 255 寄存器上限，
ptxas 溢栈 ~3.3 KiB 存 / 2.9 KiB 载每线程，本地内存流量 261 MB > K/V 全局流量 128 MB）。
新配置 `Q_in_reg=false` + `nbatch_fa/nbatch_K2` 缩小 ⇒ 溢栈归零、每 SM 仍驻 2 块
（他们 V100 上 hs=256 kv=20000 nb=512 是 20078 → 14770 µs，Qwen3.6-27B pp4096 +7.7%）。

### 17.1 我们自己的回归证据

| 项 | 改动前 | 改动后 | 判定 |
|---|---|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | 2/2 backends passed（not supported 1688、CUDA error=2） | 逐项相同 | ✅ 无新增失败 |
| PPL 50K 语料 / q8_0 KV / 12 chunks | 4.6649 ± 0.069 | **4.6649 ± 0.069** | ✅ 逐位相同 |
| pp16384（b 8192 / ub 1024 / q8_0） | ~990–1025 量级 | 966.37 ± 1.81 | ✅ 区间内 |

单看探针会误判：回归脚本里 12K 只有 805.9 t/s、100K 605.2 t/s，比历史最好的 860 / 647 低 ~6%。
所以做了**交替 A/B**（同一个探针、同机、picks 与"当前线上旧二进制"轮流启动，各 2 次 12K + 1 次 100K）：

| 臂 | 12K #1 | 12K #2 | 100K | 解码（12K） |
|---|---|---|---|---|
| picks（含补丁） | 802.8 t/s | 804.8 t/s | 602.6 t/s | 40.3 / 40.0 t/s |
| prod（未含补丁） | 805.3 t/s | 801.8 t/s | 602.9 t/s | 41.7 / 41.2 t/s |

⇒ **两边同时低到 ~803**：那 6% 是环境漂移（本机 WDDM + 桌面负载），不是补丁。
部署新二进制后线上复测 12K = **853.6 t/s**（恢复历史水平），进一步确认。

### 17.2 UB 不是洁癖：`compute-sanitizer --tool synccheck` 的硬证据

先确认实验干净：两棵树的 `ggml/src/ggml-cuda/fattn-mma-f16.cuh` **差异恰好只有这个补丁**
（`git diff --no-index` = 22 增 / 6 删 / 3 hunk），prod 侧二进制是 2026-09-26 11:18 构建的 `build-v2`。

| 二进制 | `-o FLASH_ATTN_EXT` 全量 | hsk=64 子集 |
|---|---|---|
| 未修复（build-v2） | **8576 个 Barrier error**（打印上限内的 40 条全部是 `flash_attn_ext_f16<(int)64,(int)64,(int)16,(int)4,(bool)0,(bool)0>`） | 4000 个 |
| 已修复（build-picks） | **0 个** | **0 个**（1137/1137 tests passed） |

即：那个被删掉的 `__syncthreads()` 是**每次 launch、每个 warp 都在分歧**，只是 V100 上没抛出可见错误；
修复后两次跑 PPL 逐位相同，说明删除的分隔栅栏（leader warp 自己"读 meta → 写回 meta"之间的那道）
在语义上确实可省。两条命令与日志：
`work\tmp\sanitizer-synccheck.ps1`（`--tool synccheck --print-limit 40`）、
`work\tmp\sanitizer-synccheck-{prod,picks}.log`。

### 17.3 提交与部署

* 提交：`0de9b7f53 port(volta-fa): retune hs=256 MMA config + fix np>1 barrier UB`（`local/picks-20260926` 分支）
* 部署：`work\pr210\deploy-picks.ps1` → 旧运行时快照 `work\bin-backup\20260927-135103-full`（可回滚）
* 生效后线上配置未变（ctx 131072 / q8_0 KV / b8192 ub1024 / MTP n_max=2），12K 探针 853.6 t/s、解码 39.6 t/s、显存 14431 MiB
* 剩余未用的一条是 DV=512 配置（我们无此形状）；FA 这条线到此收尾，下一步若要继续压 prefill 只剩 GDN/PTQ 内核侧

---

## 十八、GDN/PTQ 内核侧：先重抓账本，再逐条试（2026-09-27 16:xx）

上一节的结论"剩下的空间在 GDN/PTQ 内核侧"要落到具体的内核上，所以先用 nsys
（`--cuda-graph-trace=node --cuda-event-trace=false`）把**当前生产构建**的账本重新抓了一遍，
并把"预填充 / 解码"两段拆开（脚本：`work\tmp\kern-profile-now.ps1`（bench 形态）、
`work\tmp\kern-profile-mtp.ps1`（128K ctx + MTP 的生产形态）、分析用
`work\tmp\kern-census2.py` / `kern-phase-split.py` / `kern-shapes.py` / `nsys-export.ps1`）。

### 18.1 当前账本（与 9/26 那版差别很大）

**预填充**（pp16384，无 MTP；总 kernel 时间 30.45 s / 83664 launches）

| 内核 | 占比 | 说明 |
|---|---|---|
| `cutlass::Kernel2`（cuBLAS FP16 GEMM） | **58.7%** | 27B 权重走 dequant→FP16 GEMM（by design，见第 13 节） |
| `dequantize_block_ptq1_0` | **9.1%** | 每个 ubatch 都要把整份权重重反量化一次（见 18.3） |
| `gated_delta_net_cuda` | **8.8%** | GDN 顺序扫描 |
| `flash::sm70_d256_splitd_dense_kernel` | 8.3% | 我们自己移植的 D256 FA（第 8~10 节） |
| `convert_unary`（激活 f32→f16） | 3.2% | 每个量化 matmul 一次 |
| `unary_gated_op` / `fwht_cuda_block` / `add_rms_norm_mul` / `concat_conv_state` / `cpy_scalar` / `rms_norm_f32` / `ssm_conv_long` | 2.5/2.4/1.5/1.2/1.1/0.95/0.94% | 长尾 |

**解码**（128K ctx 服务器 + MTP，12K 提示后的 96 token；总 kernel 时间 1.87 s / 165027 launches）

| 内核 | 占比 | 每次/token |
|---|---|---|
| `mul_mat_vec_q`（PTQ1_0/Q8_0 mat-vec） | **58.0%** | 323 次 |
| `flash_attn_ext_vec`（解码注意力，q8_0 KV） | **12.6%** | 15.3 次 |
| `rms_norm_f32` | 5.7% | 201 次 |
| `quantize_q8_1` | 4.6% | 328 次 |
| `fwht_cuda_block` | 4.2% | 250 次 |
| `mul_mat_vec_f`（f16ssm 权重） | 2.5% | 92 次 |
| `gated_delta_net_cuda` | 2.5% | 47 次 |

两条首要事实：mat-vec 仍是解码的绝对大头（58%，有效带宽 ~527 GB/s / 峰值 900）；
解码注意力在 12K 时 12.6%，但它是**唯一随 KV 长度线性增长**的项（见 18.4）。

### 18.2 试过并否决：PTQ1_0 单列改走 PT（planar）布局

上游注释写明这个取舍只在 Ampere/Ada 上量过（"Ampere：PT 单列 +5.9% vs SoA（3060）；
Ada 及以后保持 SOA_ISUM"），**Volta 从来没测过**。加了个诊断开关
（`GGML_PTQ1_1COL=pt|soa`，默认不生效，改的是 `common.cuh: ggml_cuda_q8_1_layout_host`），
同一份二进制交替 A/B（12K 深度解码，`llama-bench -p 12288 -n 128 -r 2`）：

| 臂 | tg128 #1 | tg128 #2 | 均值 |
|---|---|---|---|
| SoA_ISUM（现状） | 49.74 ± 0.38 | 49.10 ± 1.13 | **49.42** |
| PT（planar） | 48.10 ± 0.36 | 48.06 ± 0.68 | 48.08 |

⇒ **PT 在 Volta 上慢 2.7%**，Volta 与 Ada 同侧，保持 SoA。日志 `work\tmp\ptq1-1col-ab.log`。

### 18.3 试过并否决（仅限 128K 档）：ubatch 2048 省 dequant

账本解释了 `dequantize_block_ptq1_0` 为什么能占 9%：它在**每个 ubatch** 都把权重重反量化一次
（pp16384 里 12800 次 launch ≈ 16 个 ubatch × 800 个量化 matmul）。于是 ubatch 越大，
这份开销越少：

| 配置 | pp16384（bench，ctx≈16K） | 128K ctx + MTP 的 12K 探针 | 显存 |
|---|---|---|---|
| `-ub 1024`（生产） | 1028.15 / 1027.82 | **854.1 t/s** | 14395 MiB |
| `-ub 2048` | **1075.56 / 1076.07（+4.65%）** | **118.9 t/s（崩）** | 15746 MiB |
| `-ub 4096` | — | 151.8 t/s（崩） | 15765 MiB |

短上下文时 ub 2048 是白拿的 +4.65%；但 128K ctx 下显存冲到 15746 MiB 触发 WDDM 回退，
prefill 直接掉到 1/7。**结论：128K 生产档保持 ub 1024；面板将来可以给"短会话档"单独带 ub 2048。**
（脚本 `work\tmp\ub-prod-sweep.ps1`，日志 `ub-prod-sweep.log`。）

### 18.4 下一个主战场：解码注意力（长上下文解码的 53%）

按形状拆开 `flash_attn_ext_vec` 的 launch（`kern-shapes.py`）：

| 场景 | grid | 每次 launch | 每 token |
|---|---|---|---|
| 纯解码（q_len=1） | (1,1,24) | 39.5 µs | **0.64 ms** |
| MTP 解码（q_len=3） | (1,13,24) | 159.6 µs | **2.45 ms（3.8×）** |

12K ctx 时它只占解码 12.6%，但 KV 流量随上下文线性增长：100K ctx 时按同一系数外推 ≈ **20 ms/token
≈ 解码总量的 53%**（与实测 100K 解码 25.8 t/s = 38.8 ms/token 吻合）。而它只跑出
~170 GB/s（峰值 900）——比 mat-vec 更远离带宽墙。

原因是形状：`fattn-sm70-d256.cu` 的准入条件里写死了 `!mask || mask->ne[0] < 256 || Q->ne[1] < 256`
（"prefill only; decode/MTP/small batches -> stock"），所以 1~3 行的解码只能走 stock 的
`flash_attn_ext_vec`（q_len==1 用 cols_per_block=1，q_len≥2 用 2，且带 stream-K 切分）。

⇒ **下一步（待做）**：给 D=256 / q8_0 KV 写一个解码形态的 sm70 内核（q_len 1~8），或者先把
vec 内核在 q_len=3 下的 cols_per_block / stream-K 切分做个 A/B（便宜但天花板低）。
预期：100K 解码 26 → 33~40 t/s，12K 解码 49 → 53 t/s 量级。

### 18.5 100K 生产形态的账本：解码注意力确实是一半（决定性证据）

按 18.4 的怀疑直接抓了 100K 上下文的生产形态（99821 token 提示 + 48 token 输出，MTP 开，
`work\tmp\kern-profile-mtp.ps1` 改成 1720 段/48 token 后重跑）：

**解码**（总 kernel 1.69 s / 78711 launches；探针报 21.6 t/s）

| 内核 | 占比 | 次数 | 每次 |
|---|---|---|---|
| **`flash_attn_ext_vec`** | **53.0%** | 693 | **1.29 ms** |
| `mul_mat_vec_q` | 29.8% | 14598 | 34 µs |
| rms_norm + quantize + fwht | 7.4% | 35569 | — |
| 其余（mmvf/GDN/combine/…） | 9.8% | — | — |

**预填充**（100K 提示，总 kernel 131.2 s）

| 内核 | 占比 | 备注 |
|---|---|---|
| `cutlass::Kernel2` | 41.6% | |
| **`flash::sm70_d256_splitd_dense_kernel`** | **34.4%** | 45.1 s —— 长提示下我们自己那个 FA 成了第二大项（0.33 s/层·ubatch 级） |
| `dequantize_block_ptq1_0` | 6.5% | |
| `gated_delta_net_cuda` | 6.3% | |

把解码那 1.29 ms/次摊成带宽：100K 上下文时每层每 token 的 K+V = 2176 B（4 kv head × 256 ×
q8_0 的 34/32 开销 ×2），一层一次 launch 要读 217.6 MB → **169 GB/s，只有峰值的 19%**。
12K 时同一内核是 660 GB/s（24 个 block、96 warps）——所以不是"天生慢"，是**并行度和访存结构**问题：
100K 时 grid=(1,13,24)，z=24 = 6 个 q-head 组 × 4 个 kv head，**同一份 K/V 被 6 个 q-head 组
各读一遍**（GQA 打包缺失）；而且 255 寄存器 → 2 block/SM。

对照：上游那套 ncols2 GQA 打包（T2-001）**只支持 F16/BF16 KV**，我们 q8_0 吃不到；
我们自己的 `fattn-sm70-d256.cu` 写死了 `Q->ne[1] < 256 → stock`（prefill only）。

> 附：想用 ncu 核实 "L2 打爆 vs DRAM 打爆" 时被 `ERR_NVGPUCTRPERM` 挡住——注册表里
> `RmProfilingAdminOnly=0` 需要**重启一次**才生效（或用 UAC 提权跑一次 ncu）。

### 18.6 结论与下一步

* 解码侧：**给 D=256 / q8_0 KV 写一个 sm70 解码内核**，要点是 (1) GQA 打包（一个 block 处理同一个
  kv head 的全部 6 个 q head，K/V 只读一遍）、(2) q8_0 走 16 字节向量化读 + dp4a（K·Q 用 q8 量化
  Q，和 mat-vec 同一套），(3) 列数 1~8（覆盖 MTP 的 1+2 行）、(4) 复用现成的
  `dst_tmp/dst_tmp_meta` + `flash_attn_combine_results` 协议（这样 KV 切分/fixup 不用重写）。
  预期：100K 解码 21.6 → 30~40 t/s，12K 解码 49 → 55 t/s 量级；预填充不受影响。
* 预填充侧：100K 下 sm70 D256 FA 占 34.4%，是仅次于 cuBLAS 的第二项；等解码版落地后，
  同一套 GQA 打包/向量化思路可以回头喂给 prefill 版（它现在也是按 kv head 分组、q 头不打包）。
* GDN 仍占预填充 6.3%、解码 1.2%，暂不做（第 16、17 节已把便宜的做完）。

### 18.7 Volta 的 FA 派发规则 + 硬件计数器（提权 ncu，2026-09-27 17:xx）

读 `fattn.cu: ggml_cuda_get_best_fattn_kernel()` 把 Volta 的派发规则确定了：

```cpp
if (volta_mma_available(cc) && ...) {
    if (sm70_d256_supported(cc, dst))                    return SM70_D256;   // 我们的 prefill 内核
    if (can_use_vector_kernel && Q->ne[1]*gqa_ratio_eff <= 2) return VEC;
    if (Q->ne[1]*gqa_ratio_eff <= 16)                    return TILE;
    return MMA_F16;
}
```

`gqa_ratio_eff` = gqa_ratio 里最大的 2 的幂（D=256 时上限 8）。我们模型 gqa_ratio=6 → eff=2，于是：

| 场景 | q_len × eff | 内核 |
|---|---|---|
| 纯解码（q_len=1） | 2 | **VEC** |
| MTP 验证（q_len=3） | 6 | **TILE** |

即 **开 MTP 后，解码注意力从 vec 换成 tile**——这解释了两个抓图里 kernel 名的差异（12K 那份探针接受率近 0，
验证提前退出成 q_len=1，所以看到 693 次 vec；100K 那份则 vec/tile 混合）。

提权跑 `ncu`（用户配合 UAC）拿到 tile 内核在 **kv=20000 / nb=1 / q8_0 / D=256** 上的计数器：

| 指标 | 值 | 含义 |
|---|---|---|
| `gpu__time_duration` | 846 / 816 µs | |
| `dram__bytes` | 31.9 / 29.0 MB | **37.7 GB/s = 峰值 4%** |
| `lts__t_bytes` | 258 / 264 MB | **L2 流量是 DRAM 的 8 倍**（命中率 93~95%） |
| `l1tex__data_pipe_lsu_wavefronts` | **38.9 M** | 21.8 MB 有用数据 → L1 wavefront 放大 ~230× |
| `sm__warps_active` | **12.4%** | 占用率极低 |
| `launch__registers_per_thread` | 255（上限）→ 4 block/SM | 寄存器封顶 |

⇒ 结论：**整个 decode-attention 家族（vec 与 tile）在 Volta 上都是"低占用 + 巨大访存放大"的延迟受限内核**，
离带宽墙有 20 倍以上的差距。这跟我用 nsys 算出的生产 169 GB/s 完全一致。

**设计目标（据此定）**：新内核要走"每次访存都用满"的路子——
1. GQA 打包（一个 block 处理同一 kv head 的全部 6 个 q head，K/V 只读一遍，消灭 8× L2 放大）；
2. q8_0 的 16 字节向量化读 + `dp4a`（K·Q 用 q8 量化 Q，与 mat-vec 同一套点积）；
3. 在线 softmax 用 lane 私有累加 + 少量共享内存归约（不要 255 寄存器）；
4. 复用 `dst_tmp/dst_tmp_meta` + `flash_attn_combine_results` 协议，KV 切分/fixup 不重写；
5. 列数 1~8（覆盖 MTP 的 1+2 行），只在 Volta + D=256 + q8_0 KV 时接管 VEC/TILE 两条分支。

* 诊断开关：`GGML_FATTN_FORCE_VEC=1`（`fattn.cu`，默认不生效）——用来在同一个测试形状上强制走 vec，
  方便 tile/vec 同形状 A/B。已提交（`work\tmp\ncu-fa-ab.ps1` 是配套的提权计数器脚本）。
* 待补：vec 内核在同形状下的计数器（kv 需 %256==0，用 kv=4096 的用例；这次 UAC 被取消，下次再跑）。

### 18.8 把两个现成内核都试完了：tile 是 q_len>1 的最优，vec 更慢（2026-09-27 18:xx）

在写新内核之前，先用 `GGML_FATTN_FORCE_VEC` 把"能不能不写新内核"这条路走到头。三组实测，
都用生产配置（128K ctx / q8_0 KV / MTP n_max=2 / ub 1024）：

**① spec-bench（短上下文、高接受率）**：四个任务、300 token、temperature 0

| 臂 | math | code-csv | code-bs | chat | 接受率 |
|---|---|---|---|---|---|
| tile（现状）×2 | 69.4 / **69.5** | 55.9 / 55.4 | 59.2 / 59.4 | 60.4 / 60.8 | 91/60/69/71% |
| vec（强制） | 68.1 | 崩过一次 | — | — | 91% |

短上下文时两者基本同速（注意力只占一小部分），vec 还在 spec-bench 的某个请求上**偶发崩溃**
（`vec-crash-diag.ps1` 单发请求却正常，说明是特定形状/竞态），所以先不指望它。

**② 100K 上下文 + 低接受率尾巴**（`prefill-probe.py 1720 32`，尾句"总结"）：tile 臂 **28.8 t/s**。

**③ 100K 上下文 + 高接受率尾巴**（尾句 `count`，让 MTP 真的跑 q_len=3 的验证）——**决定性**：

| 臂 | prefill | **解码** |
|---|---|---|
| **tile（生产派发）** | 646.6 t/s | **35.4 t/s** |
| vec（强制） | 643.7 t/s | **24.2 t/s（−32%）** |

⇒ 结论：
1. **vec 不能用来跑 MTP 验证**（长上下文下比 tile 慢 32%，短上下文偶发崩溃）；
2. **tile 就是现成内核里 q_len>1 的最优**（虽然它 DRAM 利用率只有 4%、warp 占用 12%）。
   所以"改派发"这条捷径不存在，**必须写新内核**；
3. 新内核的验收基线定下来了：
   * 短上下文 spec-bench math ≥ 70 t/s（tile 69.5）；
   * **100K + count 尾巴 ≥ 35.4 t/s**（tile）——这是要越过的线；
   * 顺带记住：同一条 100K 提示，尾巴可预测时 MTP 才有意义（28.8 → 35.4 t/s）。

脚本：`work\tmp\fa-vec-force-ab.ps1`（spec-bench 双臂 + 100K 探针）、`fa-vec-longcount-ab.ps1`（100K+count）、
`vec-crash-diag.ps1`、`vec-long-test.ps1`、`one-ask.py`；日志 `fa-vec-force-ab.log`、`fa-vec-longcount-ab.log`。

---

## 十九、D256 解码内核：写出来了、数值过了、但性能不达标（2026-09-27 21:xx）

### 19.1 成品

新文件 `ggml/src/ggml-cuda/fattn-sm70-d256-decode.cu`（提交 `7d1574344` + 修复 `ec1e1d155`）：

* 一个 block 吃掉**同一个 kv head 的全部 `ncols2 = 6` 个 q head + `ncols1 ≤ 4` 行 query**，
  这样一份 K/V 只读一遍（tile 路径是 6 个 q head 各读一遍、vec 路径再加每 q-tile 重复）；
* K/V 以「qs 平面 + scale 平面」暂存 shared（**不能**直接存 34 字节的 `block_q8_0`：qs 在块内偏移 +2，
  用 int 读它就是 `misaligned address`，这是踩过的坑之一）；
* Q 量化成 q8（`sQ` + `sQd`），Q·K 用 `dp4a`，逐 (行,头) 在线 softmax，V 累加 8 维/lane；
* KV 切分复用现成的 `dst_tmp / dst_tmp_meta + flash_attn_combine_results` 协议；
* 派发挂在 Volta 分支的 `BEST_FATTN_KERNEL_SM70_D256_DEC`，**默认关闭**，`GGML_SM70_D256_DECODE=1` 才启用。

### 19.2 两个真 bug（都已修）

1. **`misaligned address`（整模型图预热即崩）**：KV 视图基址只保证 4 字节对齐，我却用 16 字节拷贝搬行 →
   改成 2 字节粒度；顺手发现 `block_q8_0.qs` 的 +2 偏移让任何 int 读都非法 → 改成 qs/scale 分平面。
2. **q8 打包符号扩展（NMSE 0.06）**：`v |= (int)roundf(x) << 8;` 在 x 为负时把符号位铺进相邻字节
   （`-5 << 8 == 0xFFFFFB00`）。定位方式很值：加了个 `SM70_D256_DEC_F32Q=1` 调试开关把 Q·K 换成
   fp32 全精度——立刻 3/3 通过，直接锁定问题在打包那 4 行。修法是每个字节 `& 0xFF`。

### 19.3 验证结果

| 闸门 | 结果 |
|---|---|
| `test-backend-ops`（自加 15 例：hsk=hsv=256/q8_0/nh=4/nr23=[6,1]/kv 32..1024/nb 1..3/有→无 mask） | **15/15 通过** |
| 模型级贪心（300 token，MTP 开，内核 on vs off） | **逐字节一致**，sha256 `d697998df4e83e43`，接受率同为 91% |
| 100K + count 尾巴（决定性性能） | 内核 on **17.2 t/s** vs tile 基线 **35.4 t/s** → **慢一倍** |

### 19.4 性能为什么输：内层循环被 5 次 shuffle 的归约串死了

现在的内层是「**warp 一起算一个 token 的点积**」：每个 (token, pair) 都要 2 个 dp4a + **5 次 `__shfl_xor` 归约**
+ 2 个 `expf` + 8 个 FMA，而且 token 是**串行**循环的。5 次 shuffle 的依赖链（~50 周期）无法被掩盖，
每 warp 只有 5 个 pair 在飞 → 实测就是 17 t/s。

tile 内核恰好相反：**每个线程管一个 token**，分数算完写 shared，softmax 归约和 V 累加都放在别处做，
所以它没有 per-token 的 warp 归约。

**重写方案（下一轮）**：tile 式的两段结构（FA2 形状）——
1. **分数段**：lane ℓ 独自算 tile 内第 ℓ 个 token 的完整 256 维点积（64 个 dp4a，K 从 shared 读 16 个 int4、
   Q 从 shared 广播读 16 个 int4，无 shuffle）→ 每 (token,pair) 摊到 ~3.3 条 warp 指令；
2. tile 内 max / sum 各一次 5-shuffle 归约（每 32 个 token 才一次 → 0.16 次/token）；
3. **V 累加段**：lane 仍管 8 个维度，对 tile 内 32 个 token 的 p_t 做累加（8 FMA/token，独立无依赖）。

粗算：每 (token,pair) 从 ~30 条 warp 指令降到 ~12 条且没有串行链 → 目标 ≥ 35.4 t/s（希望能到 45-60）。
现有结构的正确性已验证，重写只动内层，外层的 mapping/epilogue 全部保留，回归闸门就是上面那三条。

### 19.5 重写完成（两段结构）：生产 17.2 → 26.1 t/s，但仍不如 tile

按 19.4 的方案把内层改成「pass A：lane ℓ 独自算 tile 内第 ℓ 个 token 的完整 256 维点积（64 dp4a，
K/Q 全从 shared 读、无 shuffle）；tile 内 max/sum 各一次归约；pass B：lane 管 8 个维度对 tile 内
所有 token 累加」，并把 `__launch_bounds__` 的第二参数从 1 提到 3（争取 3 block/SM 的占用率）。

* 正确性：`test-backend-ops` **15/15 通过**（重写没破坏数值）。
* 100K + count：**26.1 t/s**（上一版 17.2，tile 基线 33.3~35.4）——仍慢 ~22%。

为了能 30 秒一轮调优，把这两个形状也加进了 `test-backend-ops perf` 列表
（`hsk=hsv=256, nh=4, nr23=[6,1], kv 512..16384, nb 1/3`），同形状直接对比（µs/run）：

| kv | nb | 本内核 | tile | 比值 |
|---|---|---|---|---|
| 512 | 1 | 42.6 | 36.7 | 1.16× |
| 512 | 3 | 51.0 | **24.2** | 2.10× |
| 1024 | 1 | 36.5 | 28.9 | 1.26× |
| 1024 | 3 | 75.5 | **35.9** | 2.10× |
| 4096 | 1 | 89.8 | 62.5 | 1.44× |
| 4096 | 3 | 181.6 | **73.5** | 2.47× |
| 16384 | 3 | — | 275.9 | （tile 相当于 517 GB/s） |

两个关键认识：
1. **单行（nb=1）时本内核只慢 16~44%**，说明 GQA 打包 + 一次读 K/V 的结构是对的；
   但**多行（nb=3，也就是 MTP 验证）时 tile 反而快 2.1~2.5 倍**——tile 在这个形状上会开
   `parallel_blocks × (2 行/block)` 的大量 block，SM 填得满；本内核每个 (kv head, 分裂) 才一个 block，
   并行度不够，而且每个 tile 的固定开销（2 字节粒度的 staging + 2 次 `__syncthreads` + pair 循环）
   在 kv 较小时占比很大。
2. **tile 在我们的生产形状上并没有 19.7 节那份 ncu 数据看起来那么差**：那份 4% DRAM 是
   `nh=2/gqa=16` 的测试台形状；按本表 kv=16384/nb=3 反推，tile 实际跑到 ~517 GB/s，已接近可用带宽的一半。

**下一步（若继续）**：① staging 改回 16 字节（运行时判断 4 字节对齐）；② TILE 与 parallel_blocks
联合调（现在 parallel_blocks 由 launch_fattn 的「波效率」启发式决定，对小 kv 会选偏小）；
③ pass B 一次处理 2 个 token 提升 ILP；④ 若仍追不上，就接受「tile 在 Volta 上已够好」这个结论，
把这条线的投入收掉——目前它离收益线还差 ~25%。

---

## 二十、转高 ROI 项：sm70 prefill 内核的「隐式约定」与真实性价比（2026-09-27 23:xx）

### 20.1 它到底有多快：1.85× stock（预填充形状的硬数字）

把预填充形状加进 `test-backend-ops perf`（hsk=hsv=256、nh=2/nr23=[16,1]、kv 10000/20000、**nb=512**，
与生产 prefill 同形），同形状对比：

| 形状 | sm70 D256（默认） | stock（`LLAMA_SM70_D256=0`） | 比值 |
|---|---|---|---|
| kv=10000 f16 KV | 3965 µs / 42.3 TFLOPS | 6910 µs / 24.3 TFLOPS | **1.74×** |
| kv=20000 f16 KV | 7395 µs / 45.4 TFLOPS | 14254 µs / 23.5 TFLOPS | **1.93×** |
| kv=10000 q8_0 KV | 3907 µs / 43.0 TFLOPS | 7208 µs / 23.3 TFLOPS | **1.84×** |
| kv=20000 q8_0 KV | 7521 µs / 44.6 TFLOPS | 13812 µs / 24.3 TFLOPS | **1.84×** |

⇒ 这个内核已经把预填充注意力做到 stock 的 1.8~1.9 倍，"34.4% 的预填充占比"里它是**已经优化过**的部分；
再往上的空间是 MMA 调参级别（42→50+ TFLOPS），不是结构性的。

### 20.2 测试台里那 2 个"失败"用例：是**约定的差异**，不是生产 bug

用测试台的 f16/q8_0 + nb=512/1024 形状去验，ERR≈1.9（关掉内核 2/2 通过），但查了测试台怎么造 mask：
`init_tensor_kq_mask()` 生成的是**随机 mask**（[-1,1] 均匀 + 20% 块被设成 -INF 或 0），
**不是因果 mask**。而 llama.cpp 生产里的 FA mask 一定是「因果窗口」约定（并且内核用 KV_max 跳过整块被遮的 KV）。
我们的内核正是按这个约定特化的（这也是它比 stock 快 1.8× 的原因之一：它只处理真正可见的部分）。
⇒ 测试台那类用例测的是内核**不支持的输入约定**，不能当生产 bug；但也说明内核的准入条件里
**应该把「mask 必须是因果/窗口约定」写成注释或检查**，否则以后换模型（SWA、非因果注意力）会踩。

### 20.3 一个需要留意的事实：贪心输出会与 stock 路径分叉

12K 提示、`count` 尾巴、temperature=0，内核 on vs off：

| 臂 | prefill | decode | 40 token 贪心 sha256 |
|---|---|---|---|
| sm70 on | 878.4 t/s | 39.9 t/s | `b2a8ca3c22b4bd33` |
| sm70 off | 887.4 t/s | 44.4 t/s | `dea4c1961f81fda3` |

两条路径**数值上不同**（不同的融合顺序/精度），40 token 的贪心输出分叉是正常的；这条路径的验收口径
应该是 **PPL/KL 之类的质量指标**（第 8 节：q8_0 下 sm70 12.2796 vs stock 12.2854，前者还略好），
**不是贪心哈希**（这一点和 MTP 的"开/关逐字节一致"不同——那是同一套内核开关草稿，属于同一算术）。

### 20.4 这条线的下一步（按 ROI 排序）

1. 给内核头部补上**mask 约定 + KV_max 依赖**的说明，并在准入条件里加一条「仅接受因果/窗口 mask」的
   显式检查（能检查的部分：`mask->ne[0] == K->ne[1]`、`q_len >= 256`、`kv_len > q_len` 等已经在做）；
2. 若要继续压预填充：目标是把 sm70 D256 从 42~45 TFLOPS 推到 50+（ncu 提权抓一次 stall 分布 → 调 tile/K 循环），
   预期收益 ≈ 预填充的 3~5%（因为 FA 占 34.4%，提 20% 也只有整体 7%，且它是 1.85× 已优化的部分）；
3. **更高 ROI 的方向**（如果还想在 100K 预填充上再挖）其实是那 41.6% 的 cuBLAS GEMM 与 6.5% 的反量化——
   但这两块在 9/26 已论证接近 V100 实用上限；剩下真正能动的是 **ubatch 与显存的联合调参**（第 18.3 节：
   短上下文 ub 2048 白拿 +4.65%，128K 档被 WDDM 回退卡住——也许可以只在"≤32K 上下文"的档位里启用 ub 2048）。

### 20.5 已落地：ubatch 2048 按上下文长度分档（面板预设）

把"ub 2048 在什么上下文还安全"测清楚了（生产形态服务器 + 12K 探针，`work\tmp\ub-ctx-sweep.ps1`）：

| ctx | ub | 显存 | 12K prefill | 12K decode |
|---|---|---|---|---|
| 32768 | 1024 | 9445 MiB | 849.9 t/s | 39.1 t/s |
| 32768 | **2048** | 9687 MiB | **912.6 t/s（+7.4%）** | 43.7 t/s |
| 65536 | **2048** | 11243 MiB | 894.5 t/s | 44.3 t/s |
| 98304 | **2048** | 12799 MiB | 913.0 t/s | 44.8 t/s |

⇒ 2048 在 ub 档一直到 ~96K 上下文都很安全（显存 12.8 GB），并且**预填充 +7.4%、解码 +12%**；
131072 档之前实测出现过 WDDM 回退（第 18.3 节），保持 1024。

**已改**：面板（`outputs/bonsai-dashboard.py`）的预设现在带 `ub`：
`32K → ub 2048`、`64K → ub 2048`、`128K → ub 1024`、`256K → ub 1024`
（点预设会一起把 ubatch 填进去），面板已重启生效（`/` 里能看到这四行）。

### 20.6 端到端复核：短会话档（ctx 64K + ub 2048）

「⚡ 短会话 · MTP 加速」那个按钮本来就是 ctx 65536 + ubatch 2048 + MTP n_max=2，所以直接用
spec-bench（四任务 ×300 token）+ 12K 探针复核它在同 ctx 下相对 ub 1024 的收益：

| arm | VRAM | spec-bench math/csv/bs/chat | 12K prefill | 12K decode |
|---|---|---|---|---|
| ctx 65536 **ub 2048** | 11519 MiB | 69.4 / 56.1 / 60.0 / 61.2 | **903.1 t/s** | **52.4 t/s** |
| ctx 65536 ub 1024 | 11085 MiB | （未跑，短提示下与 128K 档同值 69.4/55.9/59.2/60.4） | 868.6 t/s | 41.9 t/s |

⇒ 同 ctx 下 ub 2048：预填充 **+4%**、解码 **+25%**（52.4 vs 41.9，解码数值抖动较大，但两个独立测量
都指向 2048 更好）；spec-bench 因为提示很短，四任务数字与 128K 档完全一致（说明这个档只是"更快"，
不改变输出质量）。显存只多 434 MiB。

已把这条理由写进面板按钮的 tooltip（含"96K 时 12.8 GB、131072 档贴边所以保持 1024"）。

---

## 二十一、解码内核调参收尾：单行赢了 tile，但生产上零收益 → 收线（2026-09-27 18:xx）

按 19.5 的清单把调参做完，一共动了三处，每处都用 `test-backend-ops perf` 的 30 秒闭环验：

1. **线程数 128 → 256（8 warps）**，`__launch_bounds__(256, 2)`：kv=512/nb=1 从 42.6 → **34.2 µs**，
   nb=3 从 51.0 → 46.2；
2. **V 在 staging 阶段一次性反量化成 half**（`sVh`，fp32 累加不变）：pass B 从"每 lane 每 token
   8 次 int8→float + 8 FMA"降到"4 次 half2 读 + 4 次 half2→float2 + 8 FMA"，nb=1 再降到 33.6、
   nb=3 降到 43.2（这一项省掉了 pair 之间的重复反量化）；
3. **只接管 `Q->ne[1] == 1`**（nb≥2 交回 tile）：`GGML_SM70_D256_DEC_ROWS=1` 的 A/B 显示把行拆到
   不同 block 在 nb=3 上能再快 8%（43.2 → 39.7），但仍输 tile，所以干脆不接。

最终的逐形状对比（µs/run，同形状 tile = 默认派发；本内核默认关闭）：

| kv | nb | 本内核 | tile | |
|---|---|---|---|---|
| 512 | 1 | **33.6** | 36.7 | ✅ −8% |
| 1024 | 1 | 29.4 | 28.9 | ≈ |
| 4096 | 1 | **57.7** | 62.5 | ✅ −8% |
| 16384 | 1 | **151.1** | 190.2 | ✅ **−21%** |
| 512 | 3 | 39.7~43.2 | **24.2** | ❌ +64% |
| 1024 | 3 | 71.2 | **35.9** | ❌ +98% |
| 4096 | 3 | 163.4 | **73.5** | ❌ +122% |
| 16384 | 3 | 443.2 | **275.9** | ❌ +61% |

**生产形态验证（100K ctx，MTP 开）**：

| arm | 尾巴 | prefill | decode |
|---|---|---|---|
| 默认（内核关） | 总结（低接受率） | 645.2 | **29.3** |
| 内核开 | 总结 | 641.3 | **29.3** |
| 内核开 | count（高接受率） | 642.1 | 35.3（基线 35.4） |

调试打印确认内核**确实被选中了**（`[sm70-dec] ACCEPT Q=(256,1,24,1) K=(256,131072,4,1)` 出现 388 次，
q_len≥2 全部 reject 交给 tile）——所以"零收益"不是没接上，而是 **q_len=1 的那些调用在 100K 解码里
占比太小**（MTP 的每步验证是 q_len=3 的 tile 调用在吃时间），单行提速 21% 传导不到端到端。

⇒ **决定收线**：本内核三条闸门全过（15/15 测试台、模型级贪心逐字节一致、100K 无回归），
在 nb=1 上确实比 tile 快 8~21%，但生产端到端 **零收益**（也不会变差）。保持**默认关闭**
（`GGML_SM70_D256_DECODE=1` 才启用），代码与调参记录留档，不再投入。
这与 9/26 那批"试过并否证"的结论放在一起：V100 上这块的最后 20% 需要的是"重写内核 + 大量调参"
且收益只在单行场景，性价比低于已经拿到的那些（MTP +54% prefill、sm70 prefill 1.85×、ub2048 +7%）。

---

## 二十二、按「多轮 agent」负载定「Agent 特调档」（2026-09-27 19:xx）

数据源：作者本机 agent 客户端的会话统计（**只统计 token 计数，不落内容**），**仅作负载形态参考**。
下面是"多轮 agent"这个负载的形态——它才是档位要服务的对象：

| 指标 | 典型范围 | 说明 |
|---|---|---|
| 每轮上下文 | 约 30K~110K（中位在 60K 上下） | 尾部略超 110K |
| **每轮新增 token**（复用后真正要预填充的量） | 中位几百 token，尾部 10K~30K | cache_prompt 命中后只剩增量 |
| 每轮输出 | 几百 token | 长回答可到数千 |
| 每轮工具调用 | 1 次左右 | — |

上下文分档集中在 32K~110K ⇒ 128K 窗口 + 100K 压缩阈值正好卡在分布上。

### 22.2 按这个形态重测 ubatch：**无差别**（旧结论"512 更快"不复现）

新增 `work\tmp\agent-turn-bench.py`：8K 前言 + 10 轮，每轮追加 ~117 token 工具输出、输出 300 token、
`cache_prompt=true`（完全复刻 Codex 的形态）。ctx 131072、MTP 开：

| ub | 10 轮总耗时 | 每轮 | prefill 均值 | 解码均值 |
|---|---|---|---|---|
| 512 | 47.3 s | 4.73 s | 977 ms | 57.2 t/s |
| **1024** | **44.3 s** | 4.43 s | 996 ms | 57.5 t/s |
| 2048 | 45.1 s | 4.51 s | 1,024 ms | 58.0 t/s |

⇒ 对这种"多轮小增量"，ub 512/1024/2048 差异在噪声内（面板注释里 9/26 那次"512 更快"是当时构建/环境的结果，
不复现）。**Agent 档保持 ub 1024**（128K 下 ub 2048 会贴显存，第 18.3 节）。

### 22.3 新线索：每轮有 ~1 秒的固定预填充开销（占每轮 23%）

每轮只新增 117 token，但 `prompt_ms` 稳定 ~1.0 s（8.4 ms/token，比预填充实测的 1.2 ms/token 慢 7 倍）。
逐个排除：

| 假设 | 证据 | 结论 |
|---|---|---|
| MTP draft catch-up | `SPC_MTP_PROF=1`：每轮 14.5 ms，首轮一次性 397 ms（0.0555 ms/row） | ❌ 不是 |
| CUDA 图重捕获 | `GGML_CUDA_DISABLE_GRAPHS=1`：prefill 仍 983 ms，而解码崩到 29.8 t/s | ❌ 不是（且图必须开） |
| ubatch 太小 | ub 512/1024/2048 的 prefill 均值 977/996/1,024 ms | ❌ 不是 |

剩下的可能是**小 batch（117 行）前向本身的代价**：走 dequant→cuBLAS 时每个 ubatch 都要把 5.5 GiB 权重
重反量化一遍（约 110 ms）+ 117 行的 GEMM 效率低，加上 FA/GDN/图构建，凑出 ~1 s。这一块 **1 s/轮 × 你的
每轮 4.3 s = 23%**，是目前单点最大的可优化项；下一步用 `LLAMA_TRACE`/`SPC_MTP_PROF` 同类埋点把
117 行前向的各阶段拆开（dequant / GEMM / FA / GDN）。

### 22.4 落地

面板「🤖 Agent 最优（实测调优）」按钮的 tooltip 用上面的负载形态解释每个参数为什么这么配
（128K/110K、ub 1024、np 1、MTP+图形状缓存、关思考 256 预算），已重启面板并核对页面内容。参数本身
**不用改**——这轮的价值是"用数据证明它已经配对了"，外加把上面那条 1 s 的线索钉住。

---

## 二十三、那 1 秒拆开了：**小 batch 的权重搬运**（2026-09-27 20:xx）

方法：`work\tmp\prefill-turn-profile.ps1`（nsys 抓两次请求）+ `prefill-turn-client.py`（预热 8K 前缀 →
再发同一前缀 +117 新 token、max_tokens=1）+ `kern-gap-groups.py`（按内核时间轴空隙把两次请求切开）。

实测：预热 = 8,183 token 冷预填充 8,532 ms（960 t/s）；**目标轮 = 117 新 token / 8,179 命中 → prompt_ms 841 ms**。

### 23.1 目标轮的内核账本（busy 0.50 s / 7,112 kernels）

| 内核 | 占比 | 时间 | 说明 |
|---|---|---|---|
| `dequantize_block_ptq1_0` | **35.1%** | **174 ms** | 每 ubatch 把 5.9 GB 打包权重量化成 fp16 |
| `cutlass::Kernel2`（cuBLAS fp16 GEMM） | 26.9% | 134 ms | 大 m 时用的那个 |
| **`volta_s884gemm_fp16_128x64_ldg8_tn`** | **24.2%** | **120 ms** | **cuBLAS 自带的 Volta 老内核**，小 m 时它会从 cutlass 切过来（代码里搜不到，不是我们的） |
| `mul_mat_vec_ptq1_0_pt` / `flash_attn_ext_f16` / GDN | 各 ~2.5% | ~12 ms | 1 token 解码 + FA + GDN |
| convert_unary / fwht / rms_norm / … | 合计 ~3% | | |

对照：8.5 s 的冷预填充里 dequant 只占 10.1%、s884gemm 完全没出现（走 cutlass）。

### 23.2 结论：小 batch 时瓶颈是**固定 102 GB 的权重搬运**，不是算力

每轮（117 行）的权重相关流量：

* 反量化：读打包 5.9 GB + **写 fp16 48 GiB**
* GEMM：**再读 fp16 48 GiB**
* 合计 ≈ **102 GB**，按 900 GB/s 的 113 ms 是理论下限；实测 428 ms（174 + 254）→ 只有 ~240 GB/s，
  因为小 m 的 GEMM/反量化内核都是延迟受限的。

而这一轮真正的数学量（117 行）只需要 ~1/1000 的 FLOP。

**⇒ 优化方向：把反量化融进 GEMM，彻底不落 fp16。** 打包权重只读一遍（5.9 GB ≈ 12 ms@500GB/s），
117 行的点积用 `dp4a` 就够（我们已经有 PTQ1_0 的 dp4a vec-dot 与 PT 内核，只是它只支持 ≤8 列）。
预期：每轮 prefill 从 841 ms → 300~400 ms，**每轮总耗时 4.3 s → ~3.9 s（约 −10%）**。

⚠️ 注意这**修正**了 9/26 的旧结论（第 13 节"给 Volta 写 PTQ1_0 MMQ 没有收益"）：那条针对的是**大 batch**
（当时 m=1024/8192，走 fp16 张量核心接近实用上限，dp4a 反而慢）。**小 batch（m ≲ 256，也就是多轮
Agent 的每一轮）完全相反**：算力无所谓，fp16 物化才是纯开销。两条不矛盾，适用区间不同。

### 23.3 下一步

写一个 PTQ1_0 的「小 m 融合 GEMM」（dp4a，物化 fp16 去掉，读打包权重一次，支持 m ≤ 256），
只在这个区间替换 dequant+cuBLAS 路径；验收：`test-backend-ops` 对齐 → 模型级 PPL/贪心 →
`agent-turn-bench.py` 每轮耗时（当前基线：10 轮 44.3 s / 每轮 prefill ~1.0 s）。

---

## 二十四、小 m 融合路径：做好了、每矩阵快 2.3~13.7×、但**端到端打平**（2026-09-27 21:xx）

### 24.1 实现（提交见下，默认关闭）

`ggml_cuda_mul_mat_ptq1_0_multi_chunk()`（`mmvq.cu`，声明在 `mmvq.cuh`，挂在
`ggml_cuda_mul_mat` 的 mmvq 判断之后）：对 PTQ1_0 + Volta + 纯二维 MUL_MAT + 列数在
`(8, N]` 的情况，不再走「反量化成 fp16 → cuBLAS」，而是把激活按 PT 平面布局量化一次，
然后**按 8 列一块循环调用已有的 `mul_mat_vec_ptq1_0_pt` 融合内核**。开关：
`GGML_PTQ1_0_MULTI_CHUNK_MAX=<N>`，**默认 = 8（关闭）**。

### 24.2 单矩阵（test-backend-ops perf，PTQ1_0）

| 形状 | 融合 dp4a | stock（dequant+cuBLAS） | 加速 |
|---|---|---|---|
| m=17408 k=5120 n=16 | 170 µs | 2328 µs | **13.7×** |
| m=5120 k=17408 n=16 | 415 | 2180 | **5.2×** |
| m=17408 k=5120 n=64 | 666 | 2317 | 3.5× |
| m=5120 k=17408 n=64 | 1598 | 2198 | 1.4× |
| m=17408 k=5120 n=117 | 932 | 2644 | **2.8×** |
| m=5120 k=17408 n=117 | 911 | 2080 | **2.3×** |
| m=17408 k=5120 n=256 | （未测） | 2254 | — |

### 24.3 端到端：**打平**（10 轮 agent 形态，ctx 131072 / ub 1024 / MTP 开）

| 配置 | 10 轮总耗时 | 每轮 prefill | 解码 | 贪心哈希 |
|---|---|---|---|---|
| stock（部署版） | 44.3 s | 996 ms | 57.5 t/s | `f6c2b8350931f7ca` |
| 融合（上限 64） | 41.6 s | 938 ms | 60.1 t/s | **同哈希** ✓ |
| 融合（上限 256，m=117 真的走融合） | 45.3 s | 1055 ms | 58.9 t/s | **同哈希** ✓ |

三个数字都在 ±4% 的 run-to-run 抖动内 ⇒ **端到端是平局，不是胜利**。

### 24.4 为什么没传导？——新线索：每轮有 ~457 ms **根本不是内核**

用带融合的二进制重抓单轮：`prompt_ms` 826 ms，而**内核 busy 只有 369 ms**：

| 内核 | 融合前 | 融合后 |
|---|---|---|
| `dequantize_block_ptq1_0` | 174 ms | **80** |
| `cutlass::Kernel2` | 134 | **46** |
| `mul_mat_vec_ptq1_0_pt`（融合路） | 13.7 | **76** |
| `volta_s884gemm_fp16_*`（**f16 SSM 的 matmul，不是 PTQ1_0**） | 120 | **118** |
| 其它（FA/GDN/elementwise） | ~55 | ~48 |
| **合计** | 496 | **369（−26%）** |

即：**内核时间确实降了 26%，但 prompt_ms 只从 841 ms 降到 826 ms**——因为
**826 ms 里有约 457 ms 不是任何内核**（图构建、张量拷贝、prompt-cache 记账、采样准备、主机同步……）。
这才是那 1 秒的主项（占 55%），也解释了为什么再快的 matmul 都救不了这一轮。

⇒ **下一个（也是最后一个）明确目标**：把那 457 ms 的主机侧开销拆开。候选：`llama_context` 的
graph build / `llama_kv_cache_update`（部分前缀匹配后的 KV 处理）/ `ggml_backend_sched` 的
图重建、以及 CUDA graph 的 shape 匹配失败后重新构建。工具：`LLAMA_GRAPH_RESULT_DEBUG`、
`LLAMA_GRAPH_REUSE_DISABLE`、以及给 `llama_decode` 各阶段加埋点（SPC_MTP_PROF 的同款做法）。

### 24.5 处置

融合路**默认关闭**（opt-in）：它在单矩阵上确实快 2.3~13.7×、数值正确（贪心输出逐字节一致），
但端到端打平，没有再往生产推的必要；代码与开关留档，若以后模型/用量变成"短前缀 + 小 m"主导再启用。

---

## 二十五、那 457 ms 的主机侧拆解：**没有主机侧大头，是小 batch 的 GPU 空转**（2026-09-27 22:xx）

给 `llama_context` 加了 `SPC_DECODE_PROF=1` 埋点（`process_ubatch` 的
apply / build+alloc / inputs / compute，以及 `decode()` prologue 的
`sched_reserve` / `memory_update` / `init_batch`），跑一轮生产形态：

| 阶段 | 实测 |
|---|---|
| `sched_reserve`（每次 decode） | **0.0 ms** |
| `memory_update`（挂起的 KV shift/copy） | **0.0 ms** |
| `memory::init_batch` | 0.1~0.3 ms |
| 图复用判定 | **61 次调用里 60 次 reused=0**（KV 长度每轮都变 → 复用失败） |
| 图构建 + sched 分配（复用失败时） | **0.2~4.6 ms** |
| `set_inputs` | 0.1~5.6 ms |
| `graph_compute`（异步！只是发射时间） | 0.4~3 ms |

⇒ **主机侧根本没有大头**：所有这几项加起来每轮 < 10 ms。之前"826 ms 里 457 ms 不是内核"的判断，
是把 `graph_compute` 的返回当成了 GPU 完成时间——它是 `..._graph_compute_async`，立刻返回，
真正的 GPU 等待发生在后面的同步/回读里。

**所以那 457 ms 的真身是：小 batch（117 行）前向里 GPU 自己的空转**——
一轮里要发 ~6,000 个内核（普查：group 2 = 6,075 个），彼此有依赖、每个内核的块数都很少
（117 行 ÷ tile 大小），占用率低、尾效应重；"内核 busy 时长"之和（369~500 ms）本身就已经包含了
这些低效内核，而墙钟 840~1,100 ms。这与早前 MTP 剖析里"GPU 利用率只有 30~45%"的观测一致。

**结论**：这不是一个可以"修"的 bug，而是小 batch 前向的固有形态。要真的动它得走**算子融合**
（把 elementwise/norm/fwht 折进 matmul、减少内核数与依赖链）或者**多流并发**，都属于 llama.cpp
级别的重构，ROI 明显低于已经拿到的那些。**这条线到此收尾。**

埋点保留（`SPC_DECODE_PROF=1`，默认不打印），以后想复核任何时候都能跑。

### 25.1 收尾：把吃时间的内核按形状看了一遍（`work\tmp\turn-kernel-shapes.py`）

那一轮（117 token，6075 个内核、busy 369 ms）里最大的几项：

| 内核 | 形状 / 每次 | 合计 | 有效带宽 |
|---|---|---|---|
| `volta_s884gemm_fp16_128x64_ldg8_tn` | grid=(136,2,1)，608 µs | **118 ms（32%）** | 178 MB × 2 列tile / 608 µs ≈ **586 GB/s（65% 峰值）** |
| `dequantize_block_ptq1_0` | grid=(21760,1)，288 µs | 80 ms（22%） | |
| `mul_mat_vec_ptq1_0_pt`（融合路） | 2560 / 2902 tiles | 47 ms（13%） | ~314 GB/s |
| `cutlass::Kernel2` | grid=(8,5,4) | 46 ms（13%） | |
| `gated_delta_net_cuda` / `flash_attn_ext_f16` | | 10 / 9 ms | |

**结论：剩下的都不是"热点"，而是小 batch 的固有形态。**

* f16 的 GEMM 已经跑到 DRAM 峰值的 65%；
* 融合 dp4a 只有 35% 峰值，但它现在只占 13%（47 ms），即使翻倍也只值 23 ms/轮 ≈ 0.5%；
* 其余（内核实测带宽 35~65% 峰值 + 内核间隙）就是那 ~55% 的 GPU 空转，要动只能做算子融合/多流，
  属于 llama.cpp 级别重构。

---

## 二十六、整条线的收尾账（2026-09-27）

从"1.75bpw 三元 27B 在 V100 16GB 上跑起来"到现在，真正落到生产、并经数据验证的：

| 项 | 收益 |
|---|---|
| PTQ1_0 内核栈移植（planar mat-vec） | 解码从"跑不动"到 40 t/s 级 |
| sm70 D256 prefill FA（Split-D）+ 1.85× stock | pp16384 +8%、pp32768 +13.8% |
| MTP 草稿头嫁接 + 草稿 micro-batch 修复 | 解码 +26~38%；带 MTP 的 prefill 522→802 t/s |
| Volta hs=256 FA 配置重调 + np>1 栅栏 UB 修复（上游 #27955） | 消除 8576 个 UB，PPL 逐位一致 |
| GDN 列并行 cols_per_warp 4 | pp 全档 +2.6~2.8% |
| ubatch 分档（32K/64K → 2048） | 短档预填充 +7.4%、解码 +12~25% |
| 面板：Agent 档按多轮 agent 负载的实测形态定标 + tooltip 写明依据 | — |

**试过并否证的**（都留了档，别再试）：chunked GDN、KVMem、PTQ1_0 MMQ（大 batch）、
KV 均值中心化、单列 PT 布局、强制 vec、解码专用内核（新写，nb=1 快 8~21% 但生产打平）、
融合小 m GEMM（单矩阵 2.3~13.7×，端到端平局）、主机侧开销（实测 < 10 ms，不存在）。

**当前稳态**：12K 预填充 ~850-880 t/s、解码 43-60 t/s；100K 预填充 610-740 t/s、解码 26-35 t/s
（取决于输出可预测性）。每轮 agent 往返 ~4.3 s = 预填充 ~1 s + 解码 ~3.3 s，
其中预填充的 ~55% 是小 batch 的 GPU 空转（固有形态）。
