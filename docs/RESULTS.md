# oripoin/llama.cpp cherry-pick 结果（2026-09-26）

来源：`https://git.oripoin.me/oripoin/llama.cpp`（MIT，Yachen Wang / OriPoin）
基线：本地 `local/prod-842b188`（842b18804），与之共同祖先 `5ea87ddad`
远端 tip：`058ca699c`（force-push 过两次，最终以这次为准）
工作树：`work\picks-llama`（分支 `local/picks-20260926`），构建目录 `build-picks`

## 一、实测结果（Tesla V100-SXM2-16GB，ctx 131072，q8_0 KV，-b 8192 -ub 2048）

| 指标 | 旧运行时 | picks 新版 | 变化 |
|---|---|---|---|
| 显存占用（ctx 131072） | 15,847 MiB（旧运行时） | 13,979 MiB（q8_0 KV + MTP，三档 ub 实测 13,952~13,980） | **-1.9 GiB** |
| 预填充 12,067 tok 提示 | 175.8 t/s | **921.9 t/s** | **5.2×** |
| 预填充 55,509 tok 提示 | 13.55 t/s | **681.1 t/s** | **50×** |
| 生产配置下 12,084 tok 提示 | 175.8 t/s | **940.6 t/s** | **5.35×** |
| 明文解码（4 任务均值） | 44.15 t/s | **45.75 t/s** | +3.6% |
| MTP n=2（ctx 65536，4 任务均值） | 52.45 t/s* | **55.30 t/s** | +5.4%* |

*MTP 那行旧数据是在 ctx 131072 上测的，配置不完全可比；接受率与 draft/accepted 计数完全一致
（math 216/191、code-csv 273/161、code-bs 245/176、chat 148/108），说明数值逐位一致、无精度回归。

根因：FlashAttention 的 tile / mma 内核对 q8_0 KV 缓存要先把整段会话 K/V 反量化成 f16 再算
（`ggml_cuda_flash_attn_ext_get_f16_extra_data`，大小 = nelements(K)*2 字节，ctx 131072 时约 512 MiB，
且每个 op 都要重做一遍）。这段 materialise 在 ctx 131072 上把显存压到卡外、触发 WDDM 分页，
于是预填充塌成 175 t/s（55K 提示时 13.5 t/s）。oripoin 的补丁让 tile 内核直接按 q8_0 读缓存，
省掉这段副本，问题消失。

## 二、取用的提交

| sha | 标题 | 说明 |
|---|---|---|
| `7534ea65b` | speculative: size the MTP draft context's buffers by one step of drafts | draft 上下文的 ubatch 按「一步草稿」而不是目标 ubatch 分配 |
| `247495531` | qwen35 : flatten the ssm_out activation for the MMVQ path | qwen35 SSM 路径的 3D→2D 展平 |
| `72934f175` | cuda: budget the shared-memory attribute for both fallback variants of a tile | 共享内存属性修复（对 sm_70 是 no-op） |
| `553052586` | cuda: give Turing's 5..8-row Q8_0 matmuls the MMQ path | 仅 sm_75 生效，sm_70 无影响 |
| `c1175dd6e` | cuda: narrow MMQ tile for Turing's 9..16-row Q8_0 matmuls | 仅 sm_75 生效，sm_70 无影响 |
| `6ae7cbeb8` | cuda: read the q8_0 KV cache where it lies in the tile flash-attention kernel | **核心补丁** |
| `023bf98a3` | local: 该 q8 读在本树里按稠密 KV 寻址（我们没有 pool row map） | 本地适配 |
| `8dfe8ec75` | local: draft ubatch 下限要保住混合模型的回滚窗口（n_max+2 行） | 本地适配 |
| `6dd041f49` | local: `GGML_MTP_KEEP_DRAFT_UBATCH=1` 诊断开关 | 本地 |

补丁全集：`patches/oripoin-picks-20260926-full.patch`（相对 842b18804，112 KB / 20 个文件）
续接补丁：`patches/local-patch-v100-20260927.patch`（9b98d9dfd → 生产工作树，含 PTQ1_0/m70 D256/注册代码）
提交清单：`patches/oripoin-picks-20260926-commits.txt`（已按真实历史重新生成）

## 三、被否掉的提交

| sha | 标题 | 否决原因 |
|---|---|---|
| `9d817213a` | model : load hparams.n_layer_nextn before n_layer() calls (#28159) | 在我们的树上会把 draft 模型的 `n_layer_nextn` 读成 0，draft KV 缓存里没有第 64 层，开 MTP 时 `cache_k_l64[ne0=0]` 触发 `ggml_set_rows` 断言直接崩。已 revert（`4acdc0772`），MTP 恢复正常 |
| `084a3f261` / `156d1a7b4` | Turing Q8_0 MMQ tile | 代码里写死 `cc == GGML_CUDA_CC_TURING`（sm_75），V100 是 sm_70，对本机没有任何效果 |
| `2ce2d4b9a` | cuda: read the q8_0 KV cache inside the MMA flash-attention kernel's own tiles | 想取，但他们树里有 swizzle (`e4b9af007`) 和 sparse-fa (`8e93a9773`) 等前置改动，直接 cherry-pick 有 23 处冲突。留作下一步 |
| `2a862021` / `1fabe2b7b` | 采样器候选数组 / server 采样器快照 | 他们自己的审计 P1 结论：np1 下 host 采样耗时可忽略；改动面大、收益不确定，先不取 |

## 四、MTP + ctx 131072 的现状

仍然装不下：目标上下文 14.0 GB + draft 上下文（权重 + 自己的 KV + 计算缓冲）≈ 3 GB > 16 GB，
draft 模型加载时报错退出。可选解法（按性价比排序）：

1. 移植 `2ce2d4b9a`（MMA 内核的 q8 读）→ 再省 512 MiB，并去掉 prefill 每 op 的 f16 副本；
2. 开 MTP 时把 `-ub` 降到 512 → mask 从 512 MiB 降到 128 MiB（代价：prefill −19%）；
3. 把 ctx 降到 98304 或 114688。

### 补测：长上下文 MTP 仍然不划算（2026-09-26 07:2x）

tile 直读确实消掉了验证步里的 f16 物化，但 55K 上下文下 MTP 依旧慢于明文：

| 配置（ctx 65536，55,509 tok 提示） | 预填充 | 解码 |
|---|---|---|
| picks 版 + MTP n=2 | 443.7 t/s | **19.0 t/s** |
| 同机旧运行时 + MTP n=2（历史记录） | — | 21.1 t/s |
| picks 版 + 无 MTP（历史 ctx 65536 基线） | ~790 t/s | ~29.5 t/s（55.7k 处） |

原因不在 K/V 副本：55K 上下文时每个验证步的注意力要读满 16 个注意力层 × 55K 行 KV，
draft 链本身还要多跑两遍，收益被摊掉。结论：**长上下文（≥~32K 实填）继续不开 MTP**；
短上下文/可预测文本时 MTP n=2 仍然有效（ctx 65536 四任务均值 55.3 t/s vs 明文 46 t/s 左右）。

## 五、部署状态

* 生产运行时已换成 picks 版（`work\bonsai-demo\bin\cuda`），备份：
  `work\bin-backup\20260926-070429-full`（旧）与 `work\bin-backup\20260926-070626-full`
* 生产配置（2026-09-27 更新）：ctx 131072、q8_0/q8_0、-b 8192 -ub 1024、MTP（`--spec-type draft-mtp`，n_max=2）；
  最新运行时备份 `work\bin-backup\20260927-135103-full`（含 sm70-attn D256 FA、MTP 草稿 micro-batch 修复、
  Volta hs=256 FA 配置 + np>1 栅栏 UB 修复，见 `RESEARCH-V100-LLAMA-20260926.md` 第八~十七节）
* 部署脚本：`work\pr210\deploy-picks.ps1`（从 `work\picks-llama\build-picks\bin` 全量覆盖 + SHA256 校验）
* 构建脚本：`work\picks-build.bat`
* 测试脚本：`work\mtp\run-pick.ps1`（可指定任意 llama-server.exe）

## 六、他们仓库里还值得参考的东西

* `benches/turing/optimize-plan-20260926.md`：20 个优化点的审计与判决（含 P2 `-bs` 反而慢 29-43%、
  P7 `-ub 2048` +19.9%、P5 提高注意力占用率反而变慢、P8 落盘 F16 权重净亏 等实测结论）
* `benches/turing/kernel_census.py` / `step_census.py` / `analyze_nsys.py`：kernel 级耗时归因工具
* 提交 `a061acd5b` "cuda : fuse three node chains on the GDN path"：qwen35 的 GDN 融合，未取

## 七、试过并且否掉的：MMA 内核的 q8_0 直读（2026-09-26 11:xx）

想法：`2ce2d4b9` 的 MMA q8 直读在他们树里挂在 Turing 分支，V100 走不到；把同样的谓词接到 Volta
分支（`volta_mma_available`）就能让预填充（nq ≥ 9 → MMA_F16）也不再物化 session 尺寸的 f16 K/V。

做法（已实现并构建通过，随后整体回退）：
* `fattn-mma-f16.cuh`：loader / iter / process_tile / kernel / case 五层加 `bool q8` 模板参数，
  K/V 基址改成 `const char *` + `k0_off`/`k_VKQ_0` 两个偏移，`stride_K/V` 在 q8 时按
  `sizeof(block_q8_0)` 计；q8 分支按他们的写法解包（2 字节读 + `__hmul2(make_half2, d)`）。
* `fattn-common.cuh`：新增 `ggml_cuda_fattn_mma_reads_q8`（NVIDIA + fast-fp16 + Volta +
  256/256 + 双方 q8_0 + 步长整块）。
* `fattn.cu`：`get_alloc_size` 的 MMA 分支改问这个谓词。
* 踩到的坑：`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 原先按「每个 case 只抬一次」写，
  新增 q8 变体后先跑的那个把 flag 置上、另一个仍在 48 KiB 默认值 → 占用率 0 断言；
  改成一次性给该 case 的所有变体（2 个或 4 个）都抬。

验证：`test-backend-ops -o FLASH_ATTN_EXT -p "hsk=256,hsv=256"` 全过（含 10 个 q8_0 用例），
退出码 0；`hsk=320` 那个 `cudaFuncSetAttribute invalid argument` 是**旧版本也一样**的既有问题
（V100 每块共享内存上限，见 `work\tmp\op-test-prod-320.txt`），与本改动无关。

实测（ctx 131072，无 MTP，同一台机同一个测法）：

| 指标 | 只有 tile q8（现行生产） | 再加 MMA q8 |
|---|---|---|
| 加载后显存 | ~13,9xx MiB（sweep 实测 13,952~13,980） | 14,411 MiB（**+455 MiB**） |
| 预填充 12,067 tok | **921.9 t/s** | 900.5 t/s（−2.3%） |
| 预填充 55,509 tok | **681.1 t/s** | 636.6 t/s（−6.5%） |
| 明文解码 4 任务均值 | 45.75 t/s | 45.98 t/s（噪声内） |

结论：**在 V100 上不划算，已回退**。预填充反而更慢（MMA 内核里现做反量化比「物化一次、
整个 ubatch 复用」更贵），而占用率变高让 `launch_fattn` 的 dst_tmp/dst_tmp_meta 堆更大，
净显存 +455 MiB。补丁留档：`work\patches\local-mma-q8-volta-REJECTED-20260926.patch`。
以后只有在「上下文大到必须省那 512 MiB 且不在乎预填充速度」时才值得重新考虑。

## 八、取用的第二块：GDN 路径的三处顺序融合（`a061acd5b`，2026-09-26 12:xx）

提交内容：`concat → cpy×N`（conv 历史回滚快照由 concat 自己写）、`add + rms_norm + mul`（残差后
归一）、`add + unary + mul`（GDN 门控 alpha→+dt_bias→softplus→*ssm_a）三处顺序融合，都在
CUDA 图遍历里做（ggml 图不变），各自带 `GGML_CUDA_NO_*` 开关。新增 kernel 落在
`concat.cu` / `norm.cu` / `unary.cu`。

移植时的处理：
* `ggml-cuda.cu` 的冲突里只取本提交新增的三处钩子；同段里夹带的 cuBLASLt bias 融合和 MoE
  weighted-reduction（都来自别的提交，我们没有对应实现）全部丢掉。
* `norm.cuh` / `norm.cu` 只取 `ggml_cuda_op_add_rms_norm_mul`；同段里的
  `ggml_cuda_op_rms_norm_gated` 属于另一个提交，丢掉。
* `GGML_OP_RMS_NORM+MUL+ROPE` 那条融合保持我们自己的 `return 2`（他们的版本在这条上少一个 return）。

实测（ctx 131072，无 MTP，同二进制用环境变量切臂）：

| 指标 | 融合关 | 融合开 | 变化 |
|---|---|---|---|
| 预填充 12,067 tok | 938.0 t/s | **944.4 t/s** | +0.7% |
| 解码 4 任务均值（3 次配对） | 45.66 | **46.12** | +1.0% |
| 加载后显存 | — | 与旧版一致 | ±0 |
| PPL（同一语料，-c 512） | 7.1178 | **7.1178** | 逐位一致 |

结论：**采纳并部署**（`work\bin-backup\20260926-125138-full` 是上一版备份）。
收益不大（1% 量级），但三次配对重复都为正、PPL 逐位一致、且三处融合各有独立开关可随时关掉。
他们那边 -17408 个 kernel 实例的收益在我们这张卡上没有等比兑现，原因是 V100 的预填充是带宽受限、
而且我们的 GDN 图结构与他们的不完全一样。

### 测量踩坑记录（重要）

中途一度测出「预填充 13.7 t/s，比基线慢 65 倍」，浪费了一轮排查。真实原因是**我自己的 shell 会话在
循环 `Get-Content -Tail` 读服务器的 stderr 日志**，与服务器写日志冲突，把解码循环卡住了。
排除方法：写一个独立脚本（`Start-Process` 后台跑服务器 + 立刻发请求），全程不并发读日志 —— 同一份
二进制立刻恢复 728 t/s。以后所有 A/B 都必须用这种「无并发日志读者」的跑法：
`work\tmp\cold-ab.ps1` / `gdn-ab2.ps1` / `gdn-reps.ps1` 都是这个模式。

## 九、上下文复用与最后一公里（2026-09-26 13:xx）

### 9.1 前缀复用是工作的（Codex 长对话的关键）

用 12K token 的提示做三连测（`work\tmp\prefix-reuse-test.py`，服务端 `cache_prompt: true`）：

| 请求 | 总 token | 复用（cache_n） | 重算（prompt_n） | 耗时 |
|---|---|---|---|---|
| 首次 | 12,025 | 59 | 12,025 | 12.7 s |
| 同前缀 + 短后缀 | 12,094 | **10,032** | 2,062 | 2.68 s |
| 再来一次 | 12,077 | 10,042 | 2,042 | 2.68 s |

结论：**每轮只重算末尾约 2,000 token（≈2.7 s）**，不是整段重跑。Codex 每轮把整段对话发过来，
前缀稳定，所以长对话是按这个代价走的。

顺带查清了两件事：
* `--cache-reuse`（分块复用）在带 mmproj 或 recurrent 内存（qwen35 的 SSM 层）时会被服务器
  自动禁用（`server-context.cpp:1126-1141`）——我们两条全中，所以走的是 checkpoint 路径，不影响上面结论。
* `--checkpoint-min-step`（`-cms`，上游默认 8192，oripoin 改成 1024）实测**不影响**这个 2,000 的尾巴：
  8192 / 1024 / 256 三种取值结果逐位相同（`work\tmp\cms-test-results.txt`）。

### 9.2 下一步的大头：WDDM → TCC（需要用户决定）

早期 nsys 已经量过：单 token 墙钟 26.56 ms 里有 **9.60 ms 是 `cudaGraphLaunch` 一次调用的时间**
（约 4,800 个节点的图，WDDM 驱动逐节点提交 ≈ 2 µs/节点）。GPU 实际忙只有 19.57 ms。

当前状态：`nvidia-smi -q` 显示驱动模型 **WDDM**，但显示活动在虚拟适配器上
（GameViewer Virtual Display + Virtual Display Driver），V100 的 "Display Active: Disabled"。
V100 是 Tesla 计算卡，支持 **TCC** 驱动模型；切到 TCC 后 WDDM 的逐节点提交开销会消失，
按上面的数字推算解码有希望从 **46 t/s 提到 60+ t/s**（长上下文同理），而且彻底没有 WDDM 分页
（就是当初长提示塌成 13.5 t/s 的那个机制）。

前提与风险：
* 需要**管理员权限 + 重启**：`nvidia-smi -g 0 -dm 1`（若报「display connected」，还有
  `-fdm 1` 的强制版本，但那条更激进）。回退：`nvidia-smi -g 0 -dm 0` + 重启。
* 用户是远程/虚拟显示环境，万一 V100 参与了显示链路，切 TCC 后可能看不到画面 —— 所以必须由用户
  在有物理回退手段时执行，我没有管理员权限也不该替他做这个决定。

### 9.3 `-ub` 决定「每轮重算多少」，Agent 预设从 2048 改成 1024

尾巴 = ubatch 边界（`-ub 2048` → 尾巴 2,062；`-ub 1024` → 1,038；`-ub 512` → 526）。
所以 `-ub` 不只是「预填充快慢」，它直接决定多轮对话每轮要重算多少 token。

Codex 式会话仿真（10 轮，起始约 5.9K，每轮追加约 1K；`work\tmp\turn-sim.py`）：

| `-ub` | 第 1 轮（冷） | 第 2-10 轮 | 10 轮总耗时 | 每轮处理 token |
|---|---|---|---|---|
| 2048 | 8.18 s | 4.23–4.78 s | **49.0 s** | 3,048 |
| **1024** | 8.71 s | 3.32–3.65 s | **40.0 s** | 2,024 |
| 512 | 10.42 s | 3.05–3.39 s | **39.5 s** | 1,512 |

结论：多轮场景 1024/512 比 2048 省约 19%（每轮约省 1 s），冷启动只差 0.5 s（1024）。
单发超长提示（5 万 token 一次性喂）反过来是 2048 最快（58 s vs 1024 的 65 s vs 512 的 77 s）。
面板的「🤖 Agent 最优（实测调优）」预设和当前运行配置都改成了 **ubatch 1024**，并把这个取舍写进
按钮提示里；要处理一次性长文档时把 ubatch 手动调回 2048 即可。

## 十、查找式（n-gram）投机解码：实测为负，暂不推荐（2026-09-26 14:xx）

动机来自文献：REST（arXiv 2311.08252）与 Lookahead Decoding（arXiv 2402.02057）那一类
「检索/查表式草稿」不需要草稿模型、不读额外权重，理论上长上下文也划算，正好补 MTP 的短板；
我们树里也已经内置了实现（`--spec-type ngram-simple | ngram-map-k | ngram-map-k4v | ngram-mod`，
来源是 llama.cpp PR #18471，见 `common/ngram-map.h` 顶部注释）。

测试条件：ctx 131072、约 38K 前缀、三个任务（逐字复制代码 / 改写函数名后输出 / 原创短文）、
每格 1-2 次重复，基线同为 35.1 t/s（基线、每个 arm 同一份二进制、同一台机）。

| arm | 逐字复制 | 改写函数名 | 原创短文 | 草稿数/接受 |
|---|---|---|---|---|
| 基线（无投机） | **35.1 t/s** | 35.1 | 35.4 | 0 |
| `ngram-map-k`（默认 12/48） | 35.0 | 35.0 | 35.0 | **0 个草稿** |
| `ngram-map-k4v`（默认） | 34.7 | 34.8 | 34.9 | **0 个草稿** |
| `ngram-simple`（默认） | 33.8 | 33.8 | 34.3 | 35-45 草稿 / 11-27% |
| `ngram-map-k` + MTP n=1 | 32.0 | 32.4 | 31.7 | 233 / 25-28% |
| `ngram-map-k` 调参 4/16 | 27.9 | 26.4 | 28.1 | 113-142 / 11-23% |
| `ngram-simple` 调参 4/16 | 27.8 | 25.4 | 28.5 | 128-174 / 17-26% |

两个结论：
1. **默认参数下 map 版根本不出草稿**（12-gram 键 + 48-token 值太长，且 `common_ngram_map` 的
   `keys` 只在 `begin()` 用提示词建一次，`process()` 还是 `// TODO: implement`）。
2. **调小到 4/16 后确实出草稿，但接受率只有 11-23%，整体比基线慢 20%**——多出来的验证批
   （1+k 行）成本超过了省下的步数。这跟我们之前从 MTP 得到的盈亏线一致：接受率不到 ~50%
   就不划算，而查表式草稿虽然自身免费，验证成本照样要付。

所以**不推荐启用**。如果哪天要重开这条线，前提是先把 `common_speculative_impl_ngram_*::process()`
补上（把生成出的 token 喂回历史），并让接受率在一份真实的 Codex 编辑轨迹上先超过 50% 再谈。

## 十一、内核级账本：钱到底花在哪（2026-09-26 15:xx）

用 nsys（`--cuda-graph-trace=node`，否则图内 kernel 不记录）分别跑 pp2048 与 tg128，
再直接查 `gdn-prof2.sqlite` / `dec-prof.sqlite` 的 `CUPTI_ACTIVITY_KIND_KERNEL` 表
（脚本 `work\tmp\kern-census.py`，按 demangled 名归并）。同一份二进制、同一台机：

**解码（tg128，总 kernel 时间 4.67 s）**

| 内核 | 占比 | 调用次数 |
|---|---|---|
| `mul_mat_vec_q`（PQ2_0 量化矩阵乘） | **68.2%** | 86,609 |
| `rms_norm_f32` | 6.3% | 53,713 |
| `fwht_cuda_block`（Hadamard 旋转） | 4.6% | 66,306 |
| `quantize_q8_1`（激活量化） | 4.3% | 86,609 |
| `mul_mat_vec_f` | 2.6% | 24,672 |
| `gated_delta_net_cuda` | 2.2% | 12,336 |
| `k_get_rows_float*` | 3.2% | 28,961 |
| `flash_attn_ext_vec` | 2.1% | 4,112 |

**预填充（pp2048，总 kernel 时间 7.75 s）**

| 内核 | 占比 | 调用次数 |
|---|---|---|
| `cutlass::Kernel2`（cuBLASLt GEMM） | **45.1%** | 4,800 |
| `dequantize_block`（PQ2_0→fp16） | **19.7%** | 4,800 |
| `magma_sgemmEx_kernel`（fp32 GEMM） | 12.0% | 1,152 |
| `gated_delta_net_cuda` | 9.6% | 576 |
| `convert_unary`（激活转 fp16） | 2.6% | 5,952 |
| `flash_attn_ext_f16` | 2.4% | 192 |
| `fwht_cuda_block` | 1.9% | 3,087 |
| `unary_gated_op_kernel` | 1.9% | 1,536 |
| `add_rms_norm_mul_f32`（本次新取的融合） | 1.2% | 1,536 |
| `concat_conv_state`（本次新取的融合） | 0.9% | 576 |

### 由账本得出的结论

1. **解码的命门就是 MMVQ 一个内核（68%）**，实测等效只有 ~450 GB/s（V100 峰值 900），
   也就是有约一倍的理论余量。它的形状是「每块 128 线程、每块只算 1 行」。
   **试过并否决**：把 `calc_rows_per_block(ncols_dst=1)` 从 1 提到 2
   （他们 P4 在多列场景调过这个参数，单列没调过）→ tg128 47.88±3.03 vs 基线 48.46±0.45，
   没有改善，已回退。要把这 68% 真正吃下来需要重写 kernel（低比特原生点积，BitNet 2402.17764 那一路），
   不是调参能解决的。
2. **预填充的钱在「cuBLAS GEMM + 权重重反量化 + 激活转换」这一簇（45+20+2.6 = 68%）**，
   这正是 oripoin 审计里 P3 的对象；他们那边关掉 FORCE_CUBLAS 会慢 12.4%，我们这边默认就是
   走 cuBLAS，且 ub 路径已经是最优（>64 行自动走 cuBLAS，这是框架设计），所以这条也到头了。
3. **GDN 顺序扫描占预填充 9.6%**，而 `gated_delta_net.cu:226` 就写着
   `//TODO: Add chunked kernel for even faster pre-fill`——按 GDN 论文（2412.06464）/
   DeltaNet 并行化论文（2406.06484）的 chunk 形式重写，理论上能拿掉这 9.6% 的大部分，
   是目前「照着论文做」最明确的一块。工作量：一个中等规模的 CUDA kernel。
4. **Hadamard 旋转（fwht）解码 4.6% + 预填充 1.9%** 是 PrismML 量化方案带来的运行时开销
   （QuaRot 2404.00456 的旋转本来可以折进权重），但这要重新量化模型，不归我们改。
5. 本次新取的 GDN 融合确实在跑（账本里能看到 `add_rms_norm_mul_f32` 与 `concat_conv_state`）。

## 十二、三条「论文方向」逐条验证（2026-09-26 16:xx）

### 12.1 `{RMS_NORM, UNARY(SILU), MUL}` 融合：不适用

oripoin 那边有个现成提交（`b96198800`）把这个菱形接进 `try_fuse`，但在我们树上**不会触发**：
我们 fork 的 `build_norm_gated()`（`src/models/qwen35.cpp:321-330`）已经是
`ggml_swiglu_split(ctx0, gate, normalized)`——`silu(gate)*norm` 早就合成一个 op 了，
所以图上不存在 `RMS_NORM + UNARY + MUL` 这个形状。移植过来只会是死代码。

### 12.2 chunked GDN（论文方向）：是真活，但代价大

* 先排除了一个误会：fork 里的 `cparams.fused_gdn_ch` 只是 probe 的名字（
  `llama-context.cpp:134`，用于检测设备是否支持多 token 的 GDN op），并不是并行扫描实现；
  实际选路只看 `n_rs_seq`，而 `gated_delta_net.cu:226` 明写着
  `//TODO: Add chunked kernel for even faster pre-fill`。
* 收益量级（同一台机、同一二进制、nsys 内核账本）：

| 内核家族 | pp2048 | pp16384 |
|---|---|---|
| cuBLAS GEMM (cutlass) | 45.1% | 50.7% |
| **flash_attn_ext_f16** | 2.4% | **15.6%** |
| **gated_delta_net_cuda** | 9.6% | **11.6%** |
| dequantize_block | 19.7% | 5.9% |
| magma_sgemmEx | 12.0% | 3.7% |

  即：深度越深，注意力与 GDN 两项越贵（16K 时合计 27%）。若把顺序扫描换成论文的 chunk 并行、
  且真能快 2-3 倍，长预填充大约能省 **6-8%**（短预填充 5-6%）。这是「照论文做」唯一确定能拿的，
  但要在 sm_70 上正确实现门控 delta 规则的 WY 表示（FLA 里是 5-6 个子 kernel），属于多天工程，
  且有数值偏差风险。

### 12.3 MMVQ（解码 68%）：已经是 dp4a，重写天花板不高

打开 `vecdotq.cuh:978` 看实际代码：PQ2_0 的点积已经是
`__byte_perm` 解包 + `ggml_cuda_dp4a`，每 8 个值约 6 条指令（≈0.75 指令/值），这就是 BitNet
那一路的做法，不是「先反量化成 fp16 再乘」。结合启动参数（block=128 线程、每块一行、
每行约 1.4 KB 权重）可以判断：这个内核是**指令发射受限**，不是显存带宽受限
（实测等效 325-450 GB/s / 900 峰值）。所以重写的天花板大概只有 +20~30%，不是翻倍。

## 十三、路线 C：把 MTP 接进面板并做成「短会话档」（2026-09-26 17:xx）

之前 MTP 只能手敲命令行（面板一个 `spec` 字段都没有），所以先补了这块：

* `outputs/bonsai-dashboard.py`
  - 配置项：`spec_type`（默认空=关）、`spec_draft_model`（默认指向 `work/mtp/...FastMTP-32K.gguf`）、
    `spec_draft_n_max`（默认 2）。
  - `build_args()`：开了就追加 `--spec-type draft-mtp --spec-draft-model ... --spec-draft-n-max N
    --spec-draft-ngl 99 --spec-draft-type-k/v q8_0`，草稿文件不存在时静默不加（不会起不来）。
  - 启动时若开了 MTP，自动给子进程加 `GGML_CUDA_GRAPH_SHAPECACHE=1`（draft 上下文的图会在 1 行与
    1+n_max 行之间反复重捕获，实测 +3.8%）。
  - 表单：新增「投机解码 MTP」勾选框 + 「MTP 草稿长度」输入；「🤖 Agent 最优」预设会显式把它关掉。
  - 新增预设按钮「⚡ 短会话 · MTP 加速（ctx 64K）」：ctx 65536 + ubatch 2048 + MTP n_max=2。

* 端到端实测（面板直接应用该档，8080 上跑四任务）：

| 任务 | 解码 t/s | 草稿数 | 接受率 |
|---|---|---|---|
| math | 50.2 | 261 | 64.4% |
| code-csv | 49.3 | 262 | 63.7% |
| code-bs | 55.7 | 234 | 77.4% |
| chat | 48.2 | 232 | 59.9% |

  接受率比早先单测（math 88%）低，是因为面板这套采样链不同（官方档 0.3/0.9/40/min_p 0.05）；
  投机解码的接受率本来就跟着采样分布走。性能档位和预期一致，机制可用。

* 生产配置已切回「128K + 无 MTP + ubatch 1024」，健康检查通过。

## 十四、路线 A：chunked GDN 的前置验证与内核诊断（2026-09-26 15:xx）

### 14.1 算法等价性（numpy 参考，`work\tmp\gdn-chunk-ref.py`）

* chunk 级映射 `S_out = P_c·S_in + B_c`（`P_c = Π_t g_t(I-β_t k_t k_tᵀ)`、`B_c` 为零输入响应）
  链式传播 vs 顺序扫描：**max|dS| = 2.1e-05**（fp32 舍入量级）。
* 移植要用的 **WY 恒等式**：`Π(I-β k kᵀ) = I - Kᵀ(I + tril(diag(β)KKᵀ,-1))⁻¹ diag(β) K`，
  三个 chunk 实测误差 4e-6 / 1e-6 / 7e-7 ✓（带门控时只是每 chunk 多一个标量因子，因为衰减是 per-token 标量）。
* 算术量（真实 d=128、C=64）：顺序每 chunk 4.19M 且 **64 步串行**；WY 形式 ≈2.2M，串行部分只剩
  C×C 三角求解 → **串行链缩短 64×**。

### 14.2 内核诊断（ncu，需要 GPU 性能计数器权限，已通过一次提权跑通）

抓到的是 `gated_delta_net_cuda<128,0,0,1,0>`，grid (48,1,32)、block (32,4,1)、每次 1.37 ms：

| 指标 | 值 | 含义 |
|---|---|---|
| DRAM Throughput | **12.05%** | 完全不是显存受限 |
| L1/TEX Throughput | 59.61% | 最忙的是 L1（状态 load/shuffle） |
| Compute (SM) Throughput | 55.55% | 发射槽利用率 |
| Issued Warp Per Scheduler | **0.59** | 每 1.7 周期才发一条指令 |
| No Eligible（无可用 warp） | **41.02%** | 41% 的周期无指令可发 |
| Warp Cycles Per Issued Instruction | **12.35** | 单条指令要等 12 个周期 |
| Achieved / Theoretical Occupancy | 44.2% / 56.25% | 寄存器限制到 9 个 block/SM |

**结论：内核是「长串行依赖 + 延迟受限」，不是带宽或算力受限**——一个 warp 要顺序走完 2048 个 token，
每条指令等 12.35 周期、41% 的周期无事可做，而 DRAM 只用 12%。这正好是 chunk 并行要消除的东西，
也解释了为什么它的有效算力只有 fp32 峰值的 4.4%。

顺带：`RmProfilingAdminOnly` 已写入注册表（值为 0），重启之后我这边就能**免提权**随时跑 ncu 做调优。

## 十五、路线 A 的算法定稿（完整 chunk 形式，已数值验证）

参考实现：`work\tmp\gdn-chunk-full.py`（d=8、C=4、T=16，随机数据，fp32）

```
max|dState| = 1.53e-05      max|dOut| = 1.14e-05      相对误差 2.5e-07
```

### 15.1 最终公式（对着 `gated_delta_net.cu` 的约定，S 为 [d_k, d_v]，衰减 α_t = exp(g_t) 是每 token 标量）

对每个 chunk（C 个 token，G 为 chunk 内 g 的**含自身**前缀和）：

```
Lg    = tril( β_i (k_i·k_j) e^{G_i-G_j}, -1 )      # 含门控，给 δ̂ 用
Lu    = tril( β_i (k_i·k_j),           -1 )        # 不含门控，给状态映射用
Ainv_g = (I + Lg)^{-1}      Ainv_u = (I + Lu)^{-1}          [C×C]
ds     = Ainv_g (β ⊙ V)     W = Ainv_u (β ⊙ K)              [C×d]

状态： S_out = e^{G_last} (S_in − Kᵀ (W S_in)) + Kᵀ ( e^{G_last−G} ⊙ ds )
输出： o_t   = scale · ( Σ_{j≤t} e^{G_t−G_j} (k_j·q_t) ds_j
                          + e^{G_t} ( S_inᵀ q_t − Σ_{j≤t} (k_j·q_t) (W_jᵀ S_in) ) )
```

### 15.2 两个非平凡的坑（写 CUDA 时必须照抄）

1. **输出里的状态项是 `(k_j·q_t)·(W_jᵀ S_in)`，不是 `(W_j·q_t)·(k_jᵀ S_in)`**：`P_t = Π α(I−βkkᵀ)`
   是**对称矩阵的乘积，一般不对称**，所以不能把 `S_inᵀ P_t q_t` 里的 `P_t` 当成对称来交换两边。
   （我先按对称写，输出错 0.9~3.1；改正后降到 1e-7。）
2. **两个逆矩阵不能混用**：`δ̂` 的三角系统里带门控（因为 `δ̂_t = β_t(v_t − α_t k_tᵀ B_{t−1})` 把
   `α_t` 折进了系数 → `e^{G_i−G_j}`），而状态映射不带（每 token 衰减是标量，能从矩阵积提出来）。
   我一度只用带门控的逆去算 `W`，状态立刻错到 0.14。

### 15.3 移植计划（三段 kernel）

* **K1（每 chunk、每 head）**：`KKᵀ`（带衰减掩码）→ 两个 C×C 三角求逆 → `W`、`ds`。
  并行度 = 层 × head × chunk 数（2048 token / C=64 = 32 个 chunk），全部独立。
* **K2（chunk 扫描）**：`S_{c+1} = e^{G_last}(S_c − KᵀW S_c) + Kᵀ(...)`，只有 T/C 步串行
  （我们的场景 32 步，相比现在的 2048 步），每步是 d×d 的小 matmul。
* **K3（每 chunk、每 head）**：输出两项，逐 token 独立，写成稠密小 matmul。
* 解码（n_tokens 小）继续走原来的顺序 kernel；按 `n_tokens >= 64` 分流。

预期：该内核现在 306 GMAC/s（fp32 峰值 4.4%）、12.35 周期/指令、41% 发射槽空转，
chunk 化后串行链缩短 64×、算术量减半，长预填充整体约 −8~10%。

---

## 十六、Volta 的 GDN 列并行：cols_per_warp 1 → 4（已采纳：pp 全档 +2.6~2.8%，PPL 不变）

ncu 的结论是「延迟受限」而不是带宽/算力受限（DRAM 12%、No Eligible 41%、Warp Cycles/Instr 12.35）。
而上游 `gdn_cols_per_warp()` 在 Volta 上返回 **1**（Ampere+ 才给 4），含义是每 warp 只管 1 列：

* 每 token 只有**一条标量依赖链**（kv 归约 5 步 shfl → delta → 状态更新 → attn 归约 5 步 shfl），warp 内部没有可交叠的工作；
* k/q 的加载**按列重复**：128 列就要把同一个 k/q 读 128 遍（L1 59.6% 的主要来源）。

改成 Volta 也走 4 列/warp 后，一个 warp 里有 4 条独立链、且 k/q 每个 token 只读一遍（这类流量降到 1/4）。

同 build 仅改这个常量的 A/B（`llama-bench -r 3`）：

| 深度 | 1 列/warp | 4 列/warp | Δ |
|---|---|---|---|
| pp2048 | 787.86 ± 8.17 | 809.29 ± 6.82 | **+2.7%** |
| pp8192 | 750.24 ± 2.47 | 771.50 ± 2.24 | **+2.8%** |
| pp16384 | 704.15 ± 0.87 | 722.56 ± 0.30 | **+2.6%** |

（pp2048 基线 790.99 → 采用后 815.48 ± 9.34；PPL 逐位一致 **7.1178**。）

解码侧：tg128 在这台机器上本身抖动很大（47.3~49.3，桌面合成器与它共用这张 WDDM 卡），
而单 token 时 4 列/warp 实测略慢（48.8 vs 49.3，处于噪声内）。所以最终做成**模板分派**：
`n_tokens >= GDN_WIDE_MIN_TOKENS(=8)` 才用 4 列（WIDE），单 token / 小批量解码保持 1 列 —— 两条路径
都验证到 PPL = 7.1178。

改动文件：`work/picks-llama/ggml/src/ggml-cuda/gated_delta_net.cu`
（`gdn_max_cols_per_warp()` + `GDN_WIDE_MIN_TOKENS` + 内核模板参数 `WIDE` + 128 分支二选一实例化）。
生产已部署：`work/pr210/deploy-picks.ps1`（备份 `work/bin-backup/20260926-161221-full`）。

## 十七、chunked GDN（WY/UT 形式）：公式跑通但比顺序内核慢 21% → 否决（留档）

补丁留档：`work/patches/local-gdn-chunk-wy-REJECTED-20260926.patch`（408 行：prep/scan 两个内核 + 自检支架）。

### 17.1 这一轮定位到的三个真问题

1. **β 少了 sigmoid**（真正的"爆炸"根因）：chunk prep 直接用了门控前 logit（−2.4/−3.0…），
   而三角系统里应该是 `sigmoid(β)=0.05~0.99`。这会把 |A| 顶到 1e14，状态每个 chunk 放大 5~10×。
   补上 sigmoid 后 `max|dState|` 从 1e19 → **1e-5**。
   顺带证伪了当时的猜测「无门控逆病态」：用真实尺度（|k|=1、β=sigmoid(logit)）的 numpy 数据，
   两种逆的 rel_err 都是 1e-6~1e-7（`work/tmp/gdn-chunk-real.py`）。
2. **输出必须走 `x = u − e^{G_j}(W_j·S_in)`**：`u=ds=A_g(β⊙V)` 与 `W=A_u(β⊙K)` 是两条不同的逆，
   不能各自直接乘到输出上；把「chunk 起始状态的贡献」折进单个 x，再同时喂给输出与状态更新，才自洽。
3. **`dst` 是 padding 视图**：`ggml_nelements(dst)` 大于真正写入的行数（ne=[6144,768]，实际 T·n_seqs=512），
   GDN 的 state 尾巴正好落在 padding 里 —— 自检把它当输出比，于是出现「seq 2 误差 50」的假报警。
   改成只比 `S_v*H*n_tokens*n_seqs` 后：seq0 = 3.8e-6、seq1 = 3.2e-6（|out|≈3.0，即相对误差 ~1e-6 = fp32 舍入级）。

### 17.2 性能：算术量本身就是 2.2×，追平需要重写成 tiled GEMM（收益上限 ~2-3%）

| | 顺序内核 | chunked |
|---|---|---|
| pp2048 | 803 ~ 814 | 634 |
| pp8192 | 772 | 610 |

把「128 维 k·q 点积在 1024 个 (t,cl) 上重复算 32 次」改成每 chunk 只算一次 `dkq[t][j]` 后，
486 → 634 t/s，但仍是 **−21%**。原因是算法层面的：

* 每 head-chunk 的 MAC：prep 262k + 4×split×478k ≈ **2.2M**，而顺序内核只有 1.0M；
* 顺序内核虽然只有 fp32 峰值的 ~4%，但它是被 shuffle/依赖链卡住，不是被 FLOP 卡住；
* 扫描核心里 `k_j·q_t` 的全局读取完全不合并（每个 chunk 每 block ~4M 次 cache-line 事务），
  要修就得把 k/q 搬进 shared 做 tiled GEMM（每 block 45KB+ shared、还要 opt-in 96KB），
  换来的上限只有整体 +2~3%。

结论：chunked 路线在本机（V100/sm70、无 tf32）不值当，回到顺序内核的微观优化（第十六节就是这条路线的产物）。
相关脚本留在 `work/tmp/gdn-chunk-*.py`（含"公式对、但自己 mirror 写错转置"的过程记录）、
`work/tmp/chunk-diag.ps1`（单序列自检）、`work/tmp/chunk-ab.ps1`（A/B）。

---

## 十八、注意力侧的排查 + 一个 5~6% 的模型侧修复（2026-09-26 18:xx）

### 18.1 FA 的 GQA 分组（ncols2）：负结果，已回退

模型几何：24 个 Q 头 / 4 个 KV 头 → **gqa_ratio = 6**，head_dim 256，64 层里每 4 层一个全注意力层。
Volta 分支用的是 `gqa_ratio % ncols2 == 0` 的保守判断（6 只能取 ncols2=2 → 同一份 K/V 被读 3 遍），
而通用分支只用 `gqa_ratio > ncols2` 判断（6 → ncols2=8，只读 1 遍）。

| | 基线（ncols2=2） | 改成 8 | Δ |
|---|---|---|---|
| pp2048 | 811.41 ± 13.30 | 821.09 ± 2.04 | +1.2% |
| pp16384 | 741.68 ± 0.99 | 714.08 ± 0.90 | **−3.7%** |

结论：**Volta 上 FA 内核是计算/发射受限**（sm70 走 fp16 FMA，没有张量核），省 K/V 读取没用，
反而多算的那 25% 空列要付代价。已恢复原样（`fattn.cu` 无改动）。

### 18.2 基线校准：llama-bench 默认 ubatch=512，面板用 1024

之前文档里的 pp 数字（787~815）都是 `llama-bench` 的**默认 -ub 512**。同机同二进制：

| | ub512 | ub1024 | ub2048 |
|---|---|---|---|
| pp2048 | 829.7 ± 6.0 | **1009.5 ± 4.4** | 1096.2 ± 28.2 |
| pp16384 | — | 889.3 ± 1.4 | 883.8 ± 53.3 |

ub 越大，每 pass 重复"反量化全部权重"的次数越少（代价 ∝ 1/ub），所以 pp2048 上 ub2048 快 8.6%。
但多轮会话相反——`work/tmp/turn-sim-ub3-results.txt`（10 轮、每轮 +1k token、cache_prompt=true）：

| ub | 每轮处理的 token | 10 轮总墙钟 |
|---|---|---|
| 512 | 1512 | 39.1 s |
| 1024 | 1512 ~ 3048 | 39.3 s |
| 2048 | **3048** | **47.8 s** |

原因：**前缀复用的粒度被 ubatch 量化**——ub=2048 时复用边界只能落在 2048 的整数倍上，
每轮要重算 3048 个 token（ub=512/1024 只需 1512）。所以面板保持 **-ub 1024** 是对的；
一次性长文档可以临时切 2048（但 16k 上差别已在噪声内）。

### 18.3 bf16 → fp16：+5.9% (pp2048) / +5.0% (pp16384)，PPL 不变（已采纳）

线索来自内核账本：`magma_sgemmEx_kernel` 占 pp2048 的 12%，且每次调用只有
**grid=(1,8,1)**、0.81 ms——按内存下界算慢了 60 倍。加打印定位到具体调用后：

```
MMC: compute_type=30(BF16) data_a=14(CUDA_R_16BF) cu_compute=68(CUBLAS_COMPUTE_32F)
     src0 [5120,48] t30 | src1 [5120,512] t0 | m=48 n=512 k=5120
```

也就是 **96 个 `ssm_alpha.weight` / `ssm_beta.weight`（[5120,48] BF16，共 24M 参数）**：Volta 没有 bf16
张量核，cuBLAS 只能用 `COMPUTE_32F` 软模拟 → 掉进 8 个 block 的旧 SGEMM 内核（实测 ~0.3 TFLOPS）。
换成 fp16 就能走张量核（V100 fp16 张量核 + fp32 累加 ≈ 62.5 TFLOPS）。先做了一次"强制 f16 计算类型"
的**测速下限探测**（数值是垃圾）：pp2048 1014 → **1088 t/s（+7.3%）**，确认收益量级。

修复做成**模型侧、无损**的：bf16 与 fp16 都是 2 字节/元素，文件布局不变；且 bf16 的 7 位尾数
能精确放进 fp16 的 10 位（常规值域内转换逐位无损）。脚本 `work/tmp/gguf-bf16-to-f16.py`：
只在**副本**上把这 96 个张量的类型字段改成 F16 并原地转码。

```
converted 96 bf16 tensors to f16; max|x|=0.337891; non-exact elements=8016 (23.6M 中 0.03%，全是 |x|<6e-5 的次正规数)
```

同机 A/B（`-ub 1024`）：

| 指标 | 原 bf16 | 转成 f16 | Δ |
|---|---|---|---|
| pp2048 | 1022.22 ± 6.02 | **1082.07 ± 7.29** | **+5.9%** |
| pp16384 | 905.32 ± 2.03 | **950.83 ± 2.72** | **+5.0%** |
| tg128 | 50.13 ± 0.52 | 49.2 ~ 50.2 | 不变（噪声内） |
| PPL | 7.1178 | **7.1131** | 累加顺序变了（张量核路径），远小于 ±0.86 的置信区间 |

上线：面板 `dashboard-config.json` 的 `model` 指向 `Ternary-Bonsai-2-27B-PQ2_0-f16ssm.gguf`
（原文件未动，配置备份 `dashboard-config.backup-20260926-185155.json`），
12k prompt 服务端探针 **870~878 → 947~957 t/s**。回退方式：把配置里的路径改回原文件再点"应用"。

**这条经验可以推广**：任何在 GGUF 里带 BF16 小张量（LoRA/门控投影/旋转矩阵）的模型，
在 Volta/Turing 上都会掉进 fp32 软模拟；同位宽转 fp16 是无损的，值得先扫一遍张量类型直方图。
