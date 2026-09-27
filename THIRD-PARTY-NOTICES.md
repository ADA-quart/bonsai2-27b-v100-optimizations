# Third-party notices

All upstream and third-party components used by this repository are under permissive licenses
(MIT / BSD-3-Clause / Apache-2.0). Full license texts: `THIRD-PARTY-LICENSES/` and
`code/sm70-vendor/` (the two BSD-3 texts that must travel with the vendored kernel headers).
The model weights are **not** part of this repository.

| 组件 | 用途 | 许可证 | 出处 |
|---|---|---|---|
| llama.cpp | 底座（推理引擎） | MIT | https://github.com/ggml-org/llama.cpp |
| oripoin/llama.cpp | q8_0 KV 直读的 flash-attention tile 内核等 9 个提交 | MIT | https://git.oripoin.me/oripoin/llama.cpp |
| fishlikeX/sm70-attn | D256 Split-D prefill FA 内核（我们移植并改写） | MIT | https://github.com/fishlikeX/sm70-attn |
| WyvernTKC/llama.cpp-4xV100 | Volta hs=256 FA 配置重调与 `__syncthreads` UB 修复 | MIT | https://github.com/WyvernTKC/llama.cpp-4xV100 |
| 1CatAI/1Cat-vLLM v1.3.0 | D256 Split-D FA 内核原型（`fattn-sm70-d256.cu` 由其 patch 改写；`fattn-sm70-d256-kernel.cuh` 头部保留其版权与许可声明） | Apache-2.0 | https://github.com/1CatAI/1Cat-vLLM |
| Cary Palmer / bonsai-ada-surgery（professorpalmer） | PTQ1_0 planar mat-vec 栈（补丁 2 的 bea69867f）、GDN gather 融合（4d0a06e3c）、MTP catch-up deferral / graph output rows / draft-depth-max / batch-invariant flag（d5b463e13），即原 patch 系列 0001–0023 | MIT | https://github.com/professorpalmer/bonsai-ada-surgery |
| HauhauCS（HF: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`） | `patches/local-patch-fastmtp-d2t-20260926.patch` 与 `patches/oripoin-picks-20260926-full.patch` 中的 d2t draft-vocab trim 片段（`HauhauCS-FastMTP-llama.cpp.patch`，sha256 `98128540…d615`） | Apache-2.0 | https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF |
| NVIDIA CUTLASS / CuTe（vendored，152 个头文件，未随仓库分发） | sm70 kernel 的 CuTe MMA 依赖 | BSD-3-Clause | https://github.com/NVIDIA/cutlass，commit `62750a2b75c802660e4894434dc55e839f322277` |
| zhinianqin/flash-attention-v100（vendored，8 个头文件，未随仓库分发） | sm70 kernel 的 attention 头文件 | BSD-3-Clause | https://github.com/zhinianqin/flash-attention-v100，commit `c2eda5e6115b98c3ba4bfd181570668742eece22` |
| PrismML（Ternary Bonsai 2 27B 权重 / 打包格式） | 模型本身（**未随仓库分发**） | 见其模型发布页 | 模型发布页 |

## 分发注意

* 如果你重新分发 `code/ggml-cuda/fattn-sm70-d256.cu` / `-kernel.cuh` / `-decode.cu`，请一并保留
  CUTLASS/CuTe 与 flash-attention-v100 的 LICENSE 文件（BSD-3 要求保留版权与许可声明），
  见 `code/sm70-vendor/`。
* `patches/` 里的补丁包含 llama.cpp/oripoin（MIT）、1Cat-vLLM（Apache-2.0）与 HauhauCS
  （Apache-2.0）的原始代码片段；再分发时请保留 `THIRD-PARTY-LICENSES/` 里的许可文本与上表署名。
