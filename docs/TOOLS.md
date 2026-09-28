# 脚本对照表（tools/ ↔ 文档章节 ↔ 依赖）

[English](TOOLS.en.md) | **简体中文**

> `tools/` 里的脚本来自作者的工作树，路径全部用 `<REPO>` 占位（本机为仓库根目录）。
> 跑之前先把脚本里的 `<REPO>` 换成你的目录；端口默认 8080。

| 脚本 | 用途 | 依赖 | 对应文档 |
|---|---|---|---|
| `tools/prefill-probe.py` | 固定长度预填充探针，报告真实 prompt token 数与 prefill t/s（`python prefill-probe.py <label> <port> <段数> [max_tok] [tail]`；207 段≈12K、1720 段≈100K） | 无（仅标准库） | RESULTS §一/§13、BEFORE-AFTER §1 |
| `tools/agent-turn-bench.py` | 复刻 Codex agent 轮次形态的 10 轮基准：8K 前言 + 每轮 ~600 token 工具输出、`cache_prompt=true` | 无 | RESEARCH §22.2/§24 |
| `tools/spec-bench.py` | 投机解码 A/B：同一二进制、同一参数，开/关草稿对比（含 math/csv/总结三类提示） | 无 | RESEARCH §13、RESULTS §二 |
| `tools/decode-phase-prof.ps1` | 用 `SPC_DECODE_PROF=1` 抓每 ubatch 的阶段拆解（apply/build/inputs/compute） | 需要作者工作树的 `work\tmp\prefill-turn-client.py` 与 `work\tmp\grep-lines.py`（**未随包**，可自行用任意 chat 客户端替代，见脚本注释） | RESEARCH §24.4 |
| `tools/deploy-picks.ps1` | 把构建产物部署到 `work\bonsai-demo\bin\cuda\`，带 SHA256 校验与时间戳备份 | 作者工作树目录结构 | 部署记录 |

## 未随包的内部脚本（文档里提到但不在 tools/）

这些脚本与作者的机器/工作树强绑定（内部路径、专有实验），需要时按文档描述重写：

`work\mtp\prefill-probe.py`（tools 版本的来源）、`work\tmp\prefill-turn-client.py`、
`work\tmp\grep-lines.py`、`work\tmp\kern-census2.py`、`work\tmp\turn-kernel-shapes.py`、
`work\tmp\agent-turn-sweep.ps1`、`work\pr210\spec-bench.py` 的原始调用脚本、
`work\mtp\run-pick.ps1` / `stop-test.ps1`（本机服务启停）。

## 由面板承担的部分

面板（`panel/bonsai-dashboard.py`）内置了预填充/解码速度、显卡状态、显存拆分、干净基准入口，
日常测量优先用它；脚本主要用于文档里那些 A/B 的可复现实验。
