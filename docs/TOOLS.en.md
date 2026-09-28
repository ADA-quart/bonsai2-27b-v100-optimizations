# Script map (tools/ ↔ doc section ↔ dependencies)

[简体中文](TOOLS.md) | **English**

> The scripts in `tools/` come from the author's working tree; paths use a `<REPO>`
> placeholder (the repo root on the author's machine). Replace `<REPO>` with your own
> directory before running them. Default port: 8080.

| Script | Purpose | Dependencies | Doc sections |
|---|---|---|---|
| `tools/prefill-probe.py` | Fixed-size prefill probe; reports the real prompt token count and prefill t/s (`python prefill-probe.py <label> <port> <paragraphs> [max_tok] [tail]`; 207 paragraphs ≈ 12K, 1720 ≈ 100K) | none (stdlib only) | RESULTS §1/§13, BEFORE-AFTER §1 |
| `tools/agent-turn-bench.py` | 10-turn benchmark reproducing the Codex agent loop shape: 8K preamble + ~600 tokens of tool output per turn, `cache_prompt=true` | none | RESEARCH §22.2/§24 |
| `tools/spec-bench.py` | Speculative-decoding A/B: same binary, same flags, drafter on vs off (math / CSV / summarisation prompts) | none | RESEARCH §13, RESULTS §2 |
| `tools/decode-phase-prof.ps1` | Runs with `SPC_DECODE_PROF=1` and prints the per-ubatch phase breakdown (apply / build / inputs / compute) | needs `work\tmp\prefill-turn-client.py` and `work\tmp\grep-lines.py` from the author's tree (**not shipped**; any chat client works instead — see the script comments) | RESEARCH §24.4 |
| `tools/deploy-picks.ps1` | Copies build outputs into `work\bonsai-demo\bin\cuda\` with SHA256 verification and a timestamped backup | the author's working-tree layout | deployment notes |

## Internal scripts that are *not* shipped (referenced by the docs)

These are tightly coupled to the author's machine/working tree (internal paths, one-off
experiments); re-create them from the documentation if you need them:

`work\mtp\prefill-probe.py` (origin of the `tools/` version), `work\tmp\prefill-turn-client.py`,
`work\tmp\grep-lines.py`, `work\tmp\kern-census2.py`, `work\tmp\turn-kernel-shapes.py`,
`work\tmp\agent-turn-sweep.ps1`, the scripts that call `work\pr210\spec-bench.py`, and
`work\mtp\run-pick.ps1` / `stop-test.ps1` (local service start/stop).

## What the panel already covers

The panel (`panel/bonsai-dashboard.py`) has built-in prefill/decode speed, GPU status, a VRAM
breakdown and a clean-benchmark entry point — prefer it for day-to-day measurement. The scripts
exist mainly for the reproducible A/Bs described in `docs/`.
