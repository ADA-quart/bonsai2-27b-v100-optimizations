**English** | [简体中文](README.zh-CN.md)

# Ternary Bonsai 2 27B on a 16 GB V100: from "collapses" to "fast enough"

A port-and-tuning log for running a 27B ternary-quantized model (1.75 bpw PTQ1_0, 131,072 context,
q8_0 KV cache, MTP speculative decoding) on a **Tesla V100-SXM2-16GB (sm_70, Windows/WDDM)** —
from "long prompts collapse" to "usable and close to the hardware limit". Patches, kernel sources,
a control panel and every measurement are included.

## Results (same machine, same model family; ctx 131072 + q8_0 KV)

| Metric | Before (upstream runtime) | After | Change |
|---|---|---|---|
| Prefill, 12K prompt | 175.8 t/s | **854.1 t/s** (12K probe, ub=1024 production setting; 848–873 on re-runs) | **4.9×** |
| Prefill, 55.5K prompt | 13.55 t/s | 681.1 t/s | **50×** |
| Prefill, 100K prompt | unusable | 610–740 t/s | — |
| Decode, 12K | 44.15 t/s | 45.75 → **53–60 t/s** (MTP on) | +26–38% (vs a 44.1 t/s baseline, workload-dependent) |
| Decode, 100K | — | 26–35 t/s | — |
| VRAM (ctx 131072) | 15,847 MiB (upstream runtime) | 13,979 MiB incl. MTP (three ub settings: 13,952–13,980) | −1.9 GiB |
| PPL (2-chunk corpus / q8_0, PQ2_0 → PTQ1_0) | 7.1131 | **7.1131** (bit-identical) | lossless |

> For a one-page English summary of everything below, see **[docs/BEFORE-AFTER.en.md](docs/BEFORE-AFTER.en.md)**.
> A separate 50K-corpus PPL of 4.6649 belongs to the Volta FA port and the q8_0-vs-q4_0 KV A/B
> (see `docs/RESEARCH.md` §17, Chinese) — it is not the PTQ1_0 equivalence evidence. The
> bit-identical evidence for PTQ1_0 is the 2-chunk corpus in §7.2.

The "before" numbers are not the model being slow: the FlashAttention kernels dequantized the whole
session's q8_0 KV cache into f16 (**~512 MiB at 131,072 context, redone for every op**), which blew
past the card and triggered WDDM paging.

## Contents

```
patches/   1) oripoin-picks-20260926-full.patch   upstream 842b18804 → 9b98d9dfd (oripoin picks + local adapt)
           2) local-patch-v100-20260927.patch    9b98d9dfd → the production tree (PTQ1_0 / sm70 D256 / regs / MTP)
           plus *-commits.txt; single-feature/ hold single-change excerpts (read-only, do not apply on top)
code/      new/rewritten kernels (D256 prefill FA, decode prototype, PTQ1_0 planar mat-vec) — same content as patch 2
           sm70-vendor/ two BSD-3 licence texts + fetch notes; the 7 flash/ headers ship with the repo
           (cute/cutlass still have to be fetched from upstream)
panel/     the control panel (live tok/s, first token, VRAM breakdown, GPU temp/power, one-click setup, …)
           bonsai-2-chat-template.jinja — the chat template we run in production (reasoning levels,
           thinking on/off, mid-conversation system messages, and the tool-loop guard)
tools/     probes and benchmarks (prefill-probe, agent-turn bench, phase profiling, spec-bench, deploy script)
docs/      research log: per-change A/Bs, rejected directions, performance ledger, before/after (Chinese)
           English versions: BEFORE-AFTER.en.md (one-pager) and TOOLS.en.md (script map)
THIRD-PARTY-NOTICES.md + THIRD-PARTY-LICENSES/   upstream & third-party licence list and full texts
```

## How to use

1. **Base**: upstream `llama.cpp` at `842b18804` (MIT).
2. Apply both patches, in this order (the order matters):

   ```bash
   git apply patches/oripoin-picks-20260926-full.patch
   git apply patches/local-patch-v100-20260927.patch
   ```

   Patch 2 already contains the three kernel sources **and their registration code** (the D256
   branch in `fattn.cu`, the PTQ1_0 dispatch in `mmvq.cu`, the include guard in `CMakeLists.txt`),
   so nothing has to be copied out of `code/` by hand.
3. **Vendored headers** (only needed for the D256 prefill kernel) — follow
   `code/sm70-vendor/README.md`:
   ① copy this repo's `code/sm70-vendor/flash/` to `ggml/src/ggml-cuda/sm70-vendor/flash/`
   (two of those files carry our port fixes, one is our stand-in for a file upstream generates in
   CMake); ② fetch `include/cute` and `include/cutlass` from CUTLASS@`62750a2b` into the same
   directory. If you do not want the kernel, delete `fattn-sm70-d256*.cu` (or set
   `LLAMA_SM70_D256=0` at runtime).
4. Build (our sm_70 configuration is described in `tools/` and `docs/RESEARCH.md` §8, Chinese).
5. Panel: `python panel/bonsai-dashboard.py --port 8090 --server-port 8080 --demo-dir <your model dir>`.
6. Experiments: the scripts in `tools/` (`prefill-probe.py`, `agent-turn-bench.py`,
   `decode-phase-prof.ps1`, …). Each `docs/` section names the script it used; the script ↔ doc
   map is in `docs/TOOLS.en.md` (English) / `docs/TOOLS.md` (Chinese).

## Notes

* **The model weights are not in this repository.** The `Ternary-Bonsai-2-27B` weights are licensed
  by their publisher (Apache-2.0) — get them from the original source. This repo only carries code
  and measurements.
* Numerics: PTQ1_0 rounds differently from the fp16/cuBLAS path, but the end-to-end **greedy output
  is byte-identical** (`docs/RESEARCH.md` §7.2, Chinese). The **sm70 D256 prefill path does diverge from
  stock on greedy output** (§20.3, expected; validate it with PPL / logit divergence), so
  re-check PPL-style metrics after switching paths.
* `patches/single-feature/*.patch` are excerpts of patch 1 (fastmtp d2t, graph shape cache) —
  reference only, do **not** apply them on top.
* The `work\...` paths are the author's internal working paths, kept for provenance; adjust the
  paths inside the scripts for your own tree.
* Experimental kernels that are off by default (`GGML_SM70_D256_DECODE`,
  `GGML_PTQ1_0_MULTI_CHUNK_MAX`, `SPC_DECODE_PROF`, …) are kept for the record only; the
  production path does not enable them.
* **Tool-loop guard**: `panel/bonsai-2-chat-template.jinja` counts consecutive *identical* tool
  calls (same tool, same arguments); at 3 it appends a warning to the **last tool response**, right
  before the next assistant turn, telling the model to stop repeating and use a different command
  (and listing the tools it is allowed to use). Sampling-level anti-repetition (DRY, even with
  `--dry-sequence-breaker none`) does **not** break those loops — the model still has to emit *some*
  tool call — while the template-level guard does: measured on a 3-repeat conversation, the model
  switched to a different tool instead of repeating, and the server kept parsing the call.
* **Language**: the detailed research notes under `docs/` are written in Chinese; this README and
  its result tables are in English, and the key conclusions match.

## License

New code and docs in this repository: **MIT** (see `LICENSE`). Upstream and third-party licences
are listed in `THIRD-PARTY-NOTICES.md` (a MIT / BSD-3-Clause / Apache-2.0 mix — everything in use
is permissively licensed).
