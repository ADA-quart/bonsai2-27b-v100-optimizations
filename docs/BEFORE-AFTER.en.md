# Bonsai 2 27B @ V100 16 GB: before vs now (one pager)

[简体中文](BEFORE-AFTER.md) | **English**

Sources: `RESULTS-ORIPOIN-PICKS-20260926.md` §1 ("old runtime" = the upstream bonsai-demo runtime as
measured), the per-change A/B tables, and re-runs made on 2026-09-27 in the production
configuration (`work\tmp\agent-turn-bench.py`, `prefill-probe.py`, `test-backend-ops perf`).

## 1. Hard numbers (same machine, same model family; ctx 131072 + q8_0 KV)

| Metric | At the start (upstream runtime) | Now | Change |
|---|---|---|---|
| Prefill, 12K prompt | **175.8 t/s** | **854.1 t/s** (12K probe, ub=1024 production setting; 848–873 on re-runs) | **4.9×** |
| Prefill, 55.5K prompt | **13.55 t/s** | 681.1 t/s (same-context probe) | **50×** |
| Prefill, 100K prompt | (collapses, unusable) | 610–740 t/s | — |
| Decode, 12K (4-task mean, no MTP) | 44.15 t/s | 45.75 → 53–60 t/s (with MTP) | +26–38% (vs a 44.1 t/s baseline, workload-dependent) |
| Decode, 100K | — | 26–35 t/s (low/high acceptance) | — |
| VRAM (ctx 131072) | 15,847 MiB (upstream runtime; same-context prefill drops to 175.8 t/s) | 13,952–13,980 MiB (q8_0 KV + MTP, three ub settings measured; 14,395 in the probe run) | ≈ −1.9 GiB |
| Context | 131072 (but unusable) | 131072 usable; a 262144 preset is also included | — |
| PPL (2-chunk corpus / q8_0, PQ2_0 → PTQ1_0) | 7.1131 | **7.1131** (bit-identical) | lossless |

> The separate 50K-corpus PPL pair (4.6649 ↔ 4.6649) is the Volta FA port (RESEARCH §17) and the
> q8_0-vs-q4_0 KV A/B — it is not the PTQ1_0 equivalence evidence.

Those original numbers ("175.8 t/s, 13.55 t/s at 55K") are not the model being slow: the **FA
kernels dequantized the whole session's q8_0 KV cache into f16** (~512 MiB at ctx 131072, redone
for every op) → VRAM blow-out → WDDM paging.

## 2. Per-change contributions (each backed by A/B data)

| Change | Contribution |
|---|---|
| ① tile/mma FA reads q8_0 KV in place (oripoin `6ae7cbeb8` and 8 more commits) | prefill **5.2×** (12K), 50× (55K), VRAM −1.9 GiB |
| ② Full PTQ1_0 kernel-stack port (planar activation layout + dedicated mat-vec) | pp16384 +3.4%, tg128 +6.4%, weights −1.17 GiB, PPL bit-identical |
| ③ sm70 D256 Split-D prefill FA (our own port) | pp16384 +8%, pp32768 +13.8%; **1.85×** stock at the same shape |
| ④ MTP draft head grafted into the main GGUF + draft micro-batch fix | decode **+26–38%**; 12K prefill with MTP 522 → 802 t/s |
| ⑤ Volta hs=256 FA config retune + np>1 barrier UB (upstream #27955) | removed 8,576 UBs; PPL bit-identical |
| ⑥ GDN column-parallel `cols_per_warp` 1→4 | prefill +2.6–2.8% across all sizes |
| ⑦ ubatch by context (32K/64K → 2048) | short-context prefill +7.4%, decode +12–25% |

## 3. Usability (started as hand-clicking in LM Studio, now)

* A self-built panel (port 8090): live tok/s, first-token latency, prefill/decode split, VRAM
  breakdown, GPU utilisation / temperature / power, context and cache-hit display; one-click
  start/stop, one-click presets, visual parameter editing.
* "One-click connect to CC Switch / Codex": refreshes the provider entry and `config.toml`
  (including the four reasoning levels) and self-checks the result.
* Desktop shortcuts, unload-the-model-on-exit, one-click service unload.
* Presets calibrated by measurement: 32K/64K (ub 2048, MTP), Agent 128K (calibrated to the measured
  shape of a multi-turn agent workload) and a 256K extreme preset.

## 4. Honest conclusion

* **Prefill** went from "unusable" to "good enough and close to the ceiling": what remains is
  small-batch GPU idle time (~55% of the wall clock) and GEMMs already running at 65% of DRAM —
  further gains are in the 1% range.
* **Decode** gains come mostly from MTP (+26–38%) and PTQ1_0 (+6%); it is bandwidth-bound
  (5.95 GiB of weights × tokens/s) and now sits in the V100's practical range.
* A set of directions was also disproved along the way (chunked GDN, KVMem, PTQ1_0 MMQ, KV
  mean-centring, a decode-specific kernel, fused small-m GEMMs, …) — all kept in the RESEARCH log
  so nobody retries them blindly.
