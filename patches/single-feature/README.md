# 单点摘录（仅参考）

这两个补丁是 `oripoin-picks-20260926-full.patch`（补丁 1）中对应改动的**摘录**，
为方便阅读单点改动而保留。

**不要**在打完补丁 1 之后再应用它们（内容已包含，`git apply` 会因已应用而失败）。

* `local-patch-fastmtp-d2t-20260926.patch` — d2t draft-vocab trim（源自 HauhauCS 的
  `HauhauCS-FastMTP-llama.cpp.patch`，Apache-2.0，见 `THIRD-PARTY-NOTICES.md`）
* `local-patch-graphshapecache-20260926.patch` — CUDA graph shape cache
