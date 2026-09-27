# -*- coding: utf-8 -*-
"""Fixed-size prefill probe. Reports the REAL prompt token count and prefill t/s.

Usage: python prefill-probe.py <label> <port> <n_paras> [max_tokens] [tail_text]

tail_text lets you control how predictable the continuation is:
  - default          -> "summarize the text above" (unpredictable, free-form)
  - "count"          -> "keep counting 1,2,3... one per line" (highly predictable)
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
N_PARA = int(sys.argv[3]) if len(sys.argv) > 3 else 207
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TAIL = sys.argv[5] if len(sys.argv) > 5 else ""

# ~58 tokens per paragraph with this tokenizer; 207 paras ~= 12K tokens
PARA = ("Bonsai 是一个本地推理实验平台，在十六 GB 显存的加速卡上运行二十七 B 的三值量化模型，"
        "使用四比特 KV 缓存与 FlashAttention 支撑长上下文。The harness records prompt and decode "
        "throughput for every request so regressions stay visible. ")

if TAIL == "count":
    instruction = ("\n\n忽略上面所有文字。现在只做一件事：从 1 开始逐行写整数，每行一个，"
                   "一直写到 40，不要任何解释、不要标点、不要空行。")
else:
    instruction = "\n\n用一句话总结上面这段文字。"

prompt = (PARA * N_PARA) + instruction
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOK,
        "temperature": 0, "cache_prompt": False}
req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                             data=json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json"})

t0 = time.time()
with urllib.request.urlopen(req, timeout=3600) as r:
    data = json.loads(r.read().decode("utf-8", "replace"))
wall = time.time() - t0

t = data.get("timings") or {}
u = data.get("usage") or {}
print("%-24s paras=%d  prompt_n=%-7s prefill=%8.1f t/s (%6.1f s)  decode=%6.1f t/s  out=%s  wall=%6.1f s"
      % (LABEL, N_PARA, t.get("prompt_n"), t.get("prompt_per_second") or 0,
         (t.get("prompt_ms") or 0) / 1000.0, t.get("predicted_per_second") or 0,
         u.get("completion_tokens"), wall))

import hashlib
ch  = (data.get("choices") or [{}])[0]
msg = ch.get("message", {}) or {}
txt = (msg.get("content") or "") + "\x00" + (msg.get("reasoning_content") or "")
print("   sha256=%s  head=%r" % (hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16], txt[:80]))
