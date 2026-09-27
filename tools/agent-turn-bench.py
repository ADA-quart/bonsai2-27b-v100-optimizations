# Reproduce the user's real Codex-agent loop against the local server and report per-turn cost.
# Shape taken from their own rollouts: ~8K preamble, then 10 turns that each append ~600 tokens
# of tool output + a short question, with cache_prompt=true (full prefix reuse, like Codex does).
import json
import statistics as st
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PORT = int(sys.argv[1])
LABEL = sys.argv[2]

PARA = ("Bonsai 是一个本地推理实验平台，在十六 GB 显存的加速卡上运行二十七 B 的三值量化模型，"
        "使用四比特 KV 缓存与 FlashAttention 支撑长上下文。The harness records prompt and decode "
        "throughput for every request so regressions stay visible. ")
TOOL = ("$ python tests/run_case.py --case %d\n"
        "== case %d ==\n"
        "loading model ... ok\n"
        "prefill 12067 tokens in 14.2 s (849.9 t/s), decode 32 tokens at 43.7 t/s\n"
        "peak vram 14108 MiB, temperature 41 C\n")

def ask(messages, max_tokens=300):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data, time.time() - t0

msgs = [{"role": "user", "content": PARA * 140 + "\n\n请用一句话说明上面这段在讲什么。"}]
turns = []
try:
    ask(msgs, 8)   # warm-up
except Exception as exc:
    print("warm-up failed:", exc)

for turn in range(1, 11):
    msgs.append({"role": "assistant", "content": "上面这段讲的是本地推理平台的实测记录。"})
    msgs.append({"role": "user", "content": TOOL % (turn, turn) + "\n这一轮的 prefill 和 decode 分别是多少？"})
    data, wall = ask(msgs, 300)
    t = data.get("timings") or {}
    turns.append({"wall": wall, "pp": t.get("prompt_ms", 0), "pp_n": t.get("prompt_n", 0),
                  "tg": t.get("predicted_per_second", 0), "cached": t.get("cache_n", 0),
                  "out": (data.get("usage") or {}).get("completion_tokens", 0)})
    print("  turn %2d: wall %5.2f s | prefill %6.0f ms / %5d tok (cached %5d) | decode %5.1f t/s | out %d"
          % (turn, turns[-1]["wall"], turns[-1]["pp"], turns[-1]["pp_n"], turns[-1]["cached"],
             turns[-1]["tg"], turns[-1]["out"]))

tot = sum(t["wall"] for t in turns)
print("%-16s 10 turns: total %.1f s | mean %.2f s/turn | prefill mean %.0f ms | decode mean %.1f t/s"
      % (LABEL, tot, tot / 10, st.mean(t["pp"] for t in turns), st.mean(t["tg"] for t in turns)))
