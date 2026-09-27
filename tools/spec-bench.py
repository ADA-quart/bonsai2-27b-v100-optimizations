# -*- coding: utf-8 -*-
"""Speculative-decoding benchmark: same binary, same flags, with and without the drafter.

Usage: python spec-bench.py <port> <label>
"""
import json
import sys
import time
import urllib.request

PROMPTS = [
    ("math",   "Multiply 1234 by 5678 step by step and give the final result on the last line."),
    ("code-csv", "Write a Python function that parses one CSV line with quoted fields, "
                 "handling embedded commas and escaped double quotes. Include a short docstring."),
    ("code-bs", "Implement binary search in Python that returns the leftmost index of the target "
                "value, or -1 when it is absent."),
    ("chat",   "Explain what speculative decoding is in three sentences."),
]


def ask(port, prompt, max_tokens=300, timeout=900):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data, time.time() - t0


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    label = sys.argv[2] if len(sys.argv) > 2 else "run"
    print("== %s (port %d) ==" % (label, port))
    print("%-9s %8s %10s %10s %10s %8s" % ("task", "out_tok", "decode t/s", "draft_n", "accepted", "accept%"))
    rows = []
    # one warm-up
    try:
        ask(port, "Say hi.", max_tokens=8)
    except Exception as exc:
        print("warm-up failed:", exc)
    for tag, prompt in PROMPTS:
        data, wall = ask(port, prompt)
        t = data.get("timings") or {}
        u = data.get("usage") or {}
        dn = t.get("draft_n") or 0
        da = t.get("draft_n_accepted") or 0
        acc = (100.0 * da / dn) if dn else 0.0
        print("%-9s %8s %10.1f %10s %10s %7.1f%%"
              % (tag, u.get("completion_tokens"), t.get("predicted_per_second") or 0,
                 dn or "-", da or "-", acc))
        rows.append({"task": tag, "tok": u.get("completion_tokens"),
                     "tps": t.get("predicted_per_second"), "draft_n": dn,
                     "draft_accepted": da, "wall": round(wall, 2),
                     "pp": t.get("prompt_per_second")})
    with open(r"<REPO>\work\pr210\bench-%s.json" % label,
              "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print("saved bench-%s.json" % label)


if __name__ == "__main__":
    main()
