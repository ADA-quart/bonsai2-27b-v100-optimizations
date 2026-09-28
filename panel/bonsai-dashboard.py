#!/usr/bin/env python3
"""Bonsai 本地推理控制台

功能：
  * 流式对话 + 实时 token/s 曲线、首 token 延迟、预填充速度
  * 模型加载参数面板（上下文、GPU 层数、KV 缓存类型、FlashAttention、并发、采样参数…）并可一键重载
  * 显卡状态：利用率、温度、功耗、显存占用；FlashAttention / KV 缓存是否量化一目了然

只用 Python 标准库。上游是 llama-server 的 OpenAI 兼容接口。
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _cli_demo_dir():
    """--demo-dir / BONSAI_DEMO_DIR, resolved before the derived paths below."""
    for i, arg in enumerate(sys.argv):
        if arg == "--demo-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--demo-dir="):
            return arg.split("=", 1)[1]
    return os.environ.get("BONSAI_DEMO_DIR") or r"<REPO>\work\bonsai-demo"


DEMO_DIR = _cli_demo_dir()

SERVER_EXE = os.path.join(DEMO_DIR, "bin", "cuda", "llama-server.exe")
CONFIG_PATH = os.path.join(DEMO_DIR, "dashboard-config.json")
LOG_OUT = os.path.join(DEMO_DIR, "server.out.log")
LOG_ERR = os.path.join(DEMO_DIR, "server.err.log")
MODELS_DIR = os.path.join(DEMO_DIR, "models")
SERVER_PORT = 8080
UPSTREAM = f"http://127.0.0.1:{SERVER_PORT}"

MODEL_REL = r"models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PQ2_0.gguf"
MMPROJ_REL = r"models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"

DEFAULT_CONFIG = {
    "model": MODEL_REL,
    "mmproj": MMPROJ_REL,
    "use_mmproj": True,
    "ctx": 32768,
    "ngl": 99,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "flash_attn": "on",
    "parallel": "auto",
    "threads": "",
    "batch": 2048,
    "ubatch": 512,
    "temp": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    # DRY sampler (repeat suppression): 0 = off. It fights *text* loops such as
    # "but wait / let me do"; tool-call loops are handled by the anti-loop rules in
    # the model instructions (see bonsai-reasoning-protocol.md, rule 8).
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 64,
    # DRY sequence breakers: the default ('\n', ':', '"', '*') resets the penalty at every
    # newline, so a repeated multi-line tool call is not penalised; "none" removes that reset
    # (stronger; it also penalises legitimate repetition inside code).
    "dry_sequence_breaker": "",
    "chat_template_file": "",
    "reasoning_budget": "",
    "compact_tokens": 100000,
    "kv_offload": True,
    "thinking": "on",
    "mmproj_cpu": False,
    "use_kv_bias": True,
    # MTP 投机解码：草稿头已经嫁接进主 GGUF（blk.64.*，ProCreations on-policy Q8），
    # 跑在目标模型自己的权重上，只多一个草稿上下文（+1.0 GB 显存，实测 12867 -> 13906 MiB）。
    # 实测（131072 ctx / q8_0 KV / -ub 1024，V100）：
    #   解码：短提示 44.1 -> 55~61 t/s；12K 深度 39.6 -> 50.3（+27%）；
    #        24K 深度 38.9 -> 50.9（+31%）；48K 深度 33.3 -> 40.5（+22%）
    #   冷启动 prefill：+0.85 ms/token（12K 提示 12.5 s -> 22.6 s），前缀缓存不受影响
    #   所以多轮 agent 会话净赚、冷启动超长 prompt 净亏；不想付这个代价就把勾去掉
    # depth-max：本卡（V100）实测到 48K 深度草稿都还划算，所以默认 0 = 不停止。
    # 想沿用 ada-surgery 在 4070 上的做法（24K 后停草稿）可以填 24576。
    "spec_type": "draft-mtp",
    "spec_draft_model": "",
    "spec_draft_n_max": 2,
    "spec_draft_depth_max": 0,
}

CACHE_FACTOR = {"f16": 1.0, "bf16": 1.0, "q8_0": 0.53, "q5_0": 0.35, "q4_0": 0.28}
_status_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()
_busy = {"reloading": False, "message": ""}

# 实时监测（Codex / CC Switch 或面板自身的请求都会经过 llama-server，这里统一采样）。
LIVE = {
    "active": False, "phase": "idle", "source": None, "started": None,
    "ttft": None, "elapsed": None,
    "input_tokens": 0, "cache_tokens": 0, "processed_tokens": 0, "output_tokens": 0,
    "prefill_tps": None, "decode_tps": None,
    "series": [], "last": None, "recent": [],
    "totals": {"requests": 0, "in": 0, "out": 0, "seconds": 0.0, "since": time.time()},
    "updated": 0.0,
}
PANEL_INFLIGHT = 0
_live_lock = threading.Lock()


# ---------------------------------------------------------------- 基础工具
def fetch_json(url, timeout=4):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)


def log_tail(lines=12):
    out = []
    for path in (LOG_ERR, LOG_OUT):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                out.extend(fh.readlines()[-lines:])
        except Exception:
            pass
    return [ln.rstrip() for ln in out if ln.strip()][-lines:]


def gpu_info():
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        name, util, used, total, temp, power, plimit = [p.strip() for p in proc.stdout.splitlines()[0].split(",")]
        return {
            "name": name,
            "util": int(float(util)),
            "mem_used_mb": int(float(used)),
            "mem_total_mb": int(float(total)),
            "temp_c": int(float(temp)),
            "power_w": float(power) if power not in ("[N/A]", "N/A") else None,
            "power_limit_w": float(plimit) if plimit not in ("[N/A]", "N/A") else None,
        }
    except Exception:
        return None


def pids_on_port(port=SERVER_PORT):
    pids = set()
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    for line in out.splitlines():
        if "LISTENING" not in line.upper():
            continue
        if re.search(rf":{port}\s", line):
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.add(int(parts[-1]))
    return sorted(pids)


def is_online(timeout=2):
    try:
        return fetch_json(f"{UPSTREAM}/health", timeout=timeout).get("status") == "ok"
    except Exception:
        return False


def server_runtime_info():
    info = {"online": False, "model_path": None, "ftype": None, "ctx": None,
            "slots": None, "vision": None, "sampling": {}}
    if not is_online():
        return info
    info["online"] = True
    try:
        props = fetch_json(f"{UPSTREAM}/props", timeout=5)
        info["model_path"] = props.get("model_path")
        info["ftype"] = props.get("model_ftype")
        info["slots"] = props.get("total_slots")
        info["vision"] = (props.get("modalities") or {}).get("vision")
        dgs = props.get("default_generation_settings") or {}
        info["ctx"] = dgs.get("n_ctx")
        params = dgs.get("params") or {}
        info["sampling"] = {k: params.get(k) for k in ("temperature", "top_p", "top_k", "min_p")}
    except Exception:
        pass
    return info


def derived_info(cfg):
    factor = CACHE_FACTOR.get(str(cfg.get("cache_type_k", "f16")), 1.0)
    kv_gib = (64 * 1024 * int(cfg.get("ctx") or 0) * factor) / (1024 ** 3)
    k = str(cfg.get("cache_type_k", "f16"))
    v = str(cfg.get("cache_type_v", "f16"))
    quantized = not (k.startswith("f") and v.startswith("f"))
    bias = find_kv_bias(cfg.get("model"))
    bias_active = bool(bias) and k == "q4_0" and bool(cfg.get("use_kv_bias", True))
    ngl = int(cfg.get("ngl") or 0)
    raw_par = str(cfg.get("parallel", "auto")).strip().lower()
    auto_par = raw_par in ("", "auto", "0", "none")
    try:
        parallel = 1 if auto_par else max(1, int(raw_par))
    except Exception:
        auto_par, parallel = True, 1
    ctx = int(cfg.get("ctx") or 0)
    return {
        "kv_gib": round(kv_gib, 2),
        "kv_quantized": quantized,
        "kv_label": "纯净 FP16（未量化）" if not quantized else f"量化 {k}/{v}",
        "kv_location": (("显存（--kv-offload，解码更快）" if cfg.get("kv_offload", True)
                         else "内存（--no-kv-offload，会变慢）") if ngl >= 99
                        else f"部分在内存（仅 {ngl} 层上 GPU）"),
        "kv_offload": bool(cfg.get("kv_offload", True)),
        "kv_bias_file": bias["name"] if bias else None,
        "kv_bias_rot": (bias or {}).get("k_rot"),
        "kv_bias_active": bias_active,
        "flash": {"on": "开启（官方推荐）", "off": "关闭", "auto": "自动"}
                 .get(str(cfg.get("flash_attn")), "自动"),
        "flash_required_for_kv": quantized and str(cfg.get("flash_attn")) == "off",
        "ctx_per_slot": ctx if auto_par else (ctx // parallel if parallel else ctx),
        "parallel": parallel,
        "parallel_auto": auto_par,
    }


def build_args(cfg):
    model = cfg["model"]
    if not os.path.isabs(model):
        model = os.path.join(DEMO_DIR, model)
    args = [SERVER_EXE, "-m", model,
            "-ngl", str(cfg.get("ngl", 99)),
            "-c", str(cfg.get("ctx", 32768)),
            "--cache-type-k", str(cfg.get("cache_type_k", "f16")),
            "--cache-type-v", str(cfg.get("cache_type_v", "f16")),
            "--flash-attn", str(cfg.get("flash_attn", "on")),
            "-b", str(cfg.get("batch", 2048)),
            "-ub", str(cfg.get("ubatch", 512)),
            "--host", "127.0.0.1", "--port", str(SERVER_PORT), "--jinja",
            "--temp", str(cfg.get("temp", 0.5)),
            "--top-p", str(cfg.get("top_p", 0.85)),
            "--top-k", str(cfg.get("top_k", 20)),
            "--min-p", str(cfg.get("min_p", 0)),
            "--presence-penalty", str(cfg.get("presence_penalty", 0.0))]
    par = str(cfg.get("parallel", "auto")).strip().lower()
    if par not in ("", "auto", "0", "none"):
        args += ["-np", str(cfg.get("parallel"))]
    threads = cfg.get("threads")
    if str(threads).strip():
        args += ["-t", str(threads)]
    budget = cfg.get("reasoning_budget")
    if str(budget).strip():
        args += ["--reasoning-budget", str(budget)]
    thinking = str(cfg.get("thinking", "on")).lower()
    # 官方用法：--reasoning on/off/auto + --reasoning-effort LEVEL
    # （--chat-template-kwargs 传 enable_thinking 已被 fork 标记为 deprecated，
    #   见 common/arg.cpp:3660 / 3678；模型卡支持的努力档位是 xhigh(默认)/medium/low）
    #
    # 这里必须用 "auto" 而不是 "on"：--reasoning on 会把 enable_thinking="true" 写进
    # default_template_kwargs，而默认 kwargs 在 server-common.cpp:1296-1312 会覆盖请求里的
    # reasoning_effort=none（"none" 本应把 enable_thinking 置 false），结果客户端就再也
    # 关不掉思考。"auto" 同样默认开启思考，但不写这个默认 kwarg，
    # 于是 Codex / OpenAI 客户端发的每请求档位（none/low/medium/xhigh）才能生效。
    if thinking == "off":
        args += ["--reasoning", "off"]
    elif thinking == "medium":
        args += ["--reasoning", "auto", "--reasoning-effort", "medium"]
    else:
        args += ["--reasoning", "auto", "--reasoning-effort", "xhigh"]
    args += ["--kv-offload"] if cfg.get("kv_offload", True) else ["--no-kv-offload"]
    if str(cfg.get("cache_type_k", "f16")) == "q4_0" and cfg.get("use_kv_bias", True):
        bias = find_kv_bias(cfg.get("model"))
        if bias:
            args += ["--kv-mean-center", bias["path"]]
    if cfg.get("use_mmproj") and cfg.get("mmproj"):
        mm = cfg["mmproj"]
        if not os.path.isabs(mm):
            mm = os.path.join(DEMO_DIR, mm)
        if os.path.exists(mm):
            args += ["--mmproj", mm]
            if cfg.get("mmproj_cpu"):
                # 视觉塔放内存，省约 0.9 GiB 显存（只影响图片预填充速度）
                args += ["--no-mmproj-offload"]
    # 兼容补丁版对话模板：把客户端发的 high/ultra/minimal 等思考档映射到模型支持的
    # xhigh/medium/low，并容忍会话中途出现的 system/developer 消息（Codex 会这么发）
    try:
        dry = float(cfg.get("dry_multiplier") or 0)
    except Exception:
        dry = 0.0
    if dry > 0:
        try:
            dry_len = max(1, min(16, int(float(cfg.get("dry_allowed_length") or 2))))
        except Exception:
            dry_len = 2
        try:
            dry_last_n = int(float(cfg.get("dry_penalty_last_n") or 64))
        except Exception:
            dry_last_n = 64
        args += ["--dry-multiplier", f"{dry:g}",
                 "--dry-base", str(cfg.get("dry_base") or 1.75),
                 "--dry-allowed-length", str(dry_len),
                 "--dry-penalty-last-n", str(dry_last_n)]
        brk = str(cfg.get("dry_sequence_breaker") or "").strip()
        if brk:
            args += ["--dry-sequence-breaker", brk]
    tpl = str(cfg.get("chat_template_file") or "").strip()
    if tpl:
        if not os.path.isabs(tpl):
            tpl = os.path.join(DEMO_DIR, tpl)
        if os.path.exists(tpl):
            args += ["--chat-template-file", tpl]
    # MTP 投机解码。
    # draft-mtp：草稿头就在主 GGUF 里（blk.64.*），跑在目标模型自己的权重上，
    #   只多一个草稿上下文（实测 +1.0 GB）。**绝不能传 --spec-draft-model**：
    #   那会让运行时把整份 5.8 GB 权重再加载一遍，131072 ctx 下直接爆显存
    #   （实测此时 prompt 处理从 1015 t/s 掉到 3.8 t/s）。
    # 其它 spec 类型（draft-dspark / draft-eagle3 / …）才需要独立的草稿模型文件。
    spec = str(cfg.get("spec_type") or "").strip()
    if spec and spec.lower() not in ("none", "off", "0"):
        try:
            n_max = int(float(cfg.get("spec_draft_n_max") or 2))
        except Exception:
            n_max = 2
        n_max = min(max(n_max, 1), 8)
        if spec == "draft-mtp":
            args += ["--spec-type", spec,
                     "--spec-draft-n-max", str(n_max),
                     "--spec-draft-type-k", "q8_0",
                     "--spec-draft-type-v", "q8_0"]
            try:
                depth_max = int(float(cfg.get("spec_draft_depth_max") or 0))
            except Exception:
                depth_max = 24576
            if depth_max > 0:
                args += ["--spec-draft-depth-max", str(depth_max)]
        else:
            draft = str(cfg.get("spec_draft_model") or "").strip()
            if draft and not os.path.isabs(draft):
                draft = os.path.join(DEMO_DIR, draft)
            if draft and os.path.exists(draft):
                args += ["--spec-type", spec,
                         "--spec-draft-model", draft,
                         "--spec-draft-n-max", str(n_max),
                         "--spec-draft-ngl", str(cfg.get("ngl", 99)),
                         "--spec-draft-type-k", "q8_0",
                         "--spec-draft-type-v", "q8_0"]
    return args


def stop_server():
    """停掉端口上的所有 llama-server。

    Windows 的 SO_REUSEADDR 允许两个进程同时监听同一端口（旧实例不退出时，
    新实例照样 bind 成功，请求会落到旧实例上），所以这里必须反复确认端口真的空了。
    """
    killed = []
    for _attempt in range(4):
        pids = pids_on_port()
        if not pids:
            break
        for pid in pids:
            if _kill_pid(pid):
                killed.append(pid)
        time.sleep(2)
    remaining = pids_on_port()
    if remaining:
        raise RuntimeError(f"端口 {SERVER_PORT} 仍被进程占用：{remaining}（可能没有权限结束它）")
    if killed:
        time.sleep(1)
    return killed


def _kill_pid(pid):
    """优先 taskkill；某些沙箱会拦 taskkill.exe，退回 PowerShell 的 Stop-Process。"""
    cmds = (
        ["taskkill", "/PID", str(pid), "/F", "/T"],
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
    )
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        if proc.returncode == 0 and not _pid_alive(pid):
            return True
    return not _pid_alive(pid)


def _pid_alive(pid):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'alive' }}"],
                             capture_output=True, text=True, timeout=20).stdout
        return "alive" in out
    except Exception:
        return False


def start_server(cfg, wait_seconds=300):
    busy = pids_on_port()
    if busy:
        return False, f"端口 {SERVER_PORT} 仍被进程占用：{busy}；先「停止服务」再启动"
    args = build_args(cfg)
    creation = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    env = os.environ.copy()
    # 开了投机解码时顺手打开图形状缓存：draft 上下文的图会在 1 行和 1+n_max 行之间反复重捕获，
    # 打开实测 +3.8%（见 work/mtp/graph-trace-findings.md）。
    if str(cfg.get("spec_type") or "").strip().lower() not in ("", "none", "off", "0"):
        env["GGML_CUDA_GRAPH_SHAPECACHE"] = "1"
    with open(LOG_OUT, "a", encoding="utf-8") as out, open(LOG_ERR, "a", encoding="utf-8") as err:
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====\n{' '.join(args)}\n")
        out.flush()
        subprocess.Popen(args, cwd=os.path.dirname(SERVER_EXE), env=env,
                         stdout=out, stderr=err, creationflags=creation, close_fds=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_online(timeout=2):
            return True, "模型已加载"
        time.sleep(2)
    return False, "加载超时（看日志尾部）"


def apply_config(cfg):
    _busy["reloading"] = True
    _busy["message"] = "正在停止旧实例…"
    try:
        stop_server()
        _busy["message"] = "正在加载模型…"
        ok, msg = start_server(cfg)
        _busy["message"] = msg
        return ok, msg
    finally:
        _busy["reloading"] = False


# ------------------------------------------------------- 显存争用 & 干净基准
WORK_DIR = os.path.dirname(DEMO_DIR)
BENCH_HISTORY = os.path.join(WORK_DIR, "bench-history.json")
PROTOCOL_MD = os.path.join(WORK_DIR, "bonsai-reasoning-protocol.md")
_protocol_cache = {"t": 0.0, "text": ""}
PREFS_PATH = os.path.join(WORK_DIR, "dashboard-prefs.json")
DEFAULT_PREFS = {"protocol": True}
_prefs_cache = {"t": 0.0, "data": None}
BENCH_EXE = os.path.join(os.path.dirname(SERVER_EXE), "llama-bench.exe")

_bench = {
    "running": False, "stage": "", "started": 0.0, "finished": 0.0,
    "reps": 3, "compare": False, "others_gb": None,
    "results": [], "log_path": None, "error": None,
}


def nvidia_used_gb():
    g = gpu_info() or {}
    return round((g.get("mem_used_mb") or 0) / 1024.0, 2)


def vram_breakdown(cfg):
    """把显存拆成「本服务估算」和「其它程序估算」。

    WDDM 下 nvidia-smi 只给总量，本服务部分按 权重 + KV + 视觉塔 + 缓冲 估算。
    """
    g = gpu_info() or {}
    used_mb = g.get("mem_used_mb") or 0
    model = cfg.get("model") or ""
    if model and not os.path.isabs(model):
        model = os.path.join(DEMO_DIR, model)
    model_mb = os.path.getsize(model) / 1024 / 1024 if os.path.exists(model) else 0.0
    factor = CACHE_FACTOR.get(str(cfg.get("cache_type_k", "f16")), 1.0)
    kv_mb = 64 * 1024 * int(cfg.get("ctx") or 0) * factor / 1024 / 1024
    mm_mb = 0.0
    if cfg.get("use_mmproj") and not cfg.get("mmproj_cpu"):
        mm = cfg.get("mmproj") or ""
        if mm and not os.path.isabs(mm):
            mm = os.path.join(DEMO_DIR, mm)
        if mm and os.path.exists(mm):
            mm_mb = os.path.getsize(mm) / 1024 / 1024
    ours_mb = model_mb + kv_mb + mm_mb + 1200.0     # 最后一项 = 计算图/上下文缓冲经验值
    return {
        "total_gb": round((g.get("mem_total_mb") or 0) / 1024, 1),
        "used_gb": round(used_mb / 1024, 2),
        "ours_gb": round(ours_mb / 1024, 2),
        "others_gb": round(max(0.0, used_mb - ours_mb) / 1024, 2),
    }


def _bench_targets(cfg, compare):
    model = cfg.get("model") or ""
    abs_model = model if os.path.isabs(model) else os.path.join(DEMO_DIR, model)
    ctk = str(cfg.get("cache_type_k", "f16"))
    ctv = str(cfg.get("cache_type_v", "f16"))
    kv = f"{ctk}/{ctv}"
    targets = [{"label": f"{os.path.basename(abs_model)} [{kv}]",
                "exe": BENCH_EXE, "model": abs_model, "ctk": ctk, "ctv": ctv}]
    if not compare:
        return targets
    folder = os.path.dirname(abs_model)
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            low = name.lower()
            if (not low.endswith(".gguf") or "mmproj" in low
                    or "kv-bias" in low or "kv-mean-center" in low):
                continue
            path = os.path.join(folder, name)
            if os.path.abspath(path) == os.path.abspath(abs_model):
                continue
            targets.append({"label": f"{name} [{kv}]", "exe": BENCH_EXE,
                            "model": path, "ctk": ctk, "ctv": ctv})
    pre = os.path.join(DEMO_DIR, "bin", "cuda-prebuilt", "llama-bench.exe")
    if os.path.exists(pre):
        targets.append({"label": f"官方预编译二进制 + {os.path.basename(abs_model)} [{kv}]",
                        "exe": pre, "model": abs_model, "ctk": ctk, "ctv": ctv})
    return targets


def _parse_bench(stdout):
    """优先解析 llama-bench 的 JSON（-o json），失败则退回 markdown 表格。"""
    out = {}

    def test_name(row):
        tag = str(row.get("test") or "")
        if tag in ("pp512", "tg128"):
            return tag
        try:
            n_prompt = int(row.get("n_prompt") or 0)
            n_gen = int(row.get("n_gen") or 0)
        except Exception:
            return ""
        # 这个 build 的 JSON 不带 test 字段，用 n_prompt/n_gen 区分 pp / tg
        if n_prompt > 0 and n_gen == 0:
            return f"pp{n_prompt}"
        if n_gen > 0 and n_prompt == 0:
            return f"tg{n_gen}"
        return ""

    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            data = [data]
        for row in data:
            test = test_name(row)
            avg = row.get("avg_ts", row.get("t/s"))
            std = row.get("stddev_ts", row.get("stddev"))
            if test in ("pp512", "tg128") and avg is not None:
                out[test] = {"avg": float(avg), "std": float(std) if std is not None else None}
    except Exception:
        for line in stdout.splitlines():
            if "|" not in line:
                continue
            for test in ("pp512", "tg128"):
                if test not in line:
                    continue
                nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*±\s*([0-9]+(?:\.[0-9]+)?)", line)
                if nums:
                    out[test] = {"avg": float(nums[0][0]), "std": float(nums[0][1])}
    return out


def _run_bench_one(target, ngl, fa, reps, timeout=2400):
    args = [target["exe"], "-m", target["model"], "-ngl", str(ngl), "-fa", fa,
            "-ctk", target["ctk"], "-ctv", target["ctv"],
            "-p", "512", "-n", "128", "-r", str(reps), "-o", "json"]
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=DEMO_DIR)
    wall = time.time() - t0
    parsed = _parse_bench(proc.stdout or "")
    pp = parsed.get("pp512") or {}
    tg = parsed.get("tg128") or {}
    return {
        "label": target["label"], "exe": target["exe"], "model": target["model"],
        "kv": f'{target["ctk"]}/{target["ctv"]}', "reps": reps,
        "wall_s": round(wall, 1), "rc": proc.returncode,
        "pp512": pp.get("avg"), "pp512_std": pp.get("std"),
        "tg128": tg.get("avg"), "tg128_std": tg.get("std"),
        "raw": (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else ""),
    }


def _bench_worker(cfg, reps, compare):
    try:
        _bench.update({"running": True, "compare": bool(compare), "reps": reps,
                       "started": time.time(), "finished": 0.0, "results": [],
                       "log_path": None, "error": None, "others_gb": None,
                       "stage": "停止服务，腾出显存…"})
        stop_server()
        time.sleep(3)
        _bench["others_gb"] = nvidia_used_gb()

        ngl = int(cfg.get("ngl") or 99)
        fa = {"on": "1", "off": "0"}.get(str(cfg.get("flash_attn", "on")), "1")
        if str(cfg.get("cache_type_k", "f16")) != "f16":
            fa = "1"                     # 量化 KV 必须开 FlashAttention
        stime = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(WORK_DIR, f"bench-clean-{stime}.txt")
        targets = _bench_targets(cfg, compare)
        chunks, results = [], []
        for i, target in enumerate(targets, 1):
            _bench["stage"] = f"基准 {i}/{len(targets)}：{target['label']}"
            res = _run_bench_one(target, ngl, fa, reps)
            chunks.append(f"===== {target['label']} =====\n{res.pop('raw')}\n")
            results.append(res)
            _bench["results"] = list(results)

        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# 干净基准 {stime}\n")
            fh.write(f"# 其它程序占用显存（停服务后 nvidia-smi 实测）：{_bench['others_gb']} GB\n")
            fh.write(f"# ngl={ngl} fa={fa} reps={reps}\n\n")
            fh.write("\n".join(chunks))
        hist = []
        if os.path.exists(BENCH_HISTORY):
            try:
                with open(BENCH_HISTORY, "r", encoding="utf-8") as fh:
                    hist = json.load(fh)
            except Exception:
                hist = []
        hist.append({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "others_gb": _bench["others_gb"], "reps": reps,
                     "ngl": ngl, "fa": fa, "log_path": log_path, "results": results})
        with open(BENCH_HISTORY, "w", encoding="utf-8") as fh:
            json.dump(hist[-20:], fh, ensure_ascii=False, indent=2)
        _bench["log_path"] = log_path
        _bench["stage"] = "恢复服务…"
    except Exception as exc:
        _bench["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            ok, msg = start_server(cfg)
            prefix = "完成" if not _bench["error"] else f"出错（{_bench['error']}）"
            _bench["stage"] = f"{prefix} · 服务：{msg}"
        except Exception as exc:
            _bench["stage"] = f"恢复服务失败：{exc}"
        _bench["running"] = False
        _bench["finished"] = time.time()
        _status_cache["t"] = 0


def start_bench(reps=3, compare=False):
    with _lock:
        if _bench.get("running"):
            return False, "已有基准任务在跑"
        cfg = load_config()
        threading.Thread(target=_bench_worker, args=(cfg, int(reps), bool(compare)),
                         daemon=True).start()
    return True, "已开始：先停服务 → 跑 llama-bench → 自动恢复服务（期间不能对话）"


def bench_history(limit=5):
    if not os.path.exists(BENCH_HISTORY):
        return []
    try:
        with open(BENCH_HISTORY, "r", encoding="utf-8") as fh:
            return json.load(fh)[-limit:]
    except Exception:
        return []


def seed_bench_from_history():
    """面板重启后，把上次干净基准的结果填回状态，免得卡片空着。"""
    hist = bench_history(1)
    if not hist:
        return
    last = hist[-1]
    _bench["results"] = last.get("results") or []
    _bench["others_gb"] = last.get("others_gb")
    _bench["reps"] = last.get("reps")
    _bench["log_path"] = last.get("log_path")
    _bench["stage"] = f"上次干净基准：{last.get('at', '')}（{last.get('log_path', '')}）"


def list_files():
    models, mmprojs = [], []
    for root, _dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if not f.lower().endswith(".gguf"):
                continue
            rel = os.path.relpath(os.path.join(root, f), DEMO_DIR)
            (mmprojs if "mmproj" in f.lower() else models).append(rel)
    return sorted(models), sorted(mmprojs)


_limits_cache = {}
_meta_cache = {}


def gguf_meta(path):
    """只解析 GGUF 头部的 metadata（不读张量）。

    用于读层数 / 训练上下文，以及 KV 校准文件的 basis（kv_mean_center.k_rot）。
    """
    if not path or not os.path.exists(path):
        return {}
    if path in _meta_cache:
        return _meta_cache[path]
    scalars = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
               6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}

    def rd_str(fh):
        (n,) = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8", "replace")

    def rd_val(fh, t):
        if t == 8:
            return rd_str(fh)
        if t == 9:
            (et,) = struct.unpack("<I", fh.read(4))
            (cnt,) = struct.unpack("<Q", fh.read(8))
            return [rd_val(fh, et) for _ in range(min(cnt, 64))]
        fmt, size = scalars[t]
        return struct.unpack("<" + fmt, fh.read(size))[0]

    meta = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return {}
            _ver, _tensors, kv_count = struct.unpack("<IQQ", fh.read(20))
            for _ in range(kv_count):
                key = rd_str(fh)
                (t,) = struct.unpack("<I", fh.read(4))
                meta[key] = rd_val(fh, t)
    except Exception:
        return {}
    _meta_cache[path] = meta
    return meta


def gguf_limits(path):
    """从 GGUF 元数据里拿层数 / 训练上下文等硬上限。"""
    if not path or not os.path.exists(path):
        return {}
    if path in _limits_cache:
        return _limits_cache[path]
    meta = gguf_meta(path)
    if not meta:
        return {}
    arch = meta.get("general.architecture", "")
    out = {
        "arch": arch,
        "layers": meta.get(f"{arch}.block_count"),
        "ctx_train": meta.get(f"{arch}.context_length"),
        "embed": meta.get(f"{arch}.embedding_length"),
        "heads": meta.get(f"{arch}.attention.head_count"),
        "kv_heads": meta.get(f"{arch}.attention.head_count_kv"),
    }
    _limits_cache[path] = out
    return out


def limits_payload(cfg):
    model = cfg.get("model", "")
    if model and not os.path.isabs(model):
        model = os.path.join(DEMO_DIR, model)
    meta = gguf_limits(model)
    gpu = gpu_info() or {}
    total_mb = gpu.get("mem_total_mb") or 16384
    used_mb = gpu.get("mem_used_mb") or 0
    free_bytes = max(0, (total_mb - used_mb)) * 1024 * 1024
    factor = CACHE_FACTOR.get(str(cfg.get("cache_type_k", "f16")), 1.0)
    kv_per_token = 64 * 1024 * factor
    headroom = 1.2 * 1024 ** 3
    extra_ctx = int(max(0.0, free_bytes - headroom) / kv_per_token // 1024 * 1024)
    ctx_now = int(cfg.get("ctx") or 0)
    return {
        "model": meta,
        "ngl_layers": meta.get("layers") or 64,
        "ngl_max": 999,          # 99/999 = 连输出层等附加张量一起上卡
        "ctx_max": meta.get("ctx_train") or 262144,
        "ctx_max_now": ctx_now + extra_ctx,
        "threads_max": os.cpu_count() or 16,
        "parallel_max": 16,
        "batch_max": 8192,
        "ubatch_max": 2048,
        "temp_max": 2.0,
        "top_p_max": 1.0,
        "top_k_max": 1000,
        "min_p_max": 1.0,
        "vram_total_mb": total_mb,
        "vram_free_mb": max(0, total_mb - used_mb),
        "kv_per_token_kib": round(kv_per_token / 1024, 1),
    }


def find_kv_bias(model_path, need_rot=True):
    """在模型同目录查找 K 缓存均值中心校准文件，并校验 basis。

    fork 的加载器会拒绝 basis 不匹配的 bias（kv_mean_center.k_rot）：
    q4_0 K 缓存会自动开启 Hadamard 旋转，因此只挑旋转基（k_rot=True）的文件；
    官方 scripts\\make_kv_bias.sh 产出的 *-kv-bias.gguf 是非旋转基，跳过不选。
    """
    if not model_path:
        return None
    if not os.path.isabs(model_path):
        model_path = os.path.join(DEMO_DIR, model_path)
    folder = os.path.dirname(model_path)
    if not os.path.isdir(folder):
        return None
    cands = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if not low.endswith(".gguf"):
            continue
        if "kv-mean-center" in low or low.endswith("-kv-bias.gguf"):
            cands.append(os.path.join(folder, name))
    fallback = None
    for path in cands:
        k_rot = gguf_meta(path).get("kv_mean_center.k_rot")
        entry = {"path": path, "name": os.path.basename(path), "k_rot": k_rot}
        if k_rot is True:
            return entry
        if k_rot is None and fallback is None:
            fallback = entry          # 老文件没记录 basis：加载器只警告，先留作兜底
    if need_rot:
        return fallback
    for path in cands:
        if gguf_meta(path).get("kv_mean_center.k_rot") is False:
            return {"path": path, "name": os.path.basename(path), "k_rot": False}
    return fallback


def panel_inflight(delta):
    """Track how many requests the panel itself has in flight (to label the source)."""
    global PANEL_INFLIGHT
    with _live_lock:
        PANEL_INFLIGHT = max(0, PANEL_INFLIGHT + delta)


def protocol_text():
    """The visible-analysis protocol (same file the Codex catalog overlay is built from)."""
    try:
        mtime = os.path.getmtime(PROTOCOL_MD)
    except OSError:
        return ""
    with _lock:
        if _protocol_cache["text"] and _protocol_cache["t"] == mtime:
            return _protocol_cache["text"]
    try:
        with open(PROTOCOL_MD, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return ""
    idx = raw.find("【推理外显协议")
    text = raw[idx:].strip() if idx >= 0 else raw.strip()
    with _lock:
        _protocol_cache.update(t=mtime, text=text)
    return text


def load_prefs():
    """Panel UI preferences (persisted so toggles survive reloads and browser changes)."""
    try:
        mtime = os.path.getmtime(PREFS_PATH)
    except OSError:
        return dict(DEFAULT_PREFS)
    with _lock:
        if _prefs_cache["data"] is not None and _prefs_cache["t"] == mtime:
            return dict(_prefs_cache["data"])
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = {}
    merged = dict(DEFAULT_PREFS)
    if isinstance(data, dict):
        merged.update(data)
    with _lock:
        _prefs_cache.update(t=mtime, data=dict(merged))
    return merged


def save_prefs(update):
    prefs = load_prefs()
    prefs.update(update)
    tmp = PREFS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prefs, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, PREFS_PATH)
    with _lock:
        _prefs_cache.update(t=os.path.getmtime(PREFS_PATH), data=dict(prefs))
    return prefs


def _rate(points, idx, window, now):
    """tokens/s over the trailing `window` seconds of [(t, prompt_done, decoded)] samples."""
    if len(points) < 2:
        return None
    cut = now - window
    ref = points[-1]
    for p in points:
        if p[0] >= cut:
            ref = p
            break
    dt = now - ref[0]
    if dt <= 0.25:
        return None
    return max(0.0, (points[-1][idx] - ref[idx]) / dt)


def live_sampler(interval=0.3):
    """Poll llama-server /slots and turn it into live, per-request metrics.

    Works for every client (Codex / CC Switch / the panel chat) because it reads the
    server's slot state instead of watching one particular HTTP response.
    """
    hist = []
    while True:
        slots = None
        try:
            slots = fetch_json(f"{UPSTREAM}/slots", timeout=3)
        except Exception:
            slots = None
        now = time.time()
        busy = [s for s in (slots or []) if s.get("is_processing")]
        prompt_total = sum(int(s.get("n_prompt_tokens") or 0) for s in busy)
        prompt_done = sum(int(s.get("n_prompt_tokens_processed") or 0) for s in busy)
        cache = sum(int(s.get("n_prompt_tokens_cache") or 0) for s in busy)
        decoded = sum(int(((s.get("next_token") or [{}])[0]).get("n_decoded") or 0) for s in busy)

        try:
            with _live_lock:
                L = LIVE
                if busy and not L["active"]:
                    # a new request just started (any client)
                    L.update(active=True, phase="prefill", started=now, ttft=None,
                             input_tokens=prompt_total, cache_tokens=cache,
                             processed_tokens=prompt_done, output_tokens=0,
                             prefill_tps=None, decode_tps=None, elapsed=0.0,
                             series=[],
                             source=("panel" if PANEL_INFLIGHT else "external"))
                    hist = []
                if busy:
                    if decoded > 0 and L["ttft"] is None:
                        # first token seen -> TTFT includes prompt processing
                        L["ttft"] = max(0.05, now - (L["started"] or now))
                    if decoded > 0:
                        L["phase"] = "decode"
                    # lock the prompt size + cache hit at the first complete sample of the request
                    # (later samples can report a bigger n_prompt_tokens when the server shifts context)
                    if prompt_total and not L["input_tokens"]:
                        L["input_tokens"] = prompt_total
                    if cache and not L["cache_tokens"]:
                        L["cache_tokens"] = cache
                    L["processed_tokens"] = max(L["processed_tokens"], prompt_done)
                    L["output_tokens"] = max(L["output_tokens"], decoded)
                    L["elapsed"] = now - (L["started"] or now)
                    hist.append((now, prompt_done, decoded))
                    if len(hist) > 200:
                        del hist[:-200]
                    if L["phase"] == "prefill":
                        rate = _rate(hist, 1, 2.0, now)   # prompt tokens/s
                        if rate:
                            L["prefill_tps"] = rate
                    else:
                        rate = _rate(hist, 2, 2.5, now)   # decoded tokens/s
                        if rate is not None:
                            L["decode_tps"] = rate
                    v = L["decode_tps"] if L["decode_tps"] is not None else L["prefill_tps"]
                    if v:
                        L["series"].append({"t": round(now, 2), "v": round(v, 1)})
                        if len(L["series"]) > 240:
                            del L["series"][:-240]
                elif L["active"]:
                    # request finished -> keep a summary
                    el = L["elapsed"] or 0.0
                    ttft = L["ttft"]
                    out_tok = L["output_tokens"]
                    avg = None
                    if out_tok > 1 and ttft is not None and el > ttft:
                        avg = (out_tok - 1) / (el - ttft)
                    elif el > 0 and out_tok:
                        avg = out_tok / el
                    if L["prefill_tps"] is None and L["processed_tokens"] and ttft:
                        # prefill finished between two samples -> estimate from TTFT
                        L["prefill_tps"] = L["processed_tokens"] / ttft
                    rec = {"at": time.strftime("%H:%M:%S"), "source": L["source"],
                           "input": L["input_tokens"], "cache": L["cache_tokens"],
                           "output": out_tok, "ttft": round(ttft, 2) if ttft else None,
                           "seconds": round(el, 2),
                           "prefill_tps": round(L["prefill_tps"], 1) if L["prefill_tps"] else None,
                           "avg_tps": round(avg, 1) if avg else None}
                    L.update(active=False, phase="idle", last=rec,
                             decode_tps=None, elapsed=round(el, 2))
                    L["recent"] = (L["recent"] + [rec])[-8:]
                    T = L["totals"]
                    T["requests"] += 1
                    T["in"] += L["input_tokens"]
                    T["out"] += out_tok
                    T["seconds"] += el
                L["updated"] = now
        except Exception:
            pass
        time.sleep(interval)


def live_snapshot():
    with _live_lock:
        L = LIVE
        return {
            "active": L["active"], "phase": L["phase"], "source": L["source"],
            "ttft": round(L["ttft"], 2) if L["ttft"] else None,
            "elapsed": round(L["elapsed"], 2) if L["elapsed"] else None,
            "input_tokens": L["input_tokens"], "cache_tokens": L["cache_tokens"],
            "output_tokens": L["output_tokens"],
            "prefill_tps": round(L["prefill_tps"], 1) if L["prefill_tps"] else None,
            "decode_tps": round(L["decode_tps"], 1) if L["decode_tps"] is not None else None,
            "series": list(L["series"][-120:]),
            "last": dict(L["last"]) if L["last"] else None,
            "recent": [dict(r) for r in L["recent"][-4:]],
            "totals": dict(L["totals"]),
            "now": time.time(), "updated": L["updated"],
        }


def status_payload():
    now = time.time()
    with _lock:
        cached = _status_cache["data"] if (_status_cache["data"] and now - _status_cache["t"] < 1.5) else None
    if cached is None:
        cfg = load_config()
        cached = {
            "gpu": gpu_info(),
            "vram": vram_breakdown(cfg),
            "config": cfg,
            "derived": derived_info(cfg),
            "runtime": server_runtime_info(),
            "busy": dict(_busy),
            "bench": dict(_bench),
            "port": SERVER_PORT,
        }
        with _lock:
            _status_cache.update(t=time.time(), data=cached)
    data = dict(cached)
    data["live"] = live_snapshot()   # always fresh, so 1 Hz polling sees live numbers
    return data


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bonsai 2 27B · 推理控制台</title>
<style>
  :root{--bg:#0e1116;--panel:#161b22;--panel2:#1c232c;--line:#2a323d;--fg:#e6edf3;
        --dim:#8b98a5;--accent:#4ade80;--accent2:#60a5fa;--warn:#f59e0b;--hot:#f87171}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 18px;
    border-bottom:1px solid var(--line);background:linear-gradient(180deg,#12171e,#0e1116);position:sticky;top:0;z-index:5}
  .langbtn{margin-left:auto;font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--dim);cursor:pointer}
  .langbtn:hover{color:var(--fg);border-color:var(--fg)}
  .brand{font-weight:600}.brand span{color:var(--dim);font-weight:400;margin-left:8px;font-size:12px}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:4px 12px;border-radius:999px;
    background:var(--panel2);border:1px solid var(--line);font-size:12px;color:var(--dim)}
  .dot{width:8px;height:8px;border-radius:50%;background:#ef4444}.dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}
  main{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:14px;padding:14px;height:calc(100vh - 51px)}
  @media(max-width:1180px){main{grid-template-columns:1fr;height:auto}}
  .chat{display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;min-height:0}
  .messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;min-height:300px}
  .msg{max-width:92%;padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
  .msg.user{align-self:flex-end;background:#1e3a5f;border:1px solid #2b5480}
  .msg.assistant{align-self:flex-start;background:var(--panel2);border:1px solid var(--line)}
  /* Markdown 渲染：模型回复按 md 排版，用户消息保持纯文本 */
  .msg.md{white-space:normal}
  .msg.md > .body > *:first-child{margin-top:0}
  .msg.md > .body > *:last-child{margin-bottom:0}
  .msg.md p{margin:8px 0}
  .msg.md h1,.msg.md h2,.msg.md h3,.msg.md h4{margin:14px 0 6px;line-height:1.3}
  .msg.md h1{font-size:19px}.msg.md h2{font-size:17px}.msg.md h3{font-size:15.5px}
  .msg.md h4{font-size:14px;color:var(--dim)}
  .msg.md ul,.msg.md ol{margin:6px 0;padding-left:22px}
  .msg.md li{margin:3px 0}
  .msg.md a{color:var(--accent2)}
  .msg.md code{background:#0d1117;border:1px solid var(--line);border-radius:5px;padding:1px 5px;
    font:12.5px/1.5 "Cascadia Mono",Consolas,"Courier New",monospace}
  .msg.md pre{position:relative;background:#0d1117;border:1px solid var(--line);border-radius:8px;
    padding:10px 12px;margin:10px 0;overflow-x:auto}
  .msg.md pre code{background:none;border:none;padding:0;font-size:12.5px;white-space:pre}
  .msg.md pre .copy{position:absolute;top:6px;right:8px;padding:2px 8px;font-size:11px;opacity:.45}
  .msg.md pre:hover .copy{opacity:1}
  .msg.md blockquote{margin:8px 0;padding:2px 12px;border-left:3px solid #3b4756;color:#a9b6c4}
  .msg.md table{border-collapse:collapse;margin:10px 0;font-size:13px;display:block;overflow-x:auto;max-width:100%}
  .msg.md th,.msg.md td{border:1px solid var(--line);padding:4px 9px;white-space:nowrap}
  .msg.md th{background:#1b222b}
  .msg.md hr{border:none;border-top:1px solid var(--line);margin:12px 0}
  .role{font-size:11px;color:var(--dim);margin-bottom:4px;letter-spacing:.5px}
  details.think{margin:0 0 8px;border-left:2px solid #3b4756;padding-left:10px}
  details.think summary{cursor:pointer;color:var(--dim);font-size:12px}
  details.think p{margin:6px 0 0;color:#7f8b99;font-size:12.5px;white-space:pre-wrap}
  .cursor::after{content:"▍";color:var(--accent);animation:blink 1s steps(1) infinite}
  @keyframes blink{50%{opacity:0}}
  form{border-top:1px solid var(--line);padding:10px;background:var(--panel2)}
  textarea{width:100%;min-height:60px;max-height:170px;resize:vertical;background:#0f141a;color:var(--fg);
    border:1px solid var(--line);border-radius:10px;padding:10px 12px;font:inherit;outline:none}
  .row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;align-items:center}
  .chk{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--dim);cursor:pointer}
  .chk input{accent-color:var(--accent)}
  button{background:#22303f;color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:7px 14px;font:inherit;cursor:pointer}
  button:hover:not(:disabled){background:#2b3d50}button:disabled{opacity:.45;cursor:not-allowed}
  button.primary{background:#1d4ed8;border-color:#2563eb}button.primary:hover:not(:disabled){background:#2563eb}
  button.danger{background:#4a2020;border-color:#7f1d1d}
  .metrics{display:flex;flex-direction:column;gap:10px;overflow-y:auto;min-height:0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
  .card.big{padding:16px}
  .label{font-size:12px;color:var(--dim);letter-spacing:.4px;margin-bottom:6px}
  .value{font-variant-numeric:tabular-nums;font-weight:650;font-size:30px;line-height:1.1;color:var(--accent)}
  .value small{font-size:13px;color:var(--dim);font-weight:400;margin-left:6px}
  .value.small{font-size:19px;color:var(--fg)}
  .sub{font-size:12px;color:var(--dim);margin-top:5px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
  canvas{width:100%;height:82px;display:block}
  .bar{height:7px;background:#0f141a;border-radius:6px;overflow:hidden;margin-top:7px;border:1px solid var(--line)}
  .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .3s}
  .bar.hot i{background:linear-gradient(90deg,#f59e0b,var(--hot))}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
  .chip{font-size:11.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:var(--panel2);color:var(--dim)}
  .chip.ok{border-color:#14532d;background:#0f2a1a;color:#86efac}
  .chip.warn{border-color:#78350f;background:#2a1c07;color:#fcd34d}
  .formgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px 10px}
  .formgrid label{display:flex;flex-direction:column;gap:3px;font-size:11.5px;color:var(--dim)}
  .formgrid input,.formgrid select{background:#0f141a;color:var(--fg);border:1px solid var(--line);
    border-radius:7px;padding:5px 8px;font:inherit;font-size:12.5px;outline:none}
  .formgrid .wide{grid-column:1/-1;flex-direction:row;align-items:center;gap:8px}
  code{font-size:11.5px;color:#9db4c9;word-break:break-all}
  .msgline{font-size:12px;color:var(--warn);margin-left:6px}
</style>
</head>
<body>
<header>
  <div class="brand">Bonsai 2 27B <span>推理控制台</span></div>
  <button id="langBtn" type="button" class="langbtn" title="Switch to English">EN</button>
  <div class="pill"><i class="dot" id="dot"></i><span id="statusText">连接中…</span></div>
</header>
<main>
  <section class="chat">
    <div class="messages" id="messages">
      <div class="msg assistant"><div class="role">提示</div>右侧可实时看到生成速度、显卡温度与显存，也可以像 LM Studio 一样改加载参数后一键重载。</div>
    </div>
    <form id="form">
      <textarea id="input" placeholder="输入消息…（Enter 发送，Shift+Enter 换行）"></textarea>
      <div class="row">
        <button type="submit" class="primary" id="send">发送</button>
        <button type="button" id="stop">停止生成</button>
        <button type="button" id="clear">清空对话</button>
        <label class="chk" title="注入「分析/推理/存疑」外显协议并关闭隐式思考（协议文本：work\bonsai-reasoning-protocol.md）">
          <input type="checkbox" id="useProto" checked> 外显协议
        </label>
      </div>
    </form>
  </section>

  <aside class="metrics">
    <div class="card big">
      <div class="label">生成速度</div>
      <div class="value"><span id="tps">–</span><small>tok/s</small></div>
      <div class="sub" id="tpsSub">等待请求</div>
    </div>

    <div class="grid3">
      <div class="card"><div class="label">首 token</div><div class="value small"><span id="ttft">–</span><small>s</small></div></div>
      <div class="card"><div class="label">平均速度</div><div class="value small"><span id="avg">–</span><small>t/s</small></div></div>
      <div class="card"><div class="label">预填充</div><div class="value small"><span id="prefill">–</span><small>t/s</small></div></div>
      <div class="card"><div class="label">输出 tokens</div><div class="value small"><span id="ctok">0</span></div></div>
      <div class="card"><div class="label">输入 tokens</div><div class="value small"><span id="ptok">0</span></div></div>
      <div class="card"><div class="label">总耗时</div><div class="value small"><span id="total">–</span><small>s</small></div></div>
    </div>

    <div class="card"><div class="label">实时曲线（tok/s）</div><canvas id="spark" width="820" height="180"></canvas></div>

    <div class="card">
      <div class="label">显卡状态</div>
      <div class="sub" id="gpuName">–</div>
      <div class="grid3" style="margin-top:8px">
        <div><div class="label">利用率</div><div class="value small"><span id="gpuUtil">–</span><small>%</small></div></div>
        <div><div class="label">温度</div><div class="value small"><span id="gpuTemp">–</span><small>°C</small></div></div>
        <div><div class="label">功耗</div><div class="value small"><span id="gpuPower">–</span><small>W</small></div></div>
      </div>
      <div class="bar" id="vramBarWrap"><i id="vramBar"></i></div>
      <div class="sub" id="vramText">–</div>
    </div>

    <div class="card">
      <div class="label">显存争用 &amp; 干净基准</div>
      <div class="sub" id="vramSplit">–</div>
      <div class="row">
        <select id="benchReps">
          <option value="1">快测 1 次</option>
          <option value="3" selected>标准 3 次</option>
          <option value="5">精确 5 次</option>
        </select>
        <label style="margin:0"><input type="checkbox" id="benchCompare"> 对比其它量化 + 官方预编译</label>
      </div>
      <div class="row">
        <button type="button" id="benchBtn">跑干净基准（自动停/起服务）</button>
      </div>
      <div class="sub" id="benchMsg">说明：跑基准时会先停掉本服务腾出显存，<b>不会关闭其它任何程序</b>；跑完自动按当前参数恢复服务。</div>
    </div>

    <div class="card">
      <div class="label">引擎状态</div>
      <div class="chips">
        <span class="chip" id="chipFlash">FlashAttention：–</span>
        <span class="chip" id="chipKv">KV 缓存：–</span>
        <span class="chip" id="chipKvSize">KV 占用：–</span>
        <span class="chip" id="chipKvLoc">KV 位置：–</span>
        <span class="chip" id="chipCtx">上下文：–</span>
        <span class="chip" id="chipSlots">并发：–</span>
        <span class="chip" id="chipVision">视觉：–</span>
        <span class="chip" id="chipThink">思考：–</span>
        <span class="chip" id="chipKvBias">KV 校准：–</span>
      </div>
      <div class="sub" id="modelInfo">–</div>
    </div>

    <div class="card">
      <div class="label">可调参数上限（实时读取模型元数据 + 显卡）</div>
      <div class="chips" id="limitChips">加载中…</div>
      <div class="row">
        <button type="button" id="useMaxCtx">把上下文填到当前显存可支持的最大值</button>
      </div>
    </div>

    <div class="card">
      <div class="label">模型加载参数（改完点“应用并重载”）</div>
      <div class="formgrid">
        <label class="wide">模型文件<select id="cfgModel"></select></label>
        <label>上下文长度 -c<input id="cfgCtx" type="number" step="1024" min="512"></label>
        <label>GPU 层数 -ngl<input id="cfgNgl" type="number" min="0"></label>
        <label>KV 缓存 K<select id="cfgK"><option value="f16">f16（纯净）</option><option value="q8_0">q8_0（量化）</option><option value="q4_0">q4_0（量化）</option></select></label>
        <label>KV 缓存 V<select id="cfgV"><option value="f16">f16（纯净）</option><option value="q8_0">q8_0（量化）</option><option value="q4_0">q4_0（量化）</option></select></label>
        <label>FlashAttention<select id="cfgFlash"><option value="on">开启</option><option value="off">关闭</option><option value="auto">自动</option></select></label>
        <label>并发槽位 -np（留空=自动，推荐）<input id="cfgNp" type="number" min="1" max="16" placeholder="auto"></label>
        <label>CPU 线程 -t（空=自动）<input id="cfgThreads" type="number" min="1" max="64"></label>
        <label>batch -b<input id="cfgBatch" type="number" min="64" step="64"></label>
        <label>ubatch -ub<input id="cfgUbatch" type="number" min="32" step="32"></label>
        <label>温度 temp<input id="cfgTemp" type="number" step="0.05" min="0" max="2"></label>
        <label>top_p<input id="cfgTopp" type="number" step="0.05" min="0" max="1"></label>
        <label>top_k<input id="cfgTopk" type="number" min="0"></label>
        <label>min_p<input id="cfgMinp" type="number" step="0.01" min="0" max="1"></label>
        <label>presence_penalty<input id="cfgPresence" type="number" step="0.05" min="0" max="2"></label>
        <label class="wide"><input type="checkbox" id="cfgDry"> Enable DRY repeat suppression (fights "but wait / let me do" text loops; tool-call loops still need the protocol rules)</label>
        <label>DRY multiplier (0.5 mild / 0.8 recommended / 1.1 aggressive)<input id="cfgDryMult" type="number" step="0.1" min="0" max="2"></label>
        <label>DRY allowed length (repeat longer than N tokens before penalising)<input id="cfgDryLen" type="number" min="1" max="16"></label>
        <label>DRY sequence breakers (the default resets the penalty at newlines/colons/quotes; "none" also penalises repeats that span lines)<select id="cfgDryBrk">
          <option value="">default ('\n' : " *)</option>
          <option value="none">none (stronger, spans lines)</option>
        </select></label>
        <label class="wide"><input type="checkbox" id="cfgMmproj"> 启用视觉（mmproj）</label>
        <label class="wide"><input type="checkbox" id="cfgMmprojCpu"> 视觉塔放内存（省 ~0.9 GB 显存，只影响图片预填充）</label>
        <label class="wide"><input type="checkbox" id="cfgKvOffload"> KV 缓存放显存（--kv-offload，必须开才快）</label>
        <label class="wide"><input type="checkbox" id="cfgKvBias"> 使用 K 缓存校准 bias（仅 q4_0 生效，提升长上下文质量）</label>
<label class="wide"><input type="checkbox" id="cfgSpec"> 投机解码 MTP（草稿头已嫁接进主 GGUF，跑在目标权重上，+1.0 GB 显存；实测解码 +26~38%、12K 深度 +27%，代价是冷启动大 prompt 的 prefill 约 -45%，多轮会话前缀缓存不受影响）</label>
<label>MTP 草稿长度 --spec-draft-n-max<input id="cfgSpecN" type="number" min="1" max="8"></label>
<label>MTP 停止深度 --spec-draft-depth-max（0=不停止）<input id="cfgSpecDepth" type="number" min="0" max="262144" step="1024"></label>
        <label class="wide">思考预算 --reasoning-budget（空=不限）<input id="cfgBudget" type="number" min="-1"></label>
        <label class="wide">Codex 自动压缩阈值（token，写入模型目录/CC Switch，不是 llama-server 参数）<input id="cfgCompact" type="number" min="0" step="1024"></label>
        <label>思考模式<select id="cfgThinking">
          <option value="on">开启 · xhigh（官方默认，最强）</option>
          <option value="medium">开启 · medium（更快、回答更短）</option>
          <option value="off">关闭（直接回答，最快）</option>
        </select></label>
      </div>
      <div class="row">
        <button id="applyBtn" class="primary">应用并重载</button>
        <button id="stopSrvBtn">停止服务</button>
        <button id="startSrvBtn">启动服务</button>
        <button id="attachBtn">一键接入 CC Switch / Codex</button>
        <button id="exitBtn" class="danger">卸载模型并退出面板</button>
        <span class="msgline" id="cfgMsg"></span>
      </div>
      <div class="sub" id="attachMsg">重载模型后如果 CC Switch / Codex 还认旧的（模型名、上下文、思考档位不匹配），点上面的「一键接入」：会刷新 CC Switch 里的两条供应商条目、修正 Codex 的 config.toml（含四档思考目录）并做一次自检。</div>
      <div class="row" id="presetRow">
        <button type="button" data-preset="32">对话 32K · 纯净</button>
        <button type="button" data-preset="64">长文 64K · 纯净</button>
        <button type="button" data-preset="128">Agent 128K · q4_0</button>
        <button type="button" data-preset="256"
          title="模型训练上限 262144。128K 的 q8_0 KV 在 16GB 上已是极限，再往上必须换 q4_0（每 token 18.4 KB，262144 全量才 4.8 GB）。实测（2026-09-27，49K token 语料）：q4_0 相对 q8_0 的 PPL 是 4.6703 vs 4.6649 —— 统计上测不出差别；100K 深度 prefill 反而快 13%（692 vs 613 t/s），解码 19.6 vs 19.8 t/s（关 MTP 时相当）。代价：这个档位装不下 MTP（草稿上下文也要整窗 KV），而且 q4_0 的 vec 内核在线反量化，MTP 的收益会从 +30% 掉到 +3%，所以两者本就不该叠。">
          极限 256K · q4_0（自动关 MTP）
        </button>
        <button type="button" id="agentPresetBtn" class="primary"
          title="按多轮 agent 负载的实测形态定档：每轮上下文 p50 63K / p90 96K（集中在 32K~110K）→ 128K 窗口 + 100K 自动压缩（105~110K 是模型的退化区间）；每轮新增只有 p50 ~500 token（p90 ~1.3K）、输出 p50 ~300 token、每轮 1 次工具调用 → 128K + q8_0（近无损且快）、batch 8192/ubatch 1024（实测 ub 512/1024/2048 在这种"多轮小增量"下 10 轮总耗时 47.3/44.3/45.1 s，无差别；而 128K 档 ub 2048 会贴显存，故取 1024）、并发 1（KV 前缀缓存必定命中）、MTP + 图形状缓存（解码 60.8 vs 关图 29.8 t/s）、关思考 + 256 预算、精确采样档 0.3/0.9/40/min_p 0.05/presence 0、视觉塔放内存、自动压缩 100K">
          🤖 Agent 最优（实测调优）
        </button>
        <button type="button" id="shortMtpPresetBtn"
          title="短会话（ctx 64K）专用档：开 MTP 草稿头 n_max=2 + ubatch 2048（服务端自动开图形状缓存）。实测同配置解码 +10~34%（数学类任务最高），代价是 +1.5~3 GB 显存；实填超过 32K 的会话会变慢，长会话请用上面的 Agent 128K 档。ubatch 2048 的理由：权重重反量化按 ubatch 计费，实测 32K/64K/96K 上下文下预填充 +5~7%、解码 +12~25%，显存只多 ~240 MiB（96K 时 12.8 GB）；131072 档会贴边所以仍用 1024。">
          ⚡ 短会话 · MTP 加速（ctx 64K）
        </button>
      </div>
      <div class="row" id="styleRow">
        <button type="button" id="instructBtn">一键切官方 instruct 档（关思考 · 0.7 / 0.80 / 20 · presence 1.5）</button>
      </div>
      <div class="sub">提示：KV 缓存选 q8_0/q4_0 会自动要求 FlashAttention 开启；<code>-ngl 99</code> 表示全部层放显存；
        并发槽位大于 1 时总上下文会被均分（每槽 = 上下文 ÷ 并发），单人使用建议填 1。</div>
    </div>
  </aside>
</main>
<script>
const $ = (id) => document.getElementById(id);

/* ===== i18n: zh ⇄ en (generated) ===== */
const I18N_EN = {
 "推理控制台": "Inference Console",
 "CPU 线程 -t（空=自动）": "CPU threads -t (blank = auto)",
 "Codex 自动压缩阈值（token，写入模型目录/CC Switch，不是 llama-server 参数）": "Codex auto-compact threshold (tokens; written into the model catalog / CC Switch, not a llama-server flag)",
 "GPU 层数 -ngl": "GPU layers -ngl",
 "KV 位置：–": "KV location: –",
 "KV 占用：–": "KV usage: –",
 "KV 校准：–": "KV calibration: –",
 "KV 缓存 K": "KV cache K",
 "KV 缓存 V": "KV cache V",
 "KV 缓存放显存（--kv-offload，必须开才快）": "Keep the KV cache in VRAM (--kv-offload; required for speed)",
 "KV 缓存：–": "KV cache: –",
 "MTP 停止深度 --spec-draft-depth-max（0=不停止）": "MTP draft depth limit --spec-draft-depth-max (0 = never stop)",
 "MTP 草稿长度 --spec-draft-n-max": "MTP draft length --spec-draft-n-max",
 "f16（纯净）": "f16 (lossless)",
 "q4_0（量化）": "q4_0 (quantised)",
 "q8_0（量化）": "q8_0 (quantised)",
 "⚡ 短会话 · MTP 加速（ctx 64K）": "⚡ Short session · MTP boost (ctx 64K)",
 "一键切官方 instruct 档（关思考 · 0.7 / 0.80 / 20 · presence 1.5）": "Use the official instruct preset (thinking off · 0.7 / 0.80 / 20 · presence 1.5)",
 "一键接入 CC Switch / Codex": "One-click connect: CC Switch / Codex",
 "上下文长度 -c": "Context length -c",
 "上下文：–": "Context: –",
 "不会关闭其它任何程序": "no other program is touched",
 "使用 K 缓存校准 bias（仅 q4_0 生效，提升长上下文质量）": "Use the K-cache calibration bias (q4_0 only; better long-context quality)",
 "停止服务": "Stop service",
 "停止生成": "Stop generating",
 "关闭": "Off",
 "关闭（直接回答，最快）": "Off (answers directly, fastest)",
 "利用率": "Utilisation",
 "功耗": "Power",
 "加载中…": "Loading…",
 "卸载模型并退出面板": "Unload model & exit panel",
 "发送": "Send",
 "可调参数上限（实时读取模型元数据 + 显卡）": "Parameter limits (read live from the GGUF metadata + the GPU)",
 "右侧可实时看到生成速度、显卡温度与显存，也可以像 LM Studio 一样改加载参数后一键重载。": "Generation speed, GPU temperature and VRAM update live on the right; you can edit the load parameters and reload with one click, just like LM Studio.",
 "启动服务": "Start service",
 "启用视觉（mmproj）": "Enable vision (mmproj)",
 "复制": "Copy",
 "外显协议": "Visible-analysis protocol",
 "实时曲线（tok/s）": "Live curve (tok/s)",
 "对比其它量化 + 官方预编译": "Compare with other quants + the official prebuilt",
 "对话 32K · 纯净": "Chat 32K · lossless",
 "对话已清空。": "Chat cleared.",
 "已卸载模型，显存已释放": "Model unloaded, VRAM released",
 "平均速度": "Avg speed",
 "并发槽位 -np（留空=自动，推荐）": "Parallel slots -np (blank = auto, recommended)",
 "并发：–": "Parallel: –",
 "应用并重载": "Apply & reload",
 "开启": "On",
 "开启 · medium（更快、回答更短）": "On · medium (faster, shorter answers)",
 "开启 · xhigh（官方默认，最强）": "On · xhigh (official default, strongest)",
 "引擎状态": "Engine",
 "快测 1 次": "Quick · 1 run",
 "思考模式": "Thinking mode",
 "思考过程": "Thinking",
 "思考预算 --reasoning-budget（空=不限）": "Thinking budget --reasoning-budget (blank = unlimited)",
 "思考：–": "Thinking: –",
 "总耗时": "Total time",
 "把上下文填到当前显存可支持的最大值": "Fill the context to the largest value the current VRAM supports",
 "投机解码 MTP（草稿头已嫁接进主 GGUF，跑在目标权重上，+1.0 GB 显存；实测解码 +26~38%、12K 深度 +27%，代价是冷启动大 prompt 的 prefill 约 -45%，多轮会话前缀缓存不受影响）": "Speculative decoding (MTP): the draft head is grafted into the main GGUF and runs on the target weights (+1.0 GB VRAM). Measured decode +26–38%, +27% at 12K depth, at the cost of ~-45% prefill on a cold long prompt; multi-turn prefix caching is unaffected.",
 "提示": "Note",
 "提示：KV 缓存选 q8_0/q4_0 会自动要求 FlashAttention 开启；": "Note: a q8_0/q4_0 KV cache automatically requires FlashAttention;",
 "显卡状态": "GPU status",
 "显存争用 &amp; 干净基准": "VRAM contention &amp; clean benchmark",
 "极限 256K · q4_0（自动关 MTP）": "Extreme 256K · q4_0 (MTP auto-off)",
 "标准 3 次": "Standard · 3 runs",
 "模型加载参数（改完点“应用并重载”）": "Model load parameters (click “Apply & reload” when done)",
 "模型文件": "Model file",
 "清空对话": "Clear chat",
 "温度": "Temp",
 "温度 temp": "Temperature temp",
 "生成速度": "Generation speed",
 "等待请求": "Waiting for a request",
 "精确 5 次": "Precise · 5 runs",
 "自动": "Auto",
 "表示全部层放显存； 并发槽位大于 1 时总上下文会被均分（每槽 = 上下文 ÷ 并发），单人使用建议填 1。": "means all layers stay in VRAM; with parallel slots > 1 the total context is split evenly (per slot = context ÷ slots) — keep it at 1 for single-user use.",
 "视觉塔放内存（省 ~0.9 GB 显存，只影响图片预填充）": "Keep the vision tower in RAM (saves ~0.9 GB VRAM; only affects image prefill)",
 "视觉：–": "Vision: –",
 "说明：跑基准时会先停掉本服务腾出显存，": "The benchmark stops this service first to free VRAM;",
 "跑干净基准（自动停/起服务）": "Run clean benchmark (stops/starts the service)",
 "输入 tokens": "Input tokens",
 "输出 tokens": "Output tokens",
 "连接中…": "Connecting…",
 "重载模型后如果 CC Switch / Codex 还认旧的（模型名、上下文、思考档位不匹配），点上面的「一键接入」：会刷新 CC Switch 里的两条供应商条目、修正 Codex 的 config.toml（含四档思考目录）并做一次自检。": "If CC Switch / Codex still holds the old entry after a reload (model name, context or reasoning levels no longer match), click “One-click connect” above: it refreshes both provider entries in CC Switch, fixes Codex's config.toml (including the four reasoning levels) and runs a self-check.",
 "长文 64K · 纯净": "Long text 64K · lossless",
 "面板已关闭。想继续用就双击桌面「Bonsai 2 27B 控制台」。": "Panel closed. Double-click the “Bonsai 2 27B Console” desktop shortcut to start it again.",
 "预填充": "Prefill",
 "首 token": "First token",
 "；跑完自动按当前参数恢复服务。": "; the service is restored with the current parameters when it finishes.",
 "🤖 Agent 最优（实测调优）": "🤖 Best for agents (measured tuning)",
 "显存争用 & 干净基准": "VRAM contention & clean benchmark",
 "DRY multiplier（0.5 温和 / 0.8 推荐 / 1.1 激进）": "DRY multiplier (0.5 mild / 0.8 recommended / 1.1 aggressive)",
 "DRY allowed length（连续重复超过几个 token 才罚）": "DRY allowed length (penalise repeats longer than this many tokens)",
 "启用 DRY 防复读（压制 \"but wait / let me do\" 式重复输出；工具调用循环仍需协议规则）": "Enable DRY repeat suppression (fights \"but wait / let me do\" text loops; tool-call loops still need the protocol rules)",
 "DRY 分隔符（默认在换行/冒号/引号处重置惩罚；选 none 连跨行重复也罚）": "DRY sequence breakers (the default resets the penalty at newlines/colons/quotes; \"none\" also penalises repeats that span lines)",
 "默认（'\\n' : \" *）": "default ('\\n' : \" *)",
 "none（更强，跨行重复也罚）": "none (stronger, spans lines)",
 "Bonsai 2 27B · 推理控制台": "Bonsai 2 27B · Inference Console",
 "输入消息…（Enter 发送，Shift+Enter 换行）": "Type a message… (Enter to send, Shift+Enter for a new line)",
 "注入「分析/推理/存疑」外显协议并关闭隐式思考（协议文本：work\\bonsai-reasoning-protocol.md）": "Injects the analyse/reason/doubt visible-analysis protocol and turns implicit thinking off (protocol text: work\\bonsai-reasoning-protocol.md)",
 "CC Switch 供应商": "CC Switch provider",
 "CC Switch 在运行，供应商条目未动（需要时点「一键接入」）": "CC Switch is running; the provider entries were left alone (use “One-click connect” when needed)",
 "CC Switch 条目同步失败": "Failed to sync the CC Switch entries",
 "CC Switch 条目已同步": "CC Switch entries synced",
 "CPU 线程": "CPU threads",
 "Codex config 同步出错：%s": "Codex config sync error: %s",
 "Codex config.toml 已同步": "Codex config.toml synced",
 "Codex 当前供应商": "Codex's current provider",
 "Codex 模型目录": "Codex model catalog",
 "Codex 现在指向别的供应商（不是本地）—— 在 CC Switch 里点一下「Bonsai 2 27B（本地 V100）」": "Codex currently points at another provider (not the local one) — pick “Bonsai 2 27B (local V100)” in CC Switch",
 "GPU 层数": "GPU layers",
 "config.toml 已是最新（无需改动）": "config.toml is already up to date (nothing to change)",
 "llama-server 在线：%s（模型 %s）": "llama-server online: %s (model %s)",
 "ubatch 不能大于 batch，已下调": "ubatch cannot exceed batch — lowered automatically",
 "一键接入": "One-click connect",
 "上下文长度": "Context length",
 "先把本地服务跑起来，再点一次这个按钮。": "Start the local service first, then click this button again.",
 "包含 bin/cuda/llama-server.exe、models/ 与 dashboard-config.json 的目录": "the directory containing bin/cuda/llama-server.exe, models/ and dashboard-config.json",
 "即可切换；本次没有改动它的 config.toml": "to switch; its config.toml was left untouched this time",
 "就能用 /reasoning 切「关 / low / medium / xhigh」。": "then use /reasoning to switch between off / low / medium / xhigh.",
 "已停止服务（结束进程 {killed or '无'}）": "Service stopped (killed: {killed or 'none'})",
 "已写入/刷新 Codex 与 Claude 两条供应商（BASE_URL %s）": "Wrote/refreshed both the Codex and Claude provider entries (BASE_URL %s)",
 "已卸载模型（结束进程 {killed or '无'}）并关闭面板": "Model unloaded (killed: {killed or 'none'}) and the panel is closing",
 "已接入。在 CC Switch 里点一下「Bonsai 2 27B（本地 V100）」，再在 Codex 里新开一个会话，": "Connected. Pick “Bonsai 2 27B (local V100)” in CC Switch and open a new Codex session,",
 "已是最新": "up to date",
 "并发槽位": "Parallel slots",
 "找不到 %s": "%s not found",
 "找不到 ~/.codex/config.toml（未安装 Codex CLI/App？）": "~/.codex/config.toml not found (Codex CLI/App not installed?)",
 "无法连接 {UPSTREAM}：{exc}": "Cannot reach {UPSTREAM}: {exc}",
 "服务已在运行": "Service already running",
 "服务没在跑 —— 回面板点「启动服务」或「应用并重载」再接入": "The service is not running — click “Start service” or “Apply & reload” in the panel, then try again",
 "本地服务": "Local service",
 "模型目录已同步": "Model catalog synced",
 "缺少 ~/.codex/bonsai-model-catalog.json（生成脚本没跑成功）": "~/.codex/bonsai-model-catalog.json is missing (the generator script did not finish)",
 "警告：找不到 {SERVER_EXE}；请用 --demo-dir 指向正确的 bonsai-demo 目录": "Warning: {SERVER_EXE} not found; point --demo-dir at the right bonsai-demo directory",
 "（也可用环境变量 BONSAI_DEMO_DIR）": "(or set the BONSAI_DEMO_DIR environment variable)",
 "；CC Switch 已重启": "; CC Switch was restarted",
 "；CC Switch 当时没在运行": "; CC Switch was not running",
 "；CC Switch 重启失败，请手动打开": "; restarting CC Switch failed, please open it manually",
 "；数据库备份 ": "; database backup ",
 "{prefix} · 服务：{msg}": "{prefix} · service: {msg}",
 "从 GGUF 元数据里拿层数 / 训练上下文等硬上限。": "Reads the hard limits (layer count, trained context) from the GGUF metadata.",
 "优先 taskkill；某些沙箱会拦 taskkill.exe，退回 PowerShell 的 Stop-Process。": "Prefers taskkill; some sandboxes block taskkill.exe, so it falls back to PowerShell Stop-Process.",
 "优先解析 llama-bench 的 JSON（-o json），失败则退回 markdown 表格。": "Parses llama-bench JSON (-o json) first, falling back to the markdown table.",
 "停止服务，腾出显存…": "Stopping the service to free VRAM…",
 "内存（--no-kv-offload，会变慢）": "RAM (--no-kv-offload, slower)",
 "加载超时（看日志尾部）": "Loading timed out (check the tail of the log)",
 "完成": "Done",
 "官方预编译二进制 + {os.path.basename(abs_model)} [{kv}]": "Official prebuilt binary + {os.path.basename(abs_model)} [{kv}]",
 "已开始：先停服务 → 跑 llama-bench → 自动恢复服务（期间不能对话）": "Started: stop service → run llama-bench → restore the service automatically (no chat while it runs)",
 "已有基准任务在跑": "A benchmark task is already running",
 "开启（官方推荐）": "On (recommended)",
 "恢复服务…": "Restoring the service…",
 "恢复服务失败：{exc}": "Failed to restore the service: {exc}",
 "显存（--kv-offload，解码更快）": "VRAM (--kv-offload, faster decode)",
 "模型已加载": "Model loaded",
 "正在停止旧实例…": "Stopping the old instance…",
 "正在加载模型…": "Loading the model…",
 "端口 {SERVER_PORT} 仍被进程占用：{busy}；先「停止服务」再启动": "Port {SERVER_PORT} is still held by process {busy}; stop the service first",
 "端口 {SERVER_PORT} 仍被进程占用：{remaining}（可能没有权限结束它）": "Port {SERVER_PORT} is still held by {remaining} (may lack permission to kill it)",
 "纯净 FP16（未量化）": "pure FP16 (unquantised)",
 "部分在内存（仅 {ngl} 层上 GPU）": "partly in RAM (only {ngl} layers on the GPU)",
 "面板重启后，把上次干净基准的结果填回状态，免得卡片空着。": "After a panel restart the last clean-benchmark result is restored so the card is not empty.",
 "上次干净基准：{last.get('at', '')}（{last.get('log_path', '')}）": "Last clean benchmark: {last.get('at', '')} ({last.get('log_path', '')})",
 "出错（{_bench['error']}）": "Error ({_bench['error']})",
 "基准 {i}/{len(targets)}：{target['label']}": "Benchmark {i}/{len(targets)}: {target['label']}",
 "按多轮 agent 负载的实测形态定档：每轮上下文 p50 63K / p90 96K（集中在 32K~110K）→ 128K 窗口 + 100K 自动压缩（105~110K 是模型的退化区间）；每轮新增只有 p50 ~500 token（p90 ~1.3K）、输出 p50 ~300 token、每轮 1 次工具调用 → 128K + q8_0（近无损且快）、batch 8192/ubatch 1024（实测 ub 512/1024/2048 在这种": "Tuned to the measured shape of a multi-turn agent workload: context per turn p50 63K / p90 96K (mostly 32K–110K) → 128K window + 100K auto-compact (105–110K is where the model degrades); each turn adds only ~500 tokens (p90 ~1.3K) and outputs ~300 tokens with ~1 tool call → 128K + q8_0 (near-lossless and fast), batch 8192/ubatch 1024 (10-turn runs measured 47.3/44.3/45.1 s for ub 512/1024/2048 — no difference; at 128K, ub 2048 runs out of VRAM headroom, hence 1024), 1 slot (the prefix cache always hits), MTP + CUDA-graph shape cache (decode 60.8 vs 29.8 t/s with graphs off), thinking off + 256 budget, precision sampling 0.3/0.9/40/min_p 0.05/presence 0, vision tower in RAM, auto-compact 100K",
 "模型训练上限 262144。128K 的 q8_0 KV 在 16GB 上已是极限，再往上必须换 q4_0（每 token 18.4 KB，262144 全量才 4.8 GB）。实测（2026-09-27，49K token 语料）：q4_0 相对 q8_0 的 PPL 是 4.6703 vs 4.6649 —— 统计上测不出差别；100K 深度 prefill 反而快 13%（692 vs 613 t/s），解码 19.6 vs 19.8 t/s（关 MTP 时相当）。代价：这个档位装不下 MTP（草稿上下文也要整窗 KV），而且 q4_0 的 vec 内核在线反量化，MTP 的收益会从 +30% 掉到 +3%，所以两者本就不该叠。": "The training ceiling is 262,144 tokens. A 128K q8_0 KV cache is already the limit on 16 GB; beyond that you must switch to q4_0 (18.4 KB per token — the full 262,144 only needs 4.8 GB). Measured (2026-09-27, 49K-token corpus): q4_0 vs q8_0 PPL 4.6703 vs 4.6649 — statistically indistinguishable; at 100K depth prefill is 13% faster (692 vs 613 t/s) and decode 19.6 vs 19.8 t/s (a wash with MTP off). The catch: this preset cannot fit MTP (the draft context needs a full-window KV cache too) and q4_0's vector kernel dequantises inline, which drops MTP's gain from +30% to +3% — so the two should not be combined.",
 "短会话（ctx 64K）专用档：开 MTP 草稿头 n_max=2 + ubatch 2048（服务端自动开图形状缓存）。实测同配置解码 +10~34%（数学类任务最高），代价是 +1.5~3 GB 显存；实填超过 32K 的会话会变慢，长会话请用上面的 Agent 128K 档。ubatch 2048 的理由：权重重反量化按 ubatch 计费，实测 32K/64K/96K 上下文下预填充 +5~7%、解码 +12~25%，显存只多 ~240 MiB（96K 时 12.8 GB）；131072 档会贴边所以仍用 1024。": "Short-session (ctx 64K) preset: MTP draft head at n_max=2 + ubatch 2048 (the server enables the CUDA-graph shape cache automatically). Measured with the same settings: decode +10–34% (best on maths), at the cost of +1.5–3 GB VRAM; sessions that actually exceed 32K get slower — use the Agent 128K preset above for those. Why ubatch 2048: weight dequantisation is charged per ubatch — measured at 32K/64K/96K context, prefill +5–7% and decode +12–25% for only ~240 MiB extra VRAM (12.8 GB at 96K); the 131072 setting runs too close to the edge, so it stays at 1024."
};
const I18N_RULES = [
  [/^按多轮 agent 负载的实测形态定档[\s\S]*$/, "Tuned to the measured shape of a multi-turn agent workload: context per turn p50 63K / p90 96K (mostly 32K–110K) → 128K window + 100K auto-compact (105–110K is where the model degrades); each turn adds only ~500 tokens (p90 ~1.3K) and outputs ~300 tokens with ~1 tool call → 128K + q8_0 (near-lossless and fast), batch 8192/ubatch 1024, 1 slot (the prefix cache always hits), MTP + CUDA-graph shape cache (decode 60.8 vs 29.8 t/s with graphs off), thinking off + 256 budget, precision sampling 0.3/0.9/40/min_p 0.05/presence 0, vision tower in RAM, auto-compact 100K"],
  [/^按你本人的真实用法定的档[\s\S]*$/, "Tuned to the author's real Codex usage: context per turn p50 ~63K / p90 ~96K (mostly 32K–110K) → 128K window + 100K auto-compact; each turn adds only ~500 tokens (p90 ~1.3K) and outputs ~300 tokens with ~1 tool call → 128K + q8_0, batch 8192/ubatch 1024, 1 slot (prefix cache always hits), MTP + CUDA-graph shape cache, thinking off + 256 budget, precision sampling 0.3/0.9/40/min_p 0.05/presence 0, vision tower in RAM, auto-compact 100K"],
  [/^模型训练上限 262144。[\s\S]*$/, "The training ceiling is 262,144 tokens. A 128K q8_0 KV cache is already the limit on 16 GB; beyond that switch to q4_0 (18.4 KB per token — the full 262,144 needs 4.8 GB). Measured (2026-09-27, 49K-token corpus): q4_0 vs q8_0 PPL 4.6703 vs 4.6649 — statistically indistinguishable; at 100K depth prefill is 13% faster (692 vs 613 t/s) and decode 19.6 vs 19.8 t/s (a wash with MTP off). This preset cannot fit MTP (the draft context needs a full-window KV cache too) and q4_0's vector kernel dequantises inline, which drops MTP's gain from +30% to +3% — do not combine them."],
  [/^短会话（ctx 64K）专用档[\s\S]*$/, "Short-session (ctx 64K) preset: MTP draft head at n_max=2 + ubatch 2048 (the server enables the CUDA-graph shape cache automatically). Measured with the same settings: decode +10–34% (best on maths), at the cost of +1.5–3 GB VRAM; sessions that actually exceed 32K get slower — use the Agent 128K preset. Why ubatch 2048: weight dequantisation is charged per ubatch — at 32K/64K/96K context prefill +5–7% and decode +12–25% for ~240 MiB extra VRAM; the 131072 setting runs too close to the edge, so it stays 1024."],
  [/^注入「分析\/推理\/存疑」外显协议[\s\S]*$/, "Injects the analyse/reason/doubt visible-analysis protocol and turns implicit thinking off (protocol text: work\\bonsai-reasoning-protocol.md)"],
  [/^CPU 线程 -t：(.+)$/, "CPU threads -t: \\1"],
  [/^KV 每 token：(.+)$/, "KV per token: \\1"],
  [/^• (.+)：pp (.+) t\/s ｜ tg (.+) t\/s（(.+) 次）$/, "• \\1: pp \\2 t/s | tg \\3 t/s (\\4 runs)"],
  [/^官方预编译二进制 \+ (.+)$/, "Official prebuilt binary + \\1"],
  [/^估算：本服务 ≈ (.+) ｜ 其它程序 ≈ (.+)（整卡 (.+)）$/, "Estimated: this service ≈ \\1 | other processes ≈ \\2 (card \\3)"],
  [/^其它程序占显存 (.+)（干净）$/, "Other processes use \\1 VRAM (clean)"],
  [/^显存 (.+) \/ (.+) GB（(.+)%）$/, "VRAM \\1 / \\2 GB (\\3%)"],
  [/^显存：(.+) 空闲 \/ (.+) 总量$/, "VRAM: \\1 free / \\2 total"],
  [/^层数 -ngl：模型 (.+) 层（填 (.+) = 连附加张量全部上卡）$/, "Layers -ngl: the model has \\1 layers (use \\2 = every tensor on the GPU)"],
  [/^并发 -np：留空\(自动\) 或 (.+)$/, "Parallel -np: blank (auto) or \\1"],
  [/^并发槽位：(.+)$/, "Parallel slots: \\1"],
  [/^当前显存可支持：(.+)$/, "VRAM can support: \\1"],
  [/^服务在线 · (.+) · 上下文 (.+)$/, "Online · \\1 · context \\2"],
  [/^架构：(.+)$/, "Architecture: \\1"],
  [/^模型：(.+) · (.+)$/, "Model: \\1 · \\2"],
  [/^采样：温度 (.+)，(.+)$/, "Sampling: temp \\1, \\2"],
  [/^训练上下文上限：(.+)$/, "Trained context limit: \\1"],
  [/^(FlashAttention)：(.+)$/, "\\1: \\2"],
  [/^\/ 自动压缩 (.+)$/, "/ auto-compact \\1"],
  [/^(.+) → (.+)（范围 0–(.+)）$/, "\\1 → \\2 (range 0–\\3)"],
  [/^(.+) → (.+)（范围 (.+?)–(.+)）$/, "\\1 → \\2 (range \\3–\\4)"],
  [/^上游 (\d+): (.+)$/, "upstream \\1: \\2"],
  [/^量化 (.+?)\/(.+)$/, "quantised \\1/\\2"],
  [/^并发：(.+)$/, "Parallel: \\1"],
  [/^上下文：(.+)$/, "Context: \\1"],
  [/^思考：(.+)$/, "Thinking: \\1"],
  [/^视觉：(.+)$/, "Vision: \\1"],
  [/^KV 缓存：(.+)$/, "KV cache: \\1"],
  [/^KV 位置：(.+)$/, "KV location: \\1"],
  [/^KV 占用：(.+)$/, "KV usage: \\1"],
  [/^KV 校准：(.+)$/, "KV calibration: \\1"],
  [/^(.+) → 档位 (.+)$/, "\\1 → level \\2"],
  [/^已停止服务（结束进程 (.+)）$/, "Service stopped (killed: \\1)"],
  [/^已卸载模型（结束进程 (.+)）$/, "Model unloaded (killed: \\1)"],
  [/^(FlashAttention：)(.+)$/, "\\1\\2"]
];
const I18N_GLOBAL = [
  [/（(\d+) 次）/g, "(\\1 runs)"],
  [/：pp /g, ": pp "],
  [/ ｜ /g, " | "],
  [/官方预编译二进制/g, "Official prebuilt binary"],
  [/其它程序占显存/g, "Other processes use"],
  [/（干净）/g, " (clean)"]
];
const I18N_EN_WS = {};
for (const k in I18N_EN) I18N_EN_WS[k.replace(/\s+/g, " ").trim()] = I18N_EN[k];
const I18N_KEY = "bonsai.lang";
const I18N_SKIP = new Set(["SCRIPT", "STYLE", "PRE", "CODE", "TEXTAREA"]);
const _i18nOrig = new WeakMap();
const _i18nText = (k, v) => ({ ok: true, text: v });

function i18nExcluded(node) {
  for (let el = node.nodeType === 1 ? node : node.parentElement; el; el = el.parentElement) {
    if (el.id === "messages" || (el.classList && el.classList.contains("md"))) return true;
    if (I18N_SKIP.has(el.tagName)) return true;
    if (el.hasAttribute && el.hasAttribute("data-i18n-skip")) return true;
  }
  return false;
}

function i18nLookup(src) {
  const s = String(src);
  const t = s.trim();
  if (!t) return null;
  if (I18N_EN[t] !== undefined) return s.replace(t, I18N_EN[t]);
  const ws = t.replace(/\s+/g, " ");
  if (I18N_EN_WS[ws] !== undefined) return s.replace(t, I18N_EN_WS[ws]);
  for (const [re, rep] of I18N_RULES) {
    if (re.test(t)) {
      const nv = t.replace(re, rep);
      if (nv !== t) return s.replace(t, nv);
    }
  }
  let out = t;
  for (const [re, rep] of I18N_GLOBAL) out = out.replace(re, rep);
  if (out !== t) return s.replace(t, out);
  return null;
}

function i18nTextNode(node, lang) {
  if (!node || node.nodeType !== 3 || i18nExcluded(node)) return;
  let orig = _i18nOrig.get(node);
  if (orig === undefined) { orig = node.nodeValue; _i18nOrig.set(node, orig); }
  if (lang !== "en") { if (node.nodeValue !== orig) node.nodeValue = orig; return; }
  const en = i18nLookup(orig);
  if (en !== null && node.nodeValue !== en) node.nodeValue = en;
}

function i18nAttrs(root, lang) {
  const els = root.querySelectorAll ? root.querySelectorAll("[title],[placeholder]") : [];
  els.forEach(el => {
    ["title", "placeholder"].forEach(a => {
      if (!el.hasAttribute(a)) return;
      const key = "_i18n_" + a;
      if (el.dataset[key] === undefined) el.dataset[key] = el.getAttribute(a);
      const orig = el.dataset[key];
      if (lang !== "en") { el.setAttribute(a, orig); return; }
      const en = i18nLookup(orig);
      if (en !== null) el.setAttribute(a, en);
    });
  });
}

function applyI18n(lang, root) {
  const scope = root || document.body;
  const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) i18nTextNode(walker.currentNode, lang);
  i18nAttrs(scope, lang);
  const t = document.querySelector("title");
  if (t) {
    if (t.dataset.zh === undefined) t.dataset.zh = t.textContent;
    const en = lang === "en" ? i18nLookup(t.dataset.zh) : null;
    t.textContent = en !== null ? en : t.dataset.zh;
  }
}

function currentLang() { return localStorage.getItem(I18N_KEY) === "en" ? "en" : "zh"; }

function setLang(lang) {
  if (lang !== "en") lang = "zh";
  localStorage.setItem(I18N_KEY, lang);
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  applyI18n(lang);
  const b = document.getElementById("langBtn");
  if (b) { b.textContent = lang === "en" ? "中文" : "EN"; b.title = lang === "en" ? "切换到中文" : "Switch to English"; }
}
/* ===== i18n: zh ⇄ en (generated init) ===== */

























const messagesEl = $("messages"), inputEl = $("input");
let history = [], running = false, controller = null, samples = [], timer = null;
let reloading = false;

function chip(el, text, cls) { el.textContent = text; el.className = "chip" + (cls ? " " + cls : ""); }

async function poll() {
  let s;
  try { s = await (await fetch("/api/status", {cache:"no-store"})).json(); }
  catch (e) { $("dot").classList.remove("on"); $("statusText").textContent = "面板无法连接"; return; }

  const rt = s.runtime, d = s.derived, c = s.config, g = s.gpu;
  $("dot").classList.toggle("on", !!rt.online);
  const bits = [rt.online ? "服务在线" : "服务离线"];
  if (rt.ftype) bits.push(rt.ftype.split(" - ")[0]);
  if (rt.ctx) bits.push("上下文 " + rt.ctx.toLocaleString());
  $("statusText").textContent = bits.join(" · ");

  if (g) {
    $("gpuName").textContent = g.name;
    $("gpuUtil").textContent = g.util;
    $("gpuTemp").textContent = g.temp_c;
    $("gpuPower").textContent = g.power_w != null ? g.power_w.toFixed(0) : "–";
    const pct = g.mem_total_mb ? g.mem_used_mb / g.mem_total_mb * 100 : 0;
    $("vramBar").style.width = pct.toFixed(1) + "%";
    $("vramBarWrap").className = "bar" + (g.temp_c >= 80 ? " hot" : "");
    $("vramText").textContent = "显存 " + (g.mem_used_mb/1024).toFixed(2) + " / " +
      (g.mem_total_mb/1024).toFixed(1) + " GB（" + pct.toFixed(0) + "%）";
    $("gpuTemp").style.color = g.temp_c >= 80 ? "#f87171" : (g.temp_c >= 70 ? "#fcd34d" : "#e6edf3");
  }

  const lv = s.live;
  if (lv && !running) renderLive(lv);   // 面板自身没在对话时，实时接管外部请求（Codex / CC Switch）

  if (s.vram) {
    const v = s.vram;
    $("vramSplit").textContent = "估算：本服务 ≈ " + v.ours_gb.toFixed(1) + " GB ｜ 其它程序 ≈ " +
      v.others_gb.toFixed(1) + " GB（整卡 " + v.used_gb.toFixed(1) + " / " + v.total_gb.toFixed(1) + " GB）";
    $("vramSplit").style.color = v.others_gb >= 3 ? "#f87171" : (v.others_gb >= 1.5 ? "#fcd34d" : "#8b98a5");
  }
  const b = s.bench || {};
  $("benchBtn").disabled = !!b.running;
  if (b.running) {
    $("benchMsg").style.color = "#fcd34d";
    $("benchMsg").textContent = "⏳ " + (b.stage || "运行中") + "（服务暂停，跑完自动恢复）";
  } else if (b.error) {
    $("benchMsg").style.color = "#f87171";
    $("benchMsg").textContent = "❌ " + b.error + (b.stage ? " ｜ " + b.stage : "");
  } else {
    const rows = (b.results || []).filter(r => r.pp512 != null || r.tg128 != null);
    if (rows.length) {
      const lines = rows.map(r => "• " + r.label + "：pp " + (r.pp512 != null ? r.pp512.toFixed(1) : "–") +
        " t/s ｜ tg " + (r.tg128 != null ? r.tg128.toFixed(1) : "–") + " t/s（" + r.reps + " 次）");
      if (b.others_gb != null) lines.push("其它程序占显存 " + b.others_gb.toFixed(1) + " GB" +
        (b.others_gb >= 3 ? "（偏高，数字会偏慢）" : "（干净）"));
      $("benchMsg").style.color = "#4ade80";
      $("benchMsg").style.whiteSpace = "pre-wrap";
      $("benchMsg").textContent = lines.join("\n");
    } else if (b.stage) {
      $("benchMsg").style.color = "#8b98a5";
      $("benchMsg").textContent = b.stage;
    }
  }

  if (d) {
    chip($("chipFlash"), "FlashAttention：" + d.flash, d.flash === "开启" ? "ok" : (d.flash === "关闭" ? "warn" : ""));
    chip($("chipKv"), "KV 缓存：" + d.kv_label, d.kv_quantized ? "warn" : "ok");
    chip($("chipKvSize"), "KV 占用：约 " + d.kv_gib + " GiB", "");
    chip($("chipKvLoc"), "KV 位置：" + (c.ngl >= 99 ? "显存" : "部分内存"), c.ngl >= 99 ? "ok" : "warn");
    const perSlot = d.ctx_per_slot || c.ctx;
    const sharedSlots = d.parallel_auto !== true && d.parallel > 1;
    chip($("chipCtx"), "上下文：" + c.ctx.toLocaleString() +
         (sharedSlots ? "（每槽 " + perSlot.toLocaleString() + "）" : "（独占）"), sharedSlots ? "warn" : "ok");
    chip($("chipSlots"), "并发槽位：" + (d.parallel_auto ? "自动(unified)" : (rt.slots || d.parallel)),
         sharedSlots ? "warn" : "");
    chip($("chipVision"), "视觉：" + (rt.vision ? "已启用" : "未启用"), rt.vision ? "ok" : "");
    const thinkText = {on: "开启 xhigh", medium: "开启 medium", off: "关闭"}[c.thinking || "on"];
    chip($("chipThink"), "思考：" + thinkText, (c.thinking === "off") ? "warn" : "");
    if (d.kv_bias_active) chip($("chipKvBias"), "KV 校准：已启用", "ok");
    else if (d.kv_bias_file) chip($("chipKvBias"), "KV 校准：已生成（仅 q4_0 可挂，官方限制）", "");
    else chip($("chipKvBias"), "KV 校准：未生成", "");
  }
  if (rt.model_path) $("modelInfo").textContent = "模型：" + rt.model_path.split("\\").pop() + (rt.ftype ? "  ·  " + rt.ftype : "");

  if (s.busy && s.busy.reloading) { $("cfgMsg").textContent = s.busy.message || "重载中…"; reloading = true; $("applyBtn").disabled = true; }
  else if (reloading) { reloading = false; $("applyBtn").disabled = false; $("cfgMsg").textContent = s.busy.message || "完成"; }
  const benchRunning = !!(s.bench && s.bench.running);
  $("stopSrvBtn").disabled = benchRunning;
  $("startSrvBtn").disabled = benchRunning;
  if (benchRunning) { $("applyBtn").disabled = true; }
}

async function loadConfigForm() {
  const r = await (await fetch("/api/loadconfig", {cache:"no-store"})).json();
  const c = r.config;
  const lim = r.limits || {};
  window.__lim = lim;
  if (lim.ctx_max) $("cfgCtx").max = lim.ctx_max;
  if (lim.ngl_max) $("cfgNgl").max = lim.ngl_max;
  if (lim.threads_max) $("cfgThreads").max = lim.threads_max;
  if (lim.parallel_max) $("cfgNp").max = lim.parallel_max;
  if (lim.batch_max) $("cfgBatch").max = lim.batch_max;
  if (lim.ubatch_max) $("cfgUbatch").max = lim.ubatch_max;
  if (lim.top_k_max) $("cfgTopk").max = lim.top_k_max;
  const m = lim.model || {};
  const rows = [
    ["架构", m.arch || "–"],
    ["层数 -ngl", "模型 " + (lim.ngl_layers || 64) + " 层（填 99 = 连附加张量全部上卡）"],
    ["训练上下文上限", "512 – " + (lim.ctx_max || 0).toLocaleString()],
    ["当前显存可支持", "≤ " + (lim.ctx_max_now || 0).toLocaleString()],
    ["KV 每 token", (lim.kv_per_token_kib || 64) + " KiB"],
    ["CPU 线程 -t", "1 – " + (lim.threads_max || 32)],
    ["并发 -np", "留空(自动) 或 1 – " + (lim.parallel_max || 16)],
    ["batch / ubatch", "64–" + (lim.batch_max || 8192) + " / 32–" + (lim.ubatch_max || 2048)],
    ["采样", "温度 0–" + (lim.temp_max || 2) + "，top_p/min_p 0–1，top_k 0–" + (lim.top_k_max || 1000)],
    ["显存", ((lim.vram_free_mb || 0)/1024).toFixed(1) + " GB 空闲 / " + ((lim.vram_total_mb || 0)/1024).toFixed(1) + " GB 总量"]
  ];
  $("limitChips").innerHTML = rows.map(([k, v]) => '<span class="chip">' + k + "：" + v + "</span>").join("");
  const sel = $("cfgModel"); sel.innerHTML = "";
  r.models.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); });
  if (!r.models.includes(c.model)) { const o = document.createElement("option"); o.value = c.model; o.textContent = c.model; sel.appendChild(o); }
  sel.value = c.model;
  $("cfgCtx").value = c.ctx; $("cfgNgl").value = c.ngl;
  $("cfgK").value = c.cache_type_k; $("cfgV").value = c.cache_type_v;
  $("cfgFlash").value = c.flash_attn; $("cfgNp").value = c.parallel;
  $("cfgThreads").value = c.threads; $("cfgBatch").value = c.batch; $("cfgUbatch").value = c.ubatch;
  $("cfgTemp").value = c.temp; $("cfgTopp").value = c.top_p; $("cfgTopk").value = c.top_k;
  $("cfgMinp").value = c.min_p; $("cfgBudget").value = c.reasoning_budget;
  $("cfgCompact").value = (c.compact_tokens === undefined || c.compact_tokens === null) ? 100000 : c.compact_tokens;
  $("cfgPresence").value = (c.presence_penalty === undefined || c.presence_penalty === null) ? 0 : c.presence_penalty;
  $("cfgDry").checked = Number(c.dry_multiplier || 0) > 0;
  $("cfgDryMult").value = (c.dry_multiplier === undefined || c.dry_multiplier === null) ? 0.8 : c.dry_multiplier;
  $("cfgDryLen").value = (c.dry_allowed_length === undefined || c.dry_allowed_length === null) ? 2 : c.dry_allowed_length;
  $("cfgDryBrk").value = c.dry_sequence_breaker || "";
  $("cfgMmproj").checked = !!c.use_mmproj;
  $("cfgMmprojCpu").checked = !!c.mmproj_cpu;
  $("cfgKvOffload").checked = c.kv_offload !== false;
  $("cfgKvBias").checked = c.use_kv_bias !== false;
  $("cfgThinking").value = c.thinking || "on";
  $("cfgSpec").checked = !!(c.spec_type && String(c.spec_type).toLowerCase() !== "none");
  $("cfgSpecN").value = (c.spec_draft_n_max === undefined || c.spec_draft_n_max === null) ? 2 : c.spec_draft_n_max;
  $("cfgSpecDepth").value = (c.spec_draft_depth_max === undefined || c.spec_draft_depth_max === null) ? 0 : c.spec_draft_depth_max;
}

function collectConfig() {
  return {
    model: $("cfgModel").value,
    mmproj: (window.__cfg || {}).mmproj,
    use_mmproj: $("cfgMmproj").checked,
    ctx: parseInt($("cfgCtx").value || "8192", 10),
    ngl: parseInt($("cfgNgl").value || "99", 10),
    cache_type_k: $("cfgK").value, cache_type_v: $("cfgV").value,
    flash_attn: $("cfgFlash").value,
    parallel: ($("cfgNp").value.trim() === "" ? "auto" : parseInt($("cfgNp").value, 10)),
    threads: $("cfgThreads").value, batch: parseInt($("cfgBatch").value || "2048", 10),
    ubatch: parseInt($("cfgUbatch").value || "512", 10),
    temp: parseFloat($("cfgTemp").value || "0.5"), top_p: parseFloat($("cfgTopp").value || "0.85"),
    top_k: parseInt($("cfgTopk").value || "20", 10), min_p: parseFloat($("cfgMinp").value || "0"),
    presence_penalty: parseFloat($("cfgPresence").value || "0"),
    dry_multiplier: ($("cfgDry").checked ? parseFloat($("cfgDryMult").value || "0.8") : 0),
    dry_base: 1.75,
    dry_allowed_length: parseInt($("cfgDryLen").value || "2", 10),
    dry_penalty_last_n: 64,
    dry_sequence_breaker: $("cfgDryBrk").value,
    reasoning_budget: $("cfgBudget").value
    , compact_tokens: parseInt($("cfgCompact").value || "0", 10)
    , kv_offload: $("cfgKvOffload").checked
    , thinking: $("cfgThinking").value
    , mmproj_cpu: $("cfgMmprojCpu").checked
    , use_kv_bias: $("cfgKvBias").checked
    , spec_type: ($("cfgSpec").checked ? "draft-mtp" : "")
    , spec_draft_n_max: parseInt($("cfgSpecN").value || "2", 10)
    , spec_draft_depth_max: parseInt($("cfgSpecDepth").value || "0", 10)
  };
}

async function postConfig(action) {
  const body = {action, config: collectConfig()};
  $("cfgMsg").textContent = action === "apply" ? "正在重载…（首次加载约 30–60 秒）" : "执行中…";
  $("applyBtn").disabled = true;
  try {
    const r = await (await fetch("/api/loadconfig", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)})).json();
    $("cfgMsg").textContent = (r.ok ? "✅ " : "⚠️ ") + (r.message || "");
    if (r.log_tail && !r.ok) console.warn(r.log_tail.join("\n"));
  } catch (e) { $("cfgMsg").textContent = "❌ " + e.message; }
  $("applyBtn").disabled = false;
  poll();
}

/* ---------- 轻量 Markdown 渲染（自带实现，不依赖外网 CDN，离线可用） ---------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function mdInline(s) {
  s = s.replace(/`([^`]+)`/g, (m, c) => "<code>" + c + "</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}
function renderMarkdown(src) {
  const lines = escapeHtml(String(src == null ? "" : src).replace(/\r\n?/g, "\n")).split("\n");
  const out = [], listStack = [];
  const closeLists = (depth) => { while (listStack.length > depth) out.push(listStack.pop()); };
  const isList = (l) => /^\s*([-*+]|\d+[.)])\s+/.test(l);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```(\S*)\s*$/);
    if (fence) {                                   // ``` 代码块
      const lang = fence[1] || "";
      const buf = []; i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; closeLists(0);
      out.push('<pre data-lang="' + lang + '"><button class="copy" type="button">复制</button><code>'
               + buf.join("\n") + "</code></pre>");
      continue;
    }
    if (/^\s*$/.test(line)) { closeLists(0); i++; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {                                       // 标题
      closeLists(0);
      out.push("<h" + h[1].length + ">" + mdInline(h[2].trim()) + "</h" + h[1].length + ">");
      i++; continue;
    }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { closeLists(0); out.push("<hr>"); i++; continue; }
    if (/^\s*&gt;\s?/.test(line)) {                 // 引用
      const buf = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*&gt;\s?/, "")); i++; }
      closeLists(0);
      out.push("<blockquote>" + mdInline(buf.join("\n")).replace(/\n/g, "<br>") + "</blockquote>");
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      closeLists(0);                                // 表格
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(line);
      const rows = []; i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { rows.push(cells(lines[i])); i++; }
      out.push("<table><thead><tr>" + head.map((c) => "<th>" + mdInline(c) + "</th>").join("")
        + "</tr></thead><tbody>"
        + rows.map((r) => "<tr>" + r.map((c) => "<td>" + mdInline(c) + "</td>").join("") + "</tr>").join("")
        + "</tbody></table>");
      continue;
    }
    if (isList(line)) {                             // 列表（支持两层缩进）
      const mm = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      const depth = Math.min(2, Math.floor(mm[1].length / 2)) + 1;
      const ordered = /\d/.test(mm[2]);
      if (listStack.length > depth) closeLists(depth);
      if (listStack.length < depth) { out.push(ordered ? "<ol>" : "<ul>"); listStack.push(ordered ? "</ol>" : "</ul>"); }
      out.push("<li>" + mdInline(mm[3]) + "</li>");
      i++; continue;
    }
    const para = [line]; i++;                       // 段落
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !isList(lines[i])
           && !/^\s*(#{1,6}\s|```|&gt;)/.test(lines[i])) { para.push(lines[i]); i++; }
    closeLists(0);
    out.push("<p>" + mdInline(para.join("\n")).replace(/\n/g, "<br>") + "</p>");
  }
  closeLists(0);
  return out.join("\n");
}
let mdLastPaint = 0;
function renderBody(el, text, force) {
  const now = performance.now();
  if (!force && now - mdLastPaint < 120) return;    // 流式输出时节流，避免每 token 重排
  mdLastPaint = now;
  el.innerHTML = renderMarkdown(text);
}
function addMessage(role, text, streaming) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const roleEl = document.createElement("div"); roleEl.className = "role";
  roleEl.textContent = role === "user" ? "你" : "Bonsai";
  const body = document.createElement("div");
  body.className = "body" + (streaming ? " cursor" : "");
  if (role === "assistant") { wrap.classList.add("md"); body.innerHTML = renderMarkdown(text || ""); }
  else { body.textContent = text; }
  wrap.appendChild(roleEl); wrap.appendChild(body); messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}
function getThink(wrap) {
  let d = wrap.querySelector("details.think");
  if (!d) { d = document.createElement("details"); d.className = "think";
    d.innerHTML = "<summary>思考过程</summary><p></p>"; wrap.insertBefore(d, wrap.querySelector(".body")); }
  return d.querySelector("p");
}
function drawSpark() {
  const w = spark.width, h = spark.height, ctx = spark.getContext("2d");
  ctx.clearRect(0,0,w,h); ctx.strokeStyle = "#232c37"; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++){ const y=(h-6)*i/4+3; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
  if (samples.length < 2) return;
  const maxV = Math.max(5, ...samples.map(s=>s.v));
  ctx.beginPath();
  samples.forEach((s,i)=>{ const x=w*i/(samples.length-1), y=h-6-(h-12)*(s.v/maxV); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.strokeStyle="#4ade80"; ctx.lineWidth=2; ctx.stroke();
  ctx.lineTo(w,h); ctx.lineTo(0,h); ctx.closePath(); ctx.fillStyle="rgba(74,222,128,.12)"; ctx.fill();
  ctx.fillStyle="#8b98a5"; ctx.font="20px sans-serif"; ctx.fillText("峰值 "+maxV.toFixed(1)+" tok/s", 8, 24);
}
function resetMetrics(){ samples=[]; drawSpark(); ["tps","avg","ttft","prefill","total"].forEach(i=>$(i).textContent="–");
  $("ctok").textContent="0"; $("ptok").textContent="0"; $("tpsSub").textContent="生成中…"; }

/* 实时接管：数据来自面板后台对 llama-server /slots 的采样，任何客户端（Codex / CC Switch / 本面板）都算数 */
function renderLive(lv){
  const src = lv.source === "panel" ? "面板对话" : "Codex / CC Switch";
  const fmt = (v, d) => (v === null || v === undefined || isNaN(v)) ? "–" : Number(v).toFixed(d);
  if (lv.active) {
    const decoding = lv.phase === "decode";
    const cur = decoding ? lv.decode_tps : lv.prefill_tps;
    $("tps").textContent = fmt(cur, decoding ? 1 : 0);
    $("tpsSub").textContent = "● " + (decoding ? "生成中" : "预填充中") + " · 来源：" + src +
      (lv.cache_tokens ? " · 缓存命中 " + lv.cache_tokens + " tok" : "");
    $("ttft").textContent = lv.ttft === null ? "…" : fmt(lv.ttft, 2);
    $("prefill").textContent = fmt(lv.prefill_tps, 0);
    $("ctok").textContent = lv.output_tokens || 0;
    $("ptok").textContent = lv.input_tokens || 0;
    $("total").textContent = fmt(lv.elapsed, 2);
    const el = lv.elapsed || 0, tt = lv.ttft || 0, out = lv.output_tokens || 0;
    $("avg").textContent = (out > 1 && el > tt) ? ((out - 1) / (el - tt)).toFixed(1) : "–";
  } else if (lv.last) {
    const r = lv.last, t = lv.totals || {};
    $("tps").textContent = fmt(r.avg_tps, 1);
    $("ttft").textContent = fmt(r.ttft, 2);
    $("avg").textContent = fmt(r.avg_tps, 1);
    $("prefill").textContent = fmt(r.prefill_tps, 0);
    $("ctok").textContent = r.output;
    $("ptok").textContent = r.input;
    $("total").textContent = fmt(r.seconds, 2);
    $("tpsSub").textContent = "上次请求（" + (r.source === "panel" ? "面板对话" : "Codex / CC Switch") +
      " " + r.at + "）· 累计 " + (t.requests || 0) + " 次 / 输出 " + (t.out || 0) + " tok";
  }
  if (lv.series && lv.series.length) {
    const nowMs = performance.now(), nowS = lv.now || (Date.now() / 1000);
    samples = lv.series.map(p => ({t: nowMs - (nowS - p.t) * 1000, v: p.v}));
    drawSpark();
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || running) return;
  inputEl.value = ""; addMessage("user", text, false); history.push({role:"user", content:text});
  const bubble = addMessage("assistant","",true), bodyEl = bubble.querySelector(".body");
  resetMetrics(); running = true; $("send").disabled = true;
  controller = new AbortController();
  const t0 = performance.now();
  let firstTokenAt=null, deltas=0, textOut="", thinkOut="", usage=null, timings=null;
  timer = setInterval(()=>{ if(!firstTokenAt) return;
    const now=performance.now(), recent=samples.filter(s=>now-s.t<2500);
    $("tps").textContent = (recent.length? recent.reduce((a,b)=>a+b.v,0)/recent.length : 0).toFixed(1); }, 200);
  try {
    const resp = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"}, signal:controller.signal,
      body: JSON.stringify({messages: history, stream:true, protocol: $("useProto").checked})});
    if (!resp.ok) throw new Error("HTTP " + resp.status + " " + (await resp.text()).slice(0,200));
    const reader = resp.body.getReader(), dec = new TextDecoder(); let buf="";
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      buf += dec.decode(value, {stream:true}); let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0,idx).trim(); buf = buf.slice(idx+1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim(); if (payload === "[DONE]") continue;
        let obj; try { obj = JSON.parse(payload); } catch(e){ continue; }
        if (obj.usage) usage = obj.usage;
        if (obj.timings) timings = obj.timings;
        const delta = ((obj.choices||[])[0]||{}).delta || {};
        if (delta.reasoning_content) { thinkOut += delta.reasoning_content; getThink(bubble).textContent = thinkOut; }
        if (delta.content) {
          const now = performance.now();
          if (firstTokenAt === null) { firstTokenAt = now; $("ttft").textContent = ((now-t0)/1000).toFixed(2); $("tpsSub").textContent="正在生成…"; }
          textOut += delta.content; deltas++;
          samples.push({t:now, v: deltas/Math.max(0.05,(now-firstTokenAt)/1000)});
          if (samples.length>400) samples.shift();
          drawSpark(); renderBody(bodyEl, textOut); $("ctok").textContent = deltas;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { renderBody(bodyEl, textOut + "\n\n_[已停止]_", true); }
    else { bodyEl.textContent = "请求失败：" + e.message; }
  } finally { clearInterval(timer); running = false; $("send").disabled = false; bodyEl.classList.remove("cursor"); }

  const tEnd = performance.now(), totalS = (tEnd-t0)/1000;
  const pt = usage? usage.prompt_tokens||0 : 0, ct = usage? usage.completion_tokens||deltas : deltas;
  $("ptok").textContent = pt || "–"; $("ctok").textContent = ct; $("total").textContent = totalS.toFixed(2);
  if (timings) {
    if (timings.predicted_per_second) $("avg").textContent = timings.predicted_per_second.toFixed(1);
    if (timings.prompt_per_second) $("prefill").textContent = timings.prompt_per_second.toFixed(0);
  }
  if (!$("avg").textContent || $("avg").textContent === "–") {
    const ds = firstTokenAt ? (tEnd-firstTokenAt)/1000 : 1; $("avg").textContent = (Math.max(1,ct-1)/ds).toFixed(1);
  }
  $("tps").textContent = $("avg").textContent;
  $("tpsSub").textContent = "首 token " + (firstTokenAt? ((firstTokenAt-t0)/1000).toFixed(2):"–") + "s · 输出 " + ct + " tok · 用时 " + totalS.toFixed(2) + "s";
  if (textOut) { renderBody(bodyEl, textOut, true); history.push({role:"assistant", content:textOut}); }
  poll();
}

const spark = $("spark");
document.getElementById("form").addEventListener("submit", (e)=>{ e.preventDefault(); send(); });
inputEl.addEventListener("keydown", (e)=>{ if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("stop").addEventListener("click", ()=>{ if (controller) controller.abort(); });
$("clear").addEventListener("click", ()=>{ if(running) return; history=[]; messagesEl.innerHTML='<div class="msg assistant"><div class="role">提示</div>对话已清空。</div>'; resetMetrics(); });

/* 面板偏好：外显协议开关持久化（服务端 work\dashboard-prefs.json） */
async function loadPrefs(){
  try {
    const p = await (await fetch("/api/prefs", {cache:"no-store"})).json();
    if (typeof p.protocol === "boolean") $("useProto").checked = p.protocol;
  } catch (e) {}
}
$("useProto").addEventListener("change", async ()=>{
  try {
    await fetch("/api/prefs", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({protocol: $("useProto").checked})});
  } catch (e) {}
});
// 代码块的“复制”按钮（事件委托，消息是动态插入的）
messagesEl.addEventListener("click", (e)=>{
  const btn = e.target.closest("pre > .copy"); if (!btn) return;
  const code = btn.parentElement.querySelector("code");
  const text = code ? code.textContent : "";
  const done = () => { btn.textContent = "已复制"; setTimeout(()=>{ btn.textContent = "复制"; }, 1200); };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, done);
  else { const ta=document.createElement("textarea"); ta.value=text; document.body.appendChild(ta); ta.select();
         try { document.execCommand("copy"); } catch(_) {} ta.remove(); done(); }
});
$("applyBtn").addEventListener("click", ()=> postConfig("apply"));
$("stopSrvBtn").addEventListener("click", ()=> postConfig("stop"));
$("startSrvBtn").addEventListener("click", ()=> postConfig("start"));
$("exitBtn").addEventListener("click", async ()=>{
  if (!confirm("会先卸载模型释放显存，然后关闭本面板。继续？")) return;
  $("cfgMsg").textContent = "正在卸载模型并退出面板…";
  $("exitBtn").disabled = true;
  try {
    const r = await (await fetch("/api/loadconfig", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action:"shutdown"})})).json();
    $("cfgMsg").textContent = "✅ " + (r.message || "已退出");
    document.body.innerHTML = '<div style="padding:40px;font:16px/1.6 \'Segoe UI\',sans-serif;color:#e6edf3">'
      + '<h2>已卸载模型，显存已释放</h2><p>面板已关闭。想继续用就双击桌面「Bonsai 2 27B 控制台」。</p></div>';
  } catch (e) {
    $("cfgMsg").textContent = "⚠️ " + e.message + "（模型可能已卸载，可直接关掉这个页面）";
  }
});
$("useMaxCtx").addEventListener("click", ()=>{
  const lim = window.__lim || {};
  if (!lim.ctx_max_now) { $("cfgMsg").textContent = "还没读到上限，稍后重试"; return; }
  $("cfgCtx").value = lim.ctx_max_now;
  $("cfgMsg").textContent = "已填入 " + lim.ctx_max_now.toLocaleString() + "（按当前空闲显存估算），点「应用并重载」生效";
});
document.querySelectorAll("#presetRow button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = {
      // ub (ubatch) 按上下文长度给：ubatch 越大，每个 ubatch 里"权重反量化"的次数越少，
      // 实测（12K 探针，见 RESEARCH 第 18.3 / 20.5 节）：ub 2048 比 1024 预填充 +7.4%、
      // 解码 +12%，显存只多 ~240 MiB；但 131072 档在 16 GiB 上会贴边（曾出现 WDDM 回退），
      // 所以 128K/256K 保持 ub 1024。
      "32":  { ctx: 32768,  k: "f16",  mmcpu: false, ub: 2048 },
      "64":  { ctx: 65536,  k: "f16",  mmcpu: true,  ub: 2048 },
      "128": { ctx: 131072, k: "q4_0", mmcpu: true,  ub: 1024 },
      // 256K must run WITHOUT MTP: the draft context needs its own KV for the whole
      // window, and 262144+q4_0+MTP does not fit in 16 GiB (measured: fit refused;
      // 229376+MTP fits at 14875 MiB but MTP buys only +3% there because the q4_0
      // vec kernel dequantizes in-kernel). See RESEARCH doc section 十五.
      "256": { ctx: 262144, k: "q4_0", mmcpu: true, mtp: false, ub: 1024 }
    }[btn.dataset.preset];
    if (!preset) return;
    $("cfgCtx").value = preset.ctx;
    $("cfgK").value = preset.k;
    $("cfgV").value = preset.k;
    if (preset.ub) $("cfgUbatch").value = preset.ub;
    $("cfgMmprojCpu").checked = preset.mmcpu;
    if (preset.mtp === false) { $("cfgSpec").checked = false; }
    if (preset.k !== "f16") $("cfgFlash").value = "on";
    $("cfgMsg").textContent = "已套用预设「" + btn.textContent + "」" +
      (preset.mtp === false ? "（已自动关闭 MTP：256K + MTP 在 16GB 上装不下）" : "") +
      "，点「应用并重载」生效";
  });
});

$("attachBtn").addEventListener("click", async ()=>{
  const btn = $("attachBtn"), msg = $("attachMsg");
  btn.disabled = true;
  msg.style.color = "#fcd34d";
  msg.textContent = "正在接入…（会短暂重启 CC Switch，约 10–20 秒）";
  try {
    const r = await (await fetch("/api/attach", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"})).json();
    const lines = (r.steps || []).map(s => (s.ok ? "✅ " : "⚠️ ") + escapeHtml(s.name) + "：" + escapeHtml(s.detail));
    if (r.hint) { lines.push(""); lines.push(escapeHtml(r.hint)); }
    msg.innerHTML = lines.join("<br>");
    msg.style.color = r.ok ? "#4ade80" : "#fcd34d";
  } catch (e) {
    msg.style.color = "#f87171";
    msg.textContent = "❌ " + e.message;
  }
  btn.disabled = false;
});

$("instructBtn").addEventListener("click", ()=>{
  $("cfgThinking").value = "off";
  $("cfgTemp").value = 0.7; $("cfgTopp").value = 0.80; $("cfgTopk").value = 20;
  $("cfgMinp").value = 0; $("cfgPresence").value = 1.5;
  $("cfgMsg").textContent = "已填入官方 instruct 档（0.7 / 0.80 / 20 / min_p 0 / presence 1.5，思考关闭），点「应用并重载」生效";
});

/* Agent 最优：本项目在 V100 上逐项实测出来的组合（见部署说明第十四～十六节） */
/* 短会话 MTP 加速档：ctx 64K + ubatch 2048 + MTP n_max=2（形状缓存由服务端自动加）。
   实测（work/mtp/RESULTS.md）：ctx 65536 下 MTP n=2 四任务解码均值 55.3 t/s vs 明文 ~48 t/s，
   数学类 62.4 t/s；但 55K 实填的长会话里 MTP 只有 19 t/s（明文 29.5），所以这一档只给短会话。 */
$("shortMtpPresetBtn").addEventListener("click", ()=>{
  $("cfgCtx").value = 65536;
  $("cfgBatch").value = 8192; $("cfgUbatch").value = 2048;
  $("cfgSpec").checked = true; $("cfgSpecN").value = 2; $("cfgSpecDepth").value = 0;
  /* DRY repeat suppression: 0.8 with allowed-length 2 is the usual llama.cpp setting;
     drop it to 0.5 (or untick the box) if normal repeated structures in code get clipped. */
  $("cfgDry").checked = true; $("cfgDryMult").value = 0.8; $("cfgDryLen").value = 2;
  $("cfgDry").checked = true; $("cfgDryMult").value = 0.8; $("cfgDryLen").value = 2;
  $("cfgMsg").textContent = "已填入「短会话 · MTP 加速」：ctx 64K + ubatch 2048 + MTP 草稿长度 2（图形状缓存自动开）—— 点「应用并重载」生效。128K 长会话请用「🤖 Agent 最优」档（那一档也开 MTP）。";
});

$("agentPresetBtn").addEventListener("click", ()=>{
  $("cfgCtx").value = 131072;
  $("cfgNgl").value = 99;
  $("cfgK").value = "q8_0"; $("cfgV").value = "q8_0";
  $("cfgFlash").value = "on";
  $("cfgNp").value = 1;
  /* ubatch 决定「每轮复用后还要重算多少 token」（≈ubatch+新 token）。10 轮会话实测总耗时：
     2048 → 49.0 s，1024 → 40.0 s，512 → 39.5 s；冷启动 5.9K 提示 2048=8.2 s / 1024=8.7 s / 512=10.4 s。
     单发超长提示（5 万 token 一次性喂）则是 2048 最快，那种用法请手动把 ubatch 调回 2048。 */
  $("cfgBatch").value = 8192; $("cfgUbatch").value = 1024;
  /* 精确档：实测 8/8 正确、token 最少（见部署说明第十六节）；聊天/创作可点「官方 instruct 档」 */
  $("cfgTemp").value = 0.3; $("cfgTopp").value = 0.9; $("cfgTopk").value = 40;
  $("cfgMinp").value = 0.05; $("cfgPresence").value = 0;
  $("cfgMmproj").checked = true; $("cfgMmprojCpu").checked = true;
  $("cfgKvOffload").checked = true; $("cfgKvBias").checked = false;
  $("cfgThinking").value = "off"; $("cfgBudget").value = 256;
  $("cfgCompact").value = 100000;
  /* MTP：嫁接头（blk.64，ProCreations on-policy Q8）实测在本卡上 0/12K/24K/48K 深度
     解码分别 +26%/+27%/+31%/+22%，冷启动 prefill 多付 0.85 ms/token；前缀缓存不受影响，
     所以多轮会话开、一次性的超长冷提示可以临时勾掉。 */
  $("cfgSpec").checked = true; $("cfgSpecN").value = 2; $("cfgSpecDepth").value = 0;
  $("cfgMsg").textContent = "已填入「Agent 最优」：128K + q8_0 + batch 8192/ubatch 1024（多轮对话最优；单发超长提示请手动改回 2048）+ 并发1 + 关思考(预算256) + 精确采样档(0.3/0.9/40/min_p 0.05/presence 0) + 视觉塔在内存 + 自动压缩 100K + MTP 草稿(嫁接头, depth 0) —— 点「应用并重载」生效";
});

$("benchBtn").addEventListener("click", async ()=>{
  const body = {action:"run", reps: parseInt($("benchReps").value || "3", 10), compare: $("benchCompare").checked};
  $("benchBtn").disabled = true;
  $("benchMsg").style.color = "#fcd34d";
  $("benchMsg").textContent = "已提交：即将停服务并开跑…";
  try {
    const r = await (await fetch("/api/bench", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)})).json();
    if (!r.ok) { $("benchMsg").style.color = "#f87171"; $("benchMsg").textContent = "⚠️ " + r.message; $("benchBtn").disabled = false; }
  } catch (e) { $("benchMsg").style.color = "#f87171"; $("benchMsg").textContent = "❌ " + e.message; $("benchBtn").disabled = false; }
  poll();
});

(async function init(){
  const r = await (await fetch("/api/loadconfig", {cache:"no-store"})).json();
  window.__cfg = r.config;
  await loadConfigForm();
})();
drawSpark(); poll(); loadPrefs(); setInterval(poll, 1000); setInterval(loadConfigForm, 20000);

















/* ---- i18n init ---- */
(function () {
  try {
    const b = $("langBtn");
    if (b) b.addEventListener("click", () => setLang(currentLang() === "en" ? "zh" : "en"));
    const q = new URLSearchParams(location.search).get("lang");   // ?lang=en / ?lang=zh
    setLang(q || currentLang());
    new MutationObserver(muts => {
      if (currentLang() !== "en") return;
      for (const m of muts) {
        if (m.type === "characterData") i18nTextNode(m.target, "en");
        else m.addedNodes.forEach(n => {
          if (n.nodeType === 3) i18nTextNode(n, "en");
          else if (n.nodeType === 1) applyI18n("en", n);
        });
      }
    }).observe(document.body, { subtree: true, childList: true, characterData: true });
  } catch (e) {
    document.title = "i18n-error: " + (e && e.message ? e.message : e);
    if (window.console) console.error("[i18n]", e);
  }
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------- 一键接入 CC Switch / Codex
CCSWITCH_DB   = os.path.join(os.path.expanduser("~"), ".cc-switch", "cc-switch.db")
CCSWITCH_EXE  = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "CC Switch", "cc-switch.exe")
CODEX_DIR     = os.path.join(os.path.expanduser("~"), ".codex")
CODEX_CONFIG  = os.path.join(CODEX_DIR, "config.toml")
CODEX_CATALOG = os.path.join(CODEX_DIR, "bonsai-model-catalog.json")
UPSERT_SCRIPT = os.path.join(WORK_DIR, "ccswitch_upsert.py")
CATALOG_SCRIPT = os.path.join(WORK_DIR, "make-bonsai-catalog.py")
ATTACH_BACKUP = os.path.join(WORK_DIR, "attach-backup")


# --- 本地模型的 Codex 精简工具面：只留 exec_command / apply_patch 等核心工具 -----------
# 27B 小模型在“全家桶”工具面里会误选 read_mcp_resource/write_stdin 并陷入循环；
# 关掉插件、桌面注入的 MCP 服务（cua_repl/node_repl）、子代理与连接器即可根治。
LEAN_PLUGIN_IDS = (
    "browser@openai-bundled",
    "visualize@openai-bundled",
    "documents@openai-primary-runtime",
    "pdf@openai-primary-runtime",
    "spreadsheets@openai-primary-runtime",
    "presentations@openai-primary-runtime",
    "template-creator@openai-primary-runtime",
    "codex-app-tools@openai-bundled",
    "unified-computer-use@openai-bundled",
    "chrome@openai-bundled",
    "computer-use@openai-bundled",
)
CUA_RUNTIMES = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                            "OpenAI", "Codex", "runtimes", "cua_node")


def _newest_runtime(*parts):
    try:
        best, best_mtime = None, -1.0
        for name in os.listdir(CUA_RUNTIMES):
            cand = os.path.join(CUA_RUNTIMES, name, *parts)
            if os.path.isfile(cand):
                mtime = os.path.getmtime(cand)
                if mtime > best_mtime:
                    best, best_mtime = cand, mtime
        if best:
            return best
    except OSError:
        pass
    return parts[-1]


def _section_bounds(lines, header):
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == header:
            start = i
            break
    if start is None:
        return None, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("[") and s.endswith("]"):
            end = j
            break
    return start, end


def _ensure_section_bool(text, header, key, value):
    """按需写入 [section] key = value；段落不存在时自动补建。"""
    text = text.rstrip("\n") + "\n"
    lines = text.splitlines()
    start, end = _section_bounds(lines, header)
    if start is None:
        return text + "\n%s\n%s = %s\n" % (header, key, value)
    target = "%s = %s" % (key, value)
    for i in range(start + 1, end):
        if re.match(r"^\s*%s\s*=" % re.escape(key), lines[i]):
            lines[i] = target
            return "\n".join(lines) + "\n"
    lines.insert(start + 1, target)
    return "\n".join(lines) + "\n"


def _ensure_mcp_disabled(text, name, command, args):
    """把 cua_repl / node_repl 关掉；段落缺失时补一份带命令的完整定义。"""
    header = "[mcp_servers.%s]" % name
    args_toml = "[%s]" % ", ".join("'%s'" % a for a in args)
    text = text.rstrip("\n") + "\n"
    lines = text.splitlines()
    start, end = _section_bounds(lines, header)
    if start is None:
        return text + ("\n%s\nenabled = false\ncommand = '%s'\nargs = %s\n"
                       "startup_timeout_sec = 120\n" % (header, command, args_toml))
    body = lines[start + 1:end]

    def upsert(k, v):
        for i, ln in enumerate(body):
            if re.match(r"^\s*%s\s*=" % re.escape(k), ln):
                body[i] = "%s = %s" % (k, v)
                return
        body.append("%s = %s" % (k, v))

    upsert("enabled", "false")
    if not any(re.match(r"^\s*command\s*=", ln) for ln in body):
        upsert("command", "'%s'" % command)
    if not any(re.match(r"^\s*args\s*=", ln) for ln in body):
        upsert("args", args_toml)
    lines[start + 1:end] = body
    return "\n".join(lines) + "\n"


def _lean_tool_profile(text):
    """本地 27B 只接 Codex：砍掉插件、MCP 服务与子代理工具，减少误调用。"""
    for plugin_id in LEAN_PLUGIN_IDS:
        text = _ensure_section_bool(text, '[plugins."%s"]' % plugin_id, "enabled", "false")
    text = _ensure_section_bool(text, "[agents]", "enabled", "false")
    text = _ensure_section_bool(text, "[apps._default]", "enabled", "false")
    text = _ensure_section_bool(text, "[features]", "js_repl", "false")
    text = _ensure_section_bool(text, "[features]", "goals", "false")
    text = _ensure_section_bool(text, "[features]", "multi_agent", "false")
    text = _ensure_section_bool(text, "[features]", "apps", "false")
    text = _ensure_section_bool(text, "[features]", "view_image", "false")
    text = _ensure_mcp_disabled(
        text, "cua_repl", _newest_runtime("bin", "node.exe"),
        [_newest_runtime("bin", "node_modules", "@oai", "cua-repl", "bin", "cua-repl.mjs")])
    text = _ensure_mcp_disabled(
        text, "node_repl", _newest_runtime("bin", "node_repl.exe"), [])
    return text


def _run(cmd, timeout=180):
    """Run a command without popping a console window; return (rc, output)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as exc:
        return -1, str(exc)


def ccswitch_pids():
    rc, out = _run(["tasklist", "/FI", "IMAGENAME eq cc-switch.exe", "/FO", "CSV", "/NH"], timeout=30)
    pids = []
    if rc == 0:
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2 and parts[0].lower().startswith("cc-switch"):
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    pass
    return pids


def start_ccswitch():
    if not os.path.exists(CCSWITCH_EXE):
        return False
    try:
        subprocess.Popen([CCSWITCH_EXE], close_fds=True,
                         creationflags=0x00000008 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except Exception:
        return False


def catalog_levels():
    try:
        with open(CODEX_CATALOG, "r", encoding="utf-8") as fh:
            cat = json.load(fh)
        levels = [lvl.get("effort") for lvl in (cat["models"][0].get("supported_reasoning_levels") or [])]
        slug = cat["models"][0].get("slug")
        return [x for x in levels if x], slug
    except Exception:
        return [], None


def refresh_codex_config(ctx, compact=None):
    """Refresh the local-provider settings in ~/.codex/config.toml, but only if it already targets us."""
    if not os.path.exists(CODEX_CONFIG):
        return False, "找不到 ~/.codex/config.toml（未安装 Codex CLI/App？）"
    with open(CODEX_CONFIG, "r", encoding="utf-8") as fh:
        text = fh.read()
    if "127.0.0.1:%d" % SERVER_PORT not in text:
        return False, ("Codex 现在指向别的供应商（不是本地）—— 在 CC Switch 里点一下「Bonsai 2 27B（本地 V100）」"
                       "即可切换；本次没有改动它的 config.toml")
    new = text
    wanted = {
        "model_context_window": str(int(ctx)),
        "model_reasoning_effort": '"none"',
        "model_catalog_json": '"bonsai-model-catalog.json"',
    }
    if compact:
        wanted["model_auto_compact_token_limit"] = str(int(compact))
    for key, val in wanted.items():
        line = "%s = %s" % (key, val)
        if re.search(r"(?m)^%s\s*=" % re.escape(key), new):
            new = re.sub(r"(?m)^%s\s*=.*$" % re.escape(key), line, new)
        elif re.search(r"(?m)^model\s*=", new):
            new = re.sub(r"(?m)^(model\s*=.*)$", r"\1\n" + line, new, count=1)
        else:
            new = line + "\n" + new
    new = _lean_tool_profile(new)
    if new == text:
        return True, "config.toml 已是最新（无需改动）"
    os.makedirs(ATTACH_BACKUP, exist_ok=True)
    shutil.copy2(CODEX_CONFIG, os.path.join(
        ATTACH_BACKUP, "config-%s.toml" % time.strftime("%Y%m%d-%H%M%S")))
    with open(CODEX_CONFIG, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    extra = ("/ 自动压缩 %d " % int(compact)) if compact else ""
    return True, ("已刷新：上下文 %d %s/ 模型目录 / 默认思考档 / 精简工具面（插件与 MCP 已关；"
                  "原文件备份在 work\\attach-backup）"
                  % (int(ctx), extra))


def sync_codex_side(cfg):
    """After a reload: regenerate the Codex model catalog, and refresh config.toml when it targets us.

    The catalog carries context_window / max_context_window / auto-compact, so any context change on
    the panel must be pushed there too -- otherwise Codex keeps the old window and compaction point.
    """
    notes = []
    if os.path.exists(CATALOG_SCRIPT):
        rc, _out = _run([sys.executable, CATALOG_SCRIPT], timeout=180)
        notes.append("模型目录已同步" if rc == 0 else "模型目录同步失败（见 work\\panel.err.log）")
    try:
        ok_cfg, _detail = refresh_codex_config(cfg.get("ctx"), cfg.get("compact_tokens"))
        if ok_cfg and "已是最新" not in _detail:
            notes.append("Codex config.toml 已同步")
    except Exception as exc:
        notes.append("Codex config 同步出错：%s" % exc)
    # CC Switch only gets the new context when its store is updated; do it if the app is closed,
    # otherwise leave it to the "一键接入" button (writing while it runs can be overwritten).
    if os.path.exists(UPSERT_SCRIPT):
        running = any(p for p in ccswitch_pids())
        if running:
            notes.append("CC Switch 在运行，供应商条目未动（需要时点「一键接入」）")
        else:
            rc, _out = _run([sys.executable, UPSERT_SCRIPT], timeout=180)
            notes.append("CC Switch 条目已同步" if rc == 0 else "CC Switch 条目同步失败")
    return " ｜ ".join(notes)


def do_attach():
    """One-click: refresh the CC Switch provider entries, the Codex catalog and the Codex config."""
    steps = []
    cfg = load_config()
    ctx = int(cfg.get("ctx") or 32768)

    online = is_online()
    steps.append({"name": "本地服务",
                  "ok": online,
                  "detail": ("llama-server 在线：%s（模型 %s）" % (UPSTREAM, os.path.basename(cfg.get("model", ""))))
                            if online else "服务没在跑 —— 回面板点「启动服务」或「应用并重载」再接入"})

    levels, slug = catalog_levels()
    if not levels and os.path.exists(CATALOG_SCRIPT):
        _run([sys.executable, CATALOG_SCRIPT], timeout=120)
        levels, slug = catalog_levels()
    steps.append({"name": "Codex 模型目录",
                  "ok": bool(levels),
                  "detail": ("%s → 档位 %s" % (slug, " / ".join(levels))) if levels
                            else "缺少 ~/.codex/bonsai-model-catalog.json（生成脚本没跑成功）"})

    target = "http://127.0.0.1:%d/v1" % SERVER_PORT
    if not os.path.exists(CCSWITCH_DB):
        steps.append({"name": "CC Switch 供应商", "ok": False,
                      "detail": "找不到 %s" % CCSWITCH_DB})
    elif not os.path.exists(UPSERT_SCRIPT):
        steps.append({"name": "CC Switch 供应商", "ok": False,
                      "detail": "找不到 %s" % UPSERT_SCRIPT})
    else:
        pids = ccswitch_pids()
        for pid in pids:
            _kill_pid(pid)
        if pids:
            time.sleep(2.0)
        rc, out = _run([sys.executable, UPSERT_SCRIPT], timeout=180)
        back = [ln for ln in out.splitlines() if ln.startswith("backup:")]
        restarted = start_ccswitch() if pids else False
        detail = "已写入/刷新 Codex 与 Claude 两条供应商（BASE_URL %s）" % target
        if back:
            detail += "；数据库备份 " + os.path.basename(back[0].split("backup:", 1)[1].strip())
        if pids:
            detail += "；CC Switch 已重启" if restarted else "；CC Switch 重启失败，请手动打开"
        else:
            detail += "；CC Switch 当时没在运行"
        steps.append({"name": "CC Switch 供应商", "ok": rc == 0, "detail": detail if rc == 0 else out[-400:]})

    ok_cfg, cfg_detail = refresh_codex_config(ctx, cfg.get("compact_tokens"))
    steps.append({"name": "Codex 当前供应商", "ok": ok_cfg, "detail": cfg_detail})

    ready = is_online() and bool(levels)
    hint = ("已接入。在 CC Switch 里点一下「Bonsai 2 27B（本地 V100）」，再在 Codex 里新开一个会话，"
            "就能用 /reasoning 切「关 / low / medium / xhigh」。")
    if not online:
        hint = "先把本地服务跑起来，再点一次这个按钮。"
    return {"ok": ready, "steps": steps, "hint": hint}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?", 1)[0]          # ignore ?lang=… and other query strings
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps(status_payload(), ensure_ascii=False))
        elif path == "/api/prefs":
            self._send(200, json.dumps(load_prefs(), ensure_ascii=False))
        elif path == "/api/loadconfig":
            models, mmprojs = list_files()
            self._send(200, json.dumps({
                "config": load_config(), "models": models, "mmprojs": mmprojs,
                "derived": derived_info(load_config()),
                "limits": limits_payload(load_config()),
            }, ensure_ascii=False))
        elif self.path == "/api/bench":
            self._send(200, json.dumps({
                "bench": dict(_bench), "history": bench_history(), "vram": vram_breakdown(load_config()),
            }, ensure_ascii=False))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as exc:
            self._send(400, json.dumps({"error": f"bad request: {exc}"}))
            return

        if self.path == "/api/prefs":
            upd = {}
            if isinstance(payload.get("protocol"), bool):
                upd["protocol"] = payload["protocol"]
            prefs = save_prefs(upd)
            self._send(200, json.dumps({"ok": True, "prefs": prefs}, ensure_ascii=False))
            return

        if self.path == "/api/loadconfig":
            action = payload.get("action", "apply")
            cfg = dict(load_config())
            cfg.update(payload.get("config") or {})
            cfg["ctx"] = int(cfg.get("ctx") or 8192)
            cfg["ngl"] = int(cfg.get("ngl") or 0)
            raw_par = str(cfg.get("parallel", "auto")).strip().lower()
            cfg["parallel"] = "auto" if raw_par in ("", "auto", "0", "none") else int(float(raw_par))
            if str(cfg.get("cache_type_k", "f16")) != "f16" or str(cfg.get("cache_type_v", "f16")) != "f16":
                if str(cfg.get("flash_attn")) == "off":
                    cfg["flash_attn"] = "on"     # 量化 KV 必须开 FA

            # ---- 按硬上限校验/收敛参数 ----
            lim = limits_payload(cfg)
            notes = []

            def clamp_int(key, lo, hi, label):
                try:
                    val = int(float(cfg.get(key)))
                except Exception:
                    return
                new = min(max(val, lo), hi)
                if new != val:
                    notes.append(f"{label} {val} → {new}（范围 {lo}–{hi}）")
                cfg[key] = new

            clamp_int("ctx", 512, int(lim["ctx_max"]), "上下文长度")
            clamp_int("ngl", 0, int(lim["ngl_max"]), "GPU 层数")
            clamp_int("batch", 64, int(lim["batch_max"]), "batch")
            clamp_int("ubatch", 32, int(lim["ubatch_max"]), "ubatch")
            clamp_int("top_k", 0, int(lim["top_k_max"]), "top_k")
            if int(cfg.get("ubatch", 512)) > int(cfg.get("batch", 2048)):
                cfg["ubatch"] = int(cfg["batch"])
                notes.append("ubatch 不能大于 batch，已下调")
            for key, label in (("temp", "温度"), ("top_p", "top_p"), ("min_p", "min_p"),
                               ("presence_penalty", "presence_penalty")):
                try:
                    val = float(cfg.get(key))
                except Exception:
                    continue
                if key == "temp":
                    hi = float(lim["temp_max"])
                elif key == "presence_penalty":
                    hi = 2.0
                else:
                    hi = 1.0
                new = min(max(val, 0.0), hi)
                if abs(new - val) > 1e-9:
                    notes.append(f"{label} {val} → {new}（范围 0–{hi}）")
                cfg[key] = new
            if str(cfg.get("threads", "")).strip():
                clamp_int("threads", 1, int(lim["threads_max"]), "CPU 线程")
            par = str(cfg.get("parallel", "auto")).strip().lower()
            if par not in ("", "auto", "0", "none"):
                clamp_int("parallel", 1, int(lim["parallel_max"]), "并发槽位")
            save_config(cfg)

            if action == "stop":
                killed = stop_server()
                _status_cache["t"] = 0
                self._send(200, json.dumps({
                    "ok": True, "message": f"已停止服务（结束进程 {killed or '无'}）"}, ensure_ascii=False))
                return
            if action == "shutdown":
                killed = stop_server()
                _status_cache["t"] = 0
                self._send(200, json.dumps({
                    "ok": True,
                    "message": f"已卸载模型（结束进程 {killed or '无'}）并关闭面板"},
                    ensure_ascii=False))
                threading.Timer(1.0, lambda: os._exit(0)).start()
                return
            if action == "start" and is_online():
                self._send(200, json.dumps({"ok": True, "message": "服务已在运行"}, ensure_ascii=False))
                return

            ok, msg = apply_config(cfg)
            _status_cache["t"] = 0
            if ok:
                note = sync_codex_side(cfg)
                if note:
                    msg = "%s ｜ %s" % (msg, note)
            self._send(200, json.dumps({
                "ok": ok, "message": msg, "log_tail": [] if ok else log_tail(15),
            }, ensure_ascii=False))
            return

        if self.path == "/api/attach":
            self._send(200, json.dumps(do_attach(), ensure_ascii=False))
            return

        if self.path == "/api/bench":
            action = payload.get("action", "run")
            if action == "run":
                ok, msg = start_bench(payload.get("reps", 3), payload.get("compare", False))
                self._send(200, json.dumps({"ok": ok, "message": msg}, ensure_ascii=False))
            else:
                self._send(400, json.dumps({"error": f"unknown action: {action}"}, ensure_ascii=False))
            return

        if self.path == "/api/chat":
            # 面板对话默认：注入「推理外显协议」+ 关思考（前端复选框可关掉）
            if payload.pop("protocol", True):
                text = protocol_text()
                msgs = payload.get("messages") or []
                if text and not any(m.get("role") == "system" for m in msgs):
                    msgs.insert(0, {"role": "system", "content": text})
                    payload["messages"] = msgs
                payload.setdefault("reasoning_effort", "none")
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
            req = urllib.request.Request(
                f"{UPSTREAM}/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                upstream = urllib.request.urlopen(req, timeout=1800)
            except urllib.error.HTTPError as exc:
                self._send(502, json.dumps({"error": f"上游 {exc.code}: {exc.read().decode('utf-8','replace')[:400]}"}, ensure_ascii=False))
                return
            except Exception as exc:
                self._send(502, json.dumps({"error": f"无法连接 {UPSTREAM}：{exc}"}, ensure_ascii=False))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            panel_inflight(+1)
            try:
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                panel_inflight(-1)
                upstream.close()
            return

        self._send(404, json.dumps({"error": "not found"}))


def main():
    global SERVER_PORT, UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=8080)
    ap.add_argument("--demo-dir", default=DEMO_DIR,
                    help="包含 bin/cuda/llama-server.exe、models/ 与 dashboard-config.json 的目录"
                         "（也可用环境变量 BONSAI_DEMO_DIR）")
    args = ap.parse_args()
    SERVER_PORT = args.server_port
    UPSTREAM = f"http://127.0.0.1:{SERVER_PORT}"
    if not os.path.exists(SERVER_EXE):
        print(f"警告：找不到 {SERVER_EXE}；请用 --demo-dir 指向正确的 bonsai-demo 目录")
    seed_bench_from_history()
    threading.Thread(target=live_sampler, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Bonsai 控制台： http://{args.host}:{args.port}    （llama-server: {UPSTREAM}）")
    srv.serve_forever()


if __name__ == "__main__":
    main()
