"""Run every pending single-seed job on whichever GPU is free, local first, then the server.

A job is done when its result JSON exists (server results are pulled back).
When nothing is running and no GPU has room, it waits 20 minutes before
looking again; otherwise it checks every 2 minutes. Log: D:/cate_data/logs/dispatch.log
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
LOG = Path("D:/cate_data/logs/dispatch.log")
REMOTE = os.environ.get("REMOTE_PROJECT_DIR", "/opt/MemoryEmotionAgent")
PY_REMOTE = os.environ.get("REMOTE_PYTHON", "python")
ENV_REMOTE = ("CATE_DATA=$PWD/data HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 "
              "TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1")
JOBS_FILE = ROOT / "scripts" / "jobs_meta.json"


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def ssh():
    for _ in range(3):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(os.environ["REMOTE_HOST"], port=int(os.environ.get("REMOTE_PORT", "22")), username=os.environ["REMOTE_USER"], password=os.environ.get("REMOTE_PASSWORD"), timeout=30, banner_timeout=30, auth_timeout=30)
            c.get_transport().set_keepalive(30)
            return c
        except Exception as e:  # the tunnel drops now and then
            log(f"ssh retry: {type(e).__name__}")
            time.sleep(20)
    return None


def run(c, cmd, timeout=120):
    _, o, _ = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace")


def free_gpus_local():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    return {int(a): int(t) - int(u) for a, u, t in (l.split(", ") for l in out.strip().splitlines())}


def free_gpus_remote(c):
    out = run(c, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits")
    return {int(a): int(t) - int(u) for a, u, t in (l.split(", ") for l in out.strip().splitlines())}


def main():
    state = {}
    reserved = {}  # (host, gpu) -> (MiB, until)
    log("dispatcher start")
    while True:
        # Re-read every cycle so jobs can be added or flags changed without a restart.
        jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        for j in jobs:
            state.setdefault(j["name"], {"where": None, "tries": 0, "proc": None, "since": 0.0})
        launched = False
        c = ssh()
        remote_done = set()
        if c is not None:
            listing = run(c, f"cd {REMOTE} && ls results/policy/*/*.json 2>/dev/null; echo; ps -eo args | grep 'job=' | grep -v grep")
            remote_files, _, remote_ps = listing.partition("\n\n")
            remote_done = {l.strip() for l in remote_files.splitlines() if l.strip()}
        pending = 0
        for j in jobs:
            st = state[j["name"]]
            local_out = ROOT / j["out"]
            if local_out.exists():
                if st["where"] is not None:
                    log(f"done {j['name']} on {st['where']}")
                    st["where"] = None
                continue
            if c is not None and j["out"] in remote_done:
                try:
                    sftp = c.open_sftp()
                    local_out.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(f"{REMOTE}/{j['out']}", str(local_out) + ".part")
                    sftp.close()
                    os.replace(str(local_out) + ".part", str(local_out))
                    log(f"pulled {j['name']}")
                    st["where"] = None
                except Exception as e:
                    log(f"pull failed {j['name']}: {type(e).__name__}")
                continue
            pending += 1
            if st["where"] and st["where"].startswith("local"):
                if st["proc"].poll() is None:
                    continue
                log(f"local job ended without result: {j['name']} (try {st['tries']})")
                st["where"] = None
            elif st["where"] and st["where"].startswith("server"):
                alive = any(l.strip().endswith(f"job={j['name']}") for l in remote_ps.splitlines())
                if c is None or alive or time.time() - st["since"] < 180:
                    continue
                log(f"server job ended without result: {j['name']} (try {st['tries']})")
                st["where"] = None
            if st["tries"] >= 3:
                continue
            now = time.time()
            res = lambda host, g: sum(m for (h, gg), (m, u) in reserved.items() if h == host and gg == g and u > now)
            target = None
            try:
                for g, free in sorted(free_gpus_local().items(), key=lambda kv: -kv[1]):
                    if free - res("local", g) >= j["mem"]:
                        target = ("local", g)
                        break
            except Exception as e:
                log(f"local nvidia-smi failed: {e}")
            if target is None and c is not None and not j.get("local_only"):
                try:
                    for g, free in sorted(free_gpus_remote(c).items(), key=lambda kv: -kv[1]):
                        if free - res("server", g) >= j["mem"]:
                            target = ("server", g)
                            break
                except Exception as e:
                    log(f"remote nvidia-smi failed: {type(e).__name__}")
            if target is None:
                continue
            host, g = target
            st["tries"] += 1
            st["since"] = now
            reserved[(host, g)] = (j["mem"], now + 300)
            if host == "local":
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HOME="D:\\hf_cache")
                lf = open(f"D:/cate_data/logs/job_{j['name']}.log", "a", encoding="utf-8")
                st["proc"] = subprocess.Popen([sys.executable, "-m", *j["cmd"].split()], cwd=str(ROOT), env=env,
                                              stdout=lf, stderr=subprocess.STDOUT)
            else:
                try:
                    run(c, f"cd {REMOTE} && mkdir -p logs && (CUDA_VISIBLE_DEVICES={g} {ENV_REMOTE} setsid nohup "
                       f"{PY_REMOTE} -m {j['cmd']} --job_tag job={j['name']} > logs/job_{j['name']}.log 2>&1 &)")
                except Exception as e:
                    log(f"remote launch failed {j['name']}: {type(e).__name__}")
                    st["tries"] -= 1
                    continue
            st["where"] = f"{host}:{g}"
            launched = True
            log(f"launch {j['name']} on {host}:{g} (try {st['tries']})")
        if c is not None:
            c.close()
        if pending == 0:
            log("all jobs done")
            return
        running = any(st["where"] for st in state.values())
        time.sleep(120 if (launched or running) else 1200)


if __name__ == "__main__":
    main()
