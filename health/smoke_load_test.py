#!/usr/bin/env python3
"""
Smoke load test: ramp up concurrency 1→2→4→8→10 models,
sampling system resources every second throughout.
Log to /tmp/smoke_loadtest.log — safe to Ctrl-C.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from health.smoke import score_model_hermes  # noqa: E402

LOG = Path("/tmp/smoke_loadtest.log")
REGISTRY = Path(__file__).parent.parent / "registry.json"
BENCHMARK = Path(__file__).parent.parent / "benchmark" / "humaneval_10.json"
PROBLEMS_PER_MODEL = 3   # keep runs short; real test uses 10


# ── resource sampler ──────────────────────────────────────────────────────────

stop_monitor = threading.Event()


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except Exception:
        return ""


def monitor_loop(log_path: Path) -> None:
    systemd_pid = next(
        (int(l.split()[1]) for l in _read("/proc/1/status").splitlines()
         if l.startswith("Pid:")), 1
    )
    with log_path.open("w") as f:
        f.write("t\tconcur\thermes_procs\tmem_avail_mb\tswap_used_mb\t"
                "load1\tsd_fd\tsd_rss_mb\tsd_swap_mb\n")
        f.flush()
        while not stop_monitor.is_set():
            t = time.strftime("%H:%M:%S")
            # memory
            mem = {k: int(v.split()[0])
                   for line in _read("/proc/meminfo").splitlines()
                   if ":" in line
                   for k, v in [line.split(":", 1)]}
            avail = mem.get("MemAvailable", 0) // 1024
            swap  = mem.get("SwapTotal", 0) // 1024 - mem.get("SwapFree", 0) // 1024

            # load
            load1 = _read("/proc/loadavg").split()[0]

            # hermes subprocess count
            try:
                hermes_n = int(subprocess.check_output(
                    ["pgrep", "-c", "-f", "hermes chat"], stderr=subprocess.DEVNULL
                ).strip())
            except Exception:
                hermes_n = 0

            # systemd stats
            sd_status = _read(f"/proc/{systemd_pid}/status")
            sd_fd  = next((int(l.split()[1])
                           for l in sd_status.splitlines()
                           if l.startswith("FDSize:")), 0)
            sd_rss = next((int(l.split()[1]) // 1024
                           for l in sd_status.splitlines()
                           if l.startswith("VmRSS:")), 0)
            sd_sw  = next((int(l.split()[1]) // 1024
                           for l in sd_status.splitlines()
                           if l.startswith("VmSwap:")), 0)

            concur = current_concurrency[0]
            line = (f"{t}\t{concur}\t{hermes_n}\t{avail}\t{swap}\t"
                    f"{load1}\t{sd_fd}\t{sd_rss}\t{sd_sw}\n")
            f.write(line)
            f.flush()
            print(f"  [{t}] concur={concur:2d} hermes={hermes_n:2d} "
                  f"avail={avail:5d}MB swap={swap:4d}MB load={load1} "
                  f"sd_fd={sd_fd} sd_rss={sd_rss}MB",
                  flush=True)
            time.sleep(1.0)


current_concurrency: list[int] = [0]


# ── load test runner ──────────────────────────────────────────────────────────

async def run_n_models(models: list[dict], problems: list[dict], n: int) -> None:
    subset = models[:n]
    current_concurrency[0] = n
    tasks = [score_model_hermes(m, problems) for m in subset]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for m, r in zip(subset, results):
        if isinstance(r, Exception):
            print(f"    {m['id']}: ERROR {r}")
        else:
            print(f"    {m['id']}: score={r[1]:.0%}")


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    problems = json.loads(BENCHMARK.read_text())[:PROBLEMS_PER_MODEL]

    models = [m for m in registry["models"]
              if m.get("hermes_model") and m.get("alive") is not False
              and m.get("tier", 3) <= 2]

    print(f"Available models: {len(models)}, problems per model: {len(problems)}")
    print(f"Log: {LOG}\n")

    # start monitor thread
    mon = threading.Thread(target=monitor_loop, args=(LOG,), daemon=True)
    mon.start()

    steps = [1, 2, 4, 6, 8, 10]
    steps = [s for s in steps if s <= len(models)]

    for n in steps:
        print(f"\n── concurrency={n} ──────────────────────────────")
        t0 = time.time()
        try:
            asyncio.run(run_n_models(models, problems, n))
        except KeyboardInterrupt:
            break
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")
        time.sleep(3)   # let processes drain before next step

    stop_monitor.set()
    current_concurrency[0] = 0
    print(f"\nDone. Full log: {LOG}")

    # summary: peak values
    data = LOG.read_text().splitlines()[1:]   # skip header
    if data:
        min_avail = min(int(r.split("\t")[3]) for r in data if r)
        max_hermes = max(int(r.split("\t")[2]) for r in data if r)
        max_swap = max(int(r.split("\t")[4]) for r in data if r)
        print(f"\nPeak summary:")
        print(f"  min available RAM : {min_avail} MB")
        print(f"  max hermes procs  : {max_hermes}")
        print(f"  max swap used     : {max_swap} MB")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (stop_monitor.set(), sys.exit(0)))
    main()
