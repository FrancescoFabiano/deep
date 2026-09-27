"""Run the planner: argv, time and memory limits, output parsing."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from .config import Config
from .instances import Instance


@dataclass
class Result:
    rc: int
    out: str
    wall_s: float
    limit: str | None        # None, "TIMEOUT" or "MEMOUT"

    def field(self, label: str) -> int | None:
        """The integer after `label ...:` in the planner's -r output."""
        m = re.search(rf"{re.escape(label)}[^:\n]*:\s*(\d+)", self.out)
        return int(m.group(1)) if m else None


def base_argv(cfg: Config, inst: Instance) -> list[str]:
    # absolute: generation runs in a private work dir, not the repo root
    return [str(cfg.deep_exe), str(inst.domain_file.resolve()), str(inst.problem_file.resolve()),
            "--act_lib", str(cfg.act_lib.resolve()), "-b", "-c"]


def run(argv: list[str], *, timeout_s: float, mem_gb: float, cwd=None) -> Result:
    start = time.time()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=cwd, start_new_session=True)
    limit: list[str] = []

    def kill(reason: str) -> None:
        limit.append(reason)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def watch() -> None:
        while proc.poll() is None:
            if _rss_gb(proc.pid) > mem_gb:
                kill("MEMOUT")
                return
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        kill("TIMEOUT")
        out, _ = proc.communicate()
    return Result(proc.returncode, out or "", time.time() - start, limit[0] if limit else None)


def _rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0
