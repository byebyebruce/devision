"""What training reports to SwanLab, shared by stage 1 (align) and stage 2 (rlcd).

One row every `log_every` steps, grouped by prefix so SwanLab draws one panel group each:
  train/  window means of the per-step metrics (single batches are too noisy to read)
  lr/     learning rate per parameter group
  perf/   step time, samples/s, progress and estimated time left
  sys/    CPU, RAM, swap and accelerator load (MPS: allocated incl. cache, and live tensors), sampled at log time
Evaluations go in their own rows (val/, pope/, ...) from the training loops.
"""
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch


RECORD = "swanlab_run.json"   # in a stage's output directory: which SwanLab run trained it


class Tracker:
    """SwanLab logging when `project` is set, otherwise a no-op. With `record_dir`, the run's project, id
    and last step are written to <record_dir>/swanlab_run.json, so evaluations can be added to it later."""

    def __init__(self, project: str, run_name: str, config: Dict[str, Any], record_dir: str = "",
                 resume: bool = False):
        """`resume`: continue the run recorded in record_dir (training picked up from a resume state)."""
        self.swanlab = None
        self.record: Dict[str, Any] = {}
        self.record_path = os.path.join(record_dir, RECORD) if record_dir else ""
        if project:
            import swanlab

            earlier = None
            if resume and self.record_path and os.path.exists(self.record_path):
                with open(self.record_path) as f:
                    earlier = json.load(f)
            if earlier and earlier.get("id"):
                run = swanlab.init(project=project, id=earlier["id"], resume="allow")
            else:
                run = swanlab.init(project=project, name=run_name or None, config=config)
            self.swanlab = swanlab
            self.record = {"project": project, "id": getattr(run, "id", None), "name": run_name,
                           "url": getattr(run, "url", None), "last_step": (earlier or {}).get("last_step", 0)}
            self._save()

    def _save(self) -> None:
        if self.record_path and self.record.get("id"):
            os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
            with open(self.record_path, "w") as f:
                json.dump(self.record, f, indent=1)

    def log(self, data: Dict[str, float], step: int) -> None:
        if self.swanlab:
            self.swanlab.log(data, step=step)
            self.record["last_step"] = max(step, self.record.get("last_step", 0))

    def finish(self) -> None:
        if self.swanlab:
            self._save()
            self.swanlab.finish()


def log_to_finished_run(record_path: str, data: Dict[str, float]) -> str:
    """Add `data` to the SwanLab run recorded in `record_path` (resumed by id), at its last step.
    Used for evaluation summaries, which belong to the run that trained the checkpoint."""
    import swanlab

    with open(record_path) as f:
        rec = json.load(f)
    swanlab.init(project=rec["project"], id=rec["id"], resume="must")
    swanlab.log(data, step=int(rec.get("last_step") or 0))
    swanlab.finish()
    return rec.get("url") or rec["id"]


_IOREG = {"Device Utilization %": "sys/gpu_util_pct", "In use system memory": "sys/gpu_mem_gb"}
GB = 1024 ** 3


def _apple_gpu() -> Dict[str, float]:
    """Apple GPU utilisation and memory in use, from `ioreg` (no sudo); {} when unavailable."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True,
                             text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    stats = {}
    for field, key in _IOREG.items():
        m = re.search(r'"%s"=(\d+)' % re.escape(field), out)
        if m:
            stats[key] = int(m.group(1)) / (GB if key.endswith("_gb") else 1)
    return stats


def system_stats(device: torch.device) -> Dict[str, float]:
    """Machine load right now: CPU (whole machine and this process), RAM, swap, accelerator."""
    import psutil

    proc = psutil.Process(os.getpid())
    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    stats = {"sys/cpu_pct": psutil.cpu_percent(interval=None),
             "sys/proc_cpu_pct": proc.cpu_percent(interval=None),
             "sys/ram_used_pct": vm.percent,
             "sys/ram_available_gb": vm.available / GB,
             "sys/swap_used_gb": swap.used / GB,
             "sys/proc_rss_gb": proc.memory_info().rss / GB}
    if device.type == "mps":
        stats["sys/torch_mps_gb"] = torch.mps.driver_allocated_memory() / GB        # incl. the allocator's cache
        stats["sys/torch_mps_live_gb"] = torch.mps.current_allocated_memory() / GB  # tensors actually alive
        stats.update(_apple_gpu())
    elif device.type == "cuda":
        stats["sys/gpu_mem_gb"] = torch.cuda.memory_allocated(device) / GB
        stats["sys/gpu_max_mem_gb"] = torch.cuda.max_memory_allocated(device) / GB
    return stats


class TrainLog:
    """Collects per-step metrics and writes one averaged row every `every` steps."""

    def __init__(self, tracker: Tracker, device: torch.device, total_steps: int, every: int):
        self.tracker, self.device, self.total, self.every = tracker, device, max(1, total_steps), max(1, every)
        self.window: Dict[str, List[float]] = defaultdict(list)
        self.samples = 0
        self.t_window = self.t_start = time.perf_counter()
        system_stats(device)  # first cpu_percent call only sets the baseline

    def add(self, samples: int, **metrics: float) -> None:
        self.samples += samples
        for k, v in metrics.items():
            self.window[k].append(float(v))

    def flush(self, step: int, epoch: int, lrs: Dict[str, float]) -> Dict[str, float]:
        """At every `every`-th step: log and return the window means (train/<metric>); else {}."""
        if step % self.every:
            return {}
        now = time.perf_counter()
        n = len(next(iter(self.window.values()), [])) or 1
        means = {"train/" + k: sum(v) / len(v) for k, v in self.window.items() if v}
        if self.window.get("batch_tokens"):   # the longest batch of the window, next to the mean
            means["train/batch_tokens_max"] = max(self.window["batch_tokens"])
        elapsed = now - self.t_start
        row = dict(means, epoch=epoch,
                   **{"lr/" + k: v for k, v in lrs.items()},
                   **{"perf/step_s": (now - self.t_window) / n,
                      "perf/samples_per_s": self.samples / max(1e-9, now - self.t_window),
                      "perf/progress_pct": 100 * step / self.total,
                      "perf/eta_h": elapsed / step * (self.total - step) / 3600},
                   **system_stats(self.device))
        self.tracker.log(row, step=step)
        self.window.clear()
        self.samples = 0
        self.t_window = now
        return means
