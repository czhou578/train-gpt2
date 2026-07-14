# benchmark.py
"""
Lightweight metrics / benchmarking harness for nanoGPT-style training.
Every experiment gets its own self-contained directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from typing import Any, Dict, Optional

import torch


class MetricsLogger:
    def __init__(self, run_dir: str, window: int = 100):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.config_path = os.path.join(run_dir, "config.json")
        self.summary_path = os.path.join(run_dir, "summary.json")

        # Rolling window for smoothed throughput
        self._tok_s_window: deque[float] = deque(maxlen=window)
        self._samples_s_window: deque[float] = deque(maxlen=window)

        # NVML (optional – gives util / power / temp / clocks)
        self._nvml = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass  # silently fall back; torch memory stats still work

        # Accumulated stats for final summary
        self._best_val_loss = float("inf")
        self._peak_alloc_gb = 0.0
        self._peak_reserved_gb = 0.0
        self._total_tokens = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------
    def log(self, event: str, **kwargs: Any) -> None:
        record = {"timestamp": time.time(), "event": event, **kwargs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_config(self, config: Dict[str, Any]) -> None:
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def write_summary(self, **extra: Any) -> None:
        summary = {
            "best_val_loss": self._best_val_loss,
            "peak_mem_allocated_gb": self._peak_alloc_gb,
            "peak_mem_reserved_gb": self._peak_reserved_gb,
            "total_tokens": self._total_tokens,
            "wall_time_sec": time.time() - self._start_time,
            "avg_tokens_per_sec": (
                sum(self._tok_s_window) / len(self._tok_s_window)
                if self._tok_s_window else None
            ),
            **extra,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.log("summary", **summary)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def log_environment(self, dtype: str, device: str) -> None:
        info = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": dtype,
            "device": device,
        }
        # git commit (best-effort)
        try:
            info["git_commit"] = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            info["git_commit"] = None

        self.log("environment", **info)

    def log_compile(self, compile_time_sec: float) -> None:
        self.log("compile", compile_time_sec=compile_time_sec)

    def log_run_start(self, config: Dict[str, Any], num_params: int) -> None:
        self.write_config(config)
        self.log("run_start", config=config, num_params=num_params)

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Return a dict of useful GPU metrics (always includes torch memory)."""
        stats: Dict[str, Any] = {}

        if torch.cuda.is_available():
            stats["mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            stats["max_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

            # update peaks for summary
            self._peak_alloc_gb = max(self._peak_alloc_gb, stats["max_mem_allocated_gb"])
            self._peak_reserved_gb = max(self._peak_reserved_gb, stats["max_mem_reserved_gb"])

        # Rich stats via NVML if available
        if self._nvml is not None and self._handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                stats["gpu_util_pct"] = util.gpu
                stats["mem_util_pct"] = util.memory

                power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW → W
                stats["power_draw_w"] = power

                try:
                    limit = self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
                    stats["power_limit_w"] = limit
                except Exception:
                    pass

                temp = self._nvml.nvmlDeviceGetTemperature(
                    self._handle, self._nvml.NVML_TEMPERATURE_GPU
                )
                stats["temperature_c"] = temp

                # clocks
                try:
                    stats["graphics_clock_mhz"] = self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_GRAPHICS
                    )
                    stats["memory_clock_mhz"] = self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_MEM
                    )
                except Exception:
                    pass
            except Exception:
                pass

        return stats

    def log_eval(self, iter_num: int, val_loss: float, lr: float) -> None:
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
        stats = self.get_gpu_stats()
        self.log(
            "eval",
            iter=iter_num,
            val_loss=val_loss,
            lr=lr,
            **stats,
        )

    def log_checkpoint(self, iter_num: int, best_val_loss: float) -> None:
        self.log("checkpoint", iter=iter_num, best_val_loss=best_val_loss)

    def log_train_step(
        self,
        iter_num: int,
        loss: float,
        time_ms: float,
        tokens_per_sec: float,
        samples_per_sec: float,
        lr: float,
        mfu_pct: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        self._tok_s_window.append(tokens_per_sec)
        self._samples_s_window.append(samples_per_sec)
        self._total_tokens += int(tokens_per_sec * (time_ms / 1000.0))

        record = {
            "iter": iter_num,
            "loss": loss,
            "time_ms": time_ms,
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": samples_per_sec,
            "rolling_tokens_per_sec": sum(self._tok_s_window) / len(self._tok_s_window),
            "lr": lr,
        }
        if mfu_pct is not None:
            record["mfu_pct"] = mfu_pct
        if grad_norm is not None:
            record["grad_norm"] = grad_norm

        # occasionally attach GPU stats (every 10 steps is plenty)
        if iter_num % 10 == 0:
            record.update(self.get_gpu_stats())

        self.log("train_step", **record)

    def log_optimizer(self, iter_num: int, grad_norm: float) -> None:
        self.log("optimizer", iter=iter_num, grad_norm=grad_norm)