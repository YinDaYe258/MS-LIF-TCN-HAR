from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PowerRecord:
    timestamp: float
    power_w: float
    gpu_util: float | None = None
    mem_used_mb: float | None = None
    temperature_c: float | None = None


def integrate_energy_j(records: list[PowerRecord] | list[dict[str, Any]]) -> float:
    """Integrate power samples with the trapezoidal rule."""

    if len(records) < 2:
        return 0.0
    normalized: list[PowerRecord] = []
    for record in records:
        if isinstance(record, PowerRecord):
            normalized.append(record)
        else:
            normalized.append(PowerRecord(timestamp=float(record["timestamp"]), power_w=float(record["power_w"])))
    normalized = sorted(normalized, key=lambda item: item.timestamp)
    energy = 0.0
    for left, right in zip(normalized[:-1], normalized[1:]):
        dt = max(0.0, right.timestamp - left.timestamp)
        energy += 0.5 * (left.power_w + right.power_w) * dt
    return float(energy)


class NVMLPowerMonitor:
    """Sample NVIDIA GPU power through NVML.

    These readings describe the CUDA/NVML software stack on an NVIDIA GPU. They
    are not neuromorphic hardware power measurements.
    """

    def __init__(self, gpu_index: int = 0, idle_power_w: float | None = None) -> None:
        self.gpu_index = int(gpu_index)
        self.idle_power_w = float(idle_power_w) if idle_power_w is not None else 0.0
        self.records: list[PowerRecord] = []
        self._pynvml = None
        self._handle = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time = 0.0
        self._end_time = 0.0

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only without NVML package
            raise RuntimeError(
                "NVML monitoring requires nvidia-ml-py. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise RuntimeError(f"Failed to initialize NVML for GPU {self.gpu_index}: {exc}") from exc

    def start(self, sample_interval_ms: int = 20) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("NVMLPowerMonitor is already running")
        if self._pynvml is None or self._handle is None:
            self._init_nvml()
        self.records = []
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        interval_s = max(float(sample_interval_ms) / 1000.0, 0.001)
        self._thread = threading.Thread(target=self._sample_loop, args=(interval_s,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._end_time = time.perf_counter()
        self._thread = None

    def measure_idle_power(self, duration_s: float = 1.0, sample_interval_ms: int = 20) -> float:
        if self._pynvml is None or self._handle is None:
            self._init_nvml()
        duration_s = max(float(duration_s), 0.05)
        interval_s = max(float(sample_interval_ms) / 1000.0, 0.001)
        end_time = time.perf_counter() + duration_s
        samples: list[float] = []
        while time.perf_counter() < end_time:
            samples.append(self._read_record().power_w)
            time.sleep(interval_s)
        self.idle_power_w = float(sum(samples) / len(samples)) if samples else 0.0
        return self.idle_power_w

    def _sample_loop(self, interval_s: float) -> None:
        while not self._stop_event.is_set():
            self.records.append(self._read_record())
            time.sleep(interval_s)
        self.records.append(self._read_record())

    def _read_record(self) -> PowerRecord:
        assert self._pynvml is not None
        assert self._handle is not None
        now = time.perf_counter()
        power_w = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_util = float(util.gpu)
        except Exception:
            gpu_util = None
        try:
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            mem_used_mb = float(mem.used) / (1024.0 * 1024.0)
        except Exception:
            mem_used_mb = None
        try:
            temperature_c = float(self._pynvml.nvmlDeviceGetTemperature(self._handle, self._pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temperature_c = None
        return PowerRecord(
            timestamp=now,
            power_w=power_w,
            gpu_util=gpu_util,
            mem_used_mb=mem_used_mb,
            temperature_c=temperature_c,
        )

    @property
    def elapsed_s(self) -> float:
        if not self.records:
            return 0.0
        return max(0.0, self.records[-1].timestamp - self.records[0].timestamp)

    @property
    def average_power_w(self) -> float:
        if not self.records:
            return 0.0
        return float(sum(record.power_w for record in self.records) / len(self.records))

    @property
    def peak_power_w(self) -> float:
        if not self.records:
            return 0.0
        return float(max(record.power_w for record in self.records))

    @property
    def active_energy_j(self) -> float:
        return integrate_energy_j(self.records)

    @property
    def net_energy_j(self) -> float:
        return max(0.0, self.active_energy_j - self.idle_power_w * self.elapsed_s)

    @property
    def peak_mem_used_mb(self) -> float:
        values = [record.mem_used_mb for record in self.records if record.mem_used_mb is not None]
        return float(max(values)) if values else 0.0

    @property
    def average_gpu_util(self) -> float:
        values = [record.gpu_util for record in self.records if record.gpu_util is not None]
        return float(sum(values) / len(values)) if values else 0.0

