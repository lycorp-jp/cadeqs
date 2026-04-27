from __future__ import annotations

import time
from statistics import quantiles
from typing import Any, Callable

from codecarbon import EmissionsTracker


class InferenceBenchmark:
    """Context manager that tracks per-call latency and optional energy."""

    def __init__(
        self,
        project_name: str = "inference",
        track_energy: bool = False,
    ):
        self.latencies: list[float] = []
        self.energy_kwh: float | None = None
        self._track_energy = track_energy
        self._project_name = project_name
        self._tracker: Any = None

    def __enter__(self) -> "InferenceBenchmark":
        if self._track_energy:
            self._tracker = EmissionsTracker(
                project_name=self._project_name,
                measure_power_secs=1,
                log_level="error",
                save_to_file=False,
            )
            self._tracker.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tracker is not None:
            self.energy_kwh = self._tracker.stop()

    def timed_call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        self.latencies.append(time.perf_counter() - t0)
        return result

    def summary(self) -> dict[str, Any]:
        n = len(self.latencies)
        total_s = sum(self.latencies)
        avg_ms = total_s / n * 1000 if n else 0.0
        p95_ms = quantiles(self.latencies, n=100)[94] * 1000 if n >= 2 else avg_ms

        d: dict[str, Any] = {
            "total_queries": n,
            "total_time_s": round(total_s, 4),
            "avg_latency_ms": round(avg_ms, 4),
            "p95_latency_ms": round(p95_ms, 4),
        }
        if self.energy_kwh is not None:
            total_j = self.energy_kwh * 3_600_000
            d["joules_per_query"] = round(total_j / n, 6) if n else 0.0
        return d
