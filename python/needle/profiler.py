"""Lightweight profiling for Needle operations and NDArray allocations.

This profiler measures Python-side operator dispatch time and counts fresh
NDArray backing-buffer allocations. GPU kernels are asynchronous, so operation
times on CUDA are host/launch times, not accurate kernel durations; use Nsight
Systems/Compute for CUDA kernel timing.

Example::

    profiler = needle.profiler.Profiler()
    with profiler:
        with profiler.phase("forward"):
            loss = model(x)
        with profiler.phase("backward"):
            loss.backward()
        with profiler.phase("optimizer"):
            optimizer.step()
    print(profiler.summary())
"""

from collections import defaultdict
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Dict, Optional


_ACTIVE_PROFILER: ContextVar[Optional["Profiler"]] = ContextVar(
    "needle_active_profiler", default=None
)
_CURRENT_OP: ContextVar[Optional[str]] = ContextVar("needle_current_op", default=None)
_HOOKS_INSTALLED = False


def _install_hooks() -> None:
    """Install low-overhead global hooks once; they are inactive outside a profile."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    from .autograd import Value
    from .backend_ndarray.ndarray import NDArray

    original_realize = Value.realize_cached_data

    def profiled_realize(value):
        profiler = _ACTIVE_PROFILER.get()
        if profiler is None or value.cached_data is not None:
            return original_realize(value)

        op = value.op
        if op is None:
            return original_realize(value)

        # Resolve inputs first. Their computations are independently timed.
        inputs = [item.realize_cached_data() for item in value.inputs]
        op_name = type(op).__name__
        op_token = _CURRENT_OP.set(op_name)
        start = perf_counter()
        succeeded = False
        try:
            result = op.compute(*inputs)
            value.cached_data = result
            succeeded = True
            return result
        finally:
            elapsed = perf_counter() - start
            _CURRENT_OP.reset(op_token)
            profiler._record_op(op_name, elapsed, succeeded)

    Value.realize_cached_data = profiled_realize

    original_make = NDArray.make

    def profiled_make(shape, strides=None, device=None, handle=None, offset=0):
        array = original_make(shape, strides, device, handle, offset)
        profiler = _ACTIVE_PROFILER.get()
        if profiler is not None and handle is None:
            profiler._record_allocation(
                device=array.device,
                elements=array.size,
                op_name=_CURRENT_OP.get(),
            )
        return array

    NDArray.make = staticmethod(profiled_make)
    _HOOKS_INSTALLED = True


class Profiler:
    """Collect per-op host timings, phase timings, and fresh NDArray allocations.

    Use ``phase(name)`` to separately measure forward, backward, and optimizer
    work. Allocation totals include buffers allocated through ``NDArray.make``;
    views that reuse an existing buffer are excluded. Bytes assume float32, as
    this Needle NDArray backend currently supports float32 only.
    """

    def __init__(self) -> None:
        self.reset()
        self._phase = "unlabeled"
        self._started_at = None
        self._token: Optional[Token] = None

    def reset(self) -> None:
        """Clear all measurements."""
        self._ops = defaultdict(lambda: {"calls": 0, "time_s": 0.0, "errors": 0})
        self._allocations = defaultdict(
            lambda: {"count": 0, "bytes": 0, "elements": 0}
        )
        self._phases = defaultdict(lambda: {"calls": 0, "time_s": 0.0})
        self._total_time_s = 0.0

    def __enter__(self) -> "Profiler":
        _install_hooks()
        self._started_at = perf_counter()
        self._token = _ACTIVE_PROFILER.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._started_at is not None:
            self._total_time_s += perf_counter() - self._started_at
            self._started_at = None
        if self._token is not None:
            _ACTIVE_PROFILER.reset(self._token)
            self._token = None
        return False

    class _Phase:
        def __init__(self, profiler: "Profiler", name: str) -> None:
            self.profiler = profiler
            self.name = name
            self.started = None
            self.previous = None

        def __enter__(self):
            self.previous = self.profiler._phase
            self.profiler._phase = self.name
            self.started = perf_counter()
            self.profiler._phases[self.name]["calls"] += 1
            return self.profiler

        def __exit__(self, exc_type, exc_value, traceback):
            self.profiler._phases[self.name]["time_s"] += perf_counter() - self.started
            self.profiler._phase = self.previous
            return False

    def phase(self, name: str) -> "Profiler._Phase":
        """Return a context manager that labels work with ``name``."""
        if not isinstance(name, str) or not name:
            raise ValueError("phase name must be a non-empty string")
        return self._Phase(self, name)

    def _record_op(self, op_name: str, elapsed: float, succeeded: bool) -> None:
        item = self._ops[(self._phase, op_name)]
        item["calls"] += 1
        item["time_s"] += elapsed
        if not succeeded:
            item["errors"] += 1

    def _record_allocation(self, device, elements: int, op_name: Optional[str]) -> None:
        device_name = getattr(device, "name", str(device))
        key = (self._phase, op_name or "unattributed", device_name)
        item = self._allocations[key]
        item["count"] += 1
        item["elements"] += elements
        item["bytes"] += elements * 4

    def summary(self) -> dict:
        """Return a JSON-serializable snapshot of collected measurements."""
        operations = {}
        for (phase, op_name), values in sorted(self._ops.items()):
            phase_ops = operations.setdefault(phase, {})
            phase_ops[op_name] = {
                **values,
                "avg_time_s": values["time_s"] / values["calls"],
            }

        allocations = {}
        allocation_totals = {"count": 0, "elements": 0, "bytes": 0}
        for (phase, op_name, device), values in sorted(self._allocations.items()):
            phase_allocs = allocations.setdefault(phase, {})
            op_allocs = phase_allocs.setdefault(op_name, {})
            op_allocs[device] = dict(values)
            for field in allocation_totals:
                allocation_totals[field] += values[field]

        phases = {name: dict(values) for name, values in sorted(self._phases.items())}
        return {
            "total_time_s": self._total_time_s,
            "phases": phases,
            "operations": operations,
            "allocations": allocations,
            "allocation_totals": allocation_totals,
        }

    def report(self) -> str:
        """Format the summary as readable text."""
        data = self.summary()
        lines = [f"Profile wall time: {data['total_time_s']:.6f}s"]
        if data["phases"]:
            lines.append("Phases:")
            for name, values in data["phases"].items():
                lines.append(
                    f"  {name}: {values['time_s']:.6f}s ({values['calls']} call(s))"
                )
        if data["operations"]:
            lines.append("Operations (host time):")
            for phase, entries in data["operations"].items():
                for name, values in entries.items():
                    lines.append(
                        f"  {phase}/{name}: {values['calls']} call(s), "
                        f"{values['time_s']:.6f}s total, "
                        f"{values['avg_time_s']:.6f}s avg"
                    )
        totals = data["allocation_totals"]
        lines.append(
            "Fresh NDArray buffers: "
            f"{totals['count']} allocation(s), {totals['bytes']} bytes"
        )
        if totals["count"]:
            lines.append("Allocations by phase/op/device:")
            for phase, by_op in data["allocations"].items():
                for op_name, by_device in by_op.items():
                    for device, values in by_device.items():
                        lines.append(
                            f"  {phase}/{op_name}/{device}: "
                            f"{values['count']} buffer(s), {values['bytes']} bytes"
                        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def count_parameters(model, trainable_only: bool = False) -> int:
    """Count parameter elements, optionally counting only trainable parameters."""
    return sum(
        parameter.size
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_report(model) -> dict:
    """Return total/trainable/frozen model parameter counts."""
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
    }
