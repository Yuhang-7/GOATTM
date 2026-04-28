from __future__ import annotations

import contextlib
import contextvars
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable[..., object])


@dataclass(frozen=True)
class FunctionTimingRecord:
    name: str
    call_count: int
    total_seconds: float

    @property
    def average_seconds(self) -> float:
        if self.call_count <= 0:
            return 0.0
        return self.total_seconds / self.call_count


class FunctionTimer:
    def __init__(self) -> None:
        self._stats: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_seconds: float) -> None:
        if name not in self._stats:
            self._stats[name] = [0.0, 0.0]
        self._stats[name][0] += 1.0
        self._stats[name][1] += float(elapsed_seconds)

    def records(self) -> list[FunctionTimingRecord]:
        return sorted(
            (
                FunctionTimingRecord(
                    name=name,
                    call_count=int(values[0]),
                    total_seconds=float(values[1]),
                )
                for name, values in self._stats.items()
            ),
            key=lambda record: (-record.total_seconds, record.name),
        )

    def to_json_ready(self) -> dict[str, object]:
        records = self.records()
        return {
            "total_profiled_seconds": float(sum(record.total_seconds for record in records)),
            "records": [
                {
                    "name": record.name,
                    "call_count": record.call_count,
                    "total_seconds": record.total_seconds,
                    "average_seconds": record.average_seconds,
                }
                for record in records
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_ready(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return path

    def write_text(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["GOATTM Function Timing Summary", "name call_count total_seconds average_seconds"]
        for record in self.records():
            lines.append(
                f"{record.name} {record.call_count:d} "
                f"{record.total_seconds:.16e} {record.average_seconds:.16e}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


_ACTIVE_FUNCTION_TIMER: contextvars.ContextVar[FunctionTimer | None] = contextvars.ContextVar(
    "goattm_active_function_timer",
    default=None,
)


@contextlib.contextmanager
def use_function_timer(timer: FunctionTimer):
    token = _ACTIVE_FUNCTION_TIMER.set(timer)
    try:
        yield timer
    finally:
        _ACTIVE_FUNCTION_TIMER.reset(token)


def active_function_timer() -> FunctionTimer | None:
    return _ACTIVE_FUNCTION_TIMER.get()


def timed(name: str | None = None):
    def decorator(function: F) -> F:
        timer_name = name if name is not None else f"{function.__module__}.{function.__qualname__}"

        def wrapped(*args, **kwargs):
            timer = active_function_timer()
            if timer is None:
                return function(*args, **kwargs)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timer.record(timer_name, time.perf_counter() - start)

        wrapped.__name__ = function.__name__
        wrapped.__qualname__ = function.__qualname__
        wrapped.__doc__ = function.__doc__
        wrapped.__module__ = function.__module__
        return wrapped  # type: ignore[return-value]

    return decorator
