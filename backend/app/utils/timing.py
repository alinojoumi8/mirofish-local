"""
Phase timing helpers for long-running pipeline tasks.

Durations use time.perf_counter() so elapsed seconds are monotonic and not
affected by wall-clock adjustments. ISO timestamps are kept for auditability.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional


def _now_iso() -> str:
    return datetime.now().isoformat()


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 4)


class PhaseTimer:
    """Record ordered phase durations for a pipeline run."""

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self.started_at = _now_iso()
        self._started_perf = time.perf_counter()
        self._completed_perf: Optional[float] = None
        self.completed_at: Optional[str] = None
        self.status = "running"
        self.error: Optional[str] = None
        self._phases: List[Dict[str, Any]] = []
        self._active: Dict[str, List[Dict[str, Any]]] = {}

    def start(
        self,
        name: str,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        phase = {
            "name": name,
            "label": label or name,
            "started_at": _now_iso(),
            "_started_perf": time.perf_counter(),
            "metadata": metadata or {},
            "status": "running",
        }
        self._active.setdefault(name, []).append(phase)
        return phase

    def finish(
        self,
        name: str,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_stack = self._active.get(name) or []
        if not active_stack:
            raise ValueError(f"Phase is not active: {name}")

        phase = active_stack.pop()
        if not active_stack:
            self._active.pop(name, None)

        phase["completed_at"] = _now_iso()
        phase["duration_seconds"] = _round_seconds(time.perf_counter() - phase["_started_perf"])
        phase["status"] = status
        if metadata:
            phase["metadata"] = {**phase.get("metadata", {}), **metadata}
        if error:
            phase["error"] = error
        phase.pop("_started_perf", None)
        self._phases.append(phase)
        return phase

    @contextmanager
    def time(
        self,
        name: str,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        self.start(name, label=label, metadata=metadata)
        try:
            yield
        except Exception as exc:
            self.finish(name, status="failed", error=str(exc))
            raise
        else:
            self.finish(name)

    def complete(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.status = "completed"
        self.completed_at = _now_iso()
        self._completed_perf = time.perf_counter()
        if metadata:
            self.metadata = {**self.metadata, **metadata}
        return self.snapshot()

    def fail(self, error: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.status = "failed"
        self.error = error
        self.completed_at = _now_iso()
        self._completed_perf = time.perf_counter()
        if metadata:
            self.metadata = {**self.metadata, **metadata}
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        now_perf = time.perf_counter()
        end_perf = self._completed_perf or now_perf
        phases = [dict(phase) for phase in self._phases]
        active_phases = []
        if self.status == "running":
            for stack in self._active.values():
                for phase in stack:
                    active_phases.append({
                        "name": phase["name"],
                        "label": phase["label"],
                        "started_at": phase["started_at"],
                        "duration_seconds": _round_seconds(now_perf - phase["_started_perf"]),
                        "metadata": phase.get("metadata", {}),
                        "status": "running",
                    })

        phase_totals: Dict[str, float] = {}
        for phase in phases + active_phases:
            phase_totals[phase["name"]] = _round_seconds(
                phase_totals.get(phase["name"], 0.0) + phase.get("duration_seconds", 0.0)
            )

        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_seconds": _round_seconds(end_perf - self._started_perf),
            "phase_totals": phase_totals,
            "phases": phases,
            "active_phases": active_phases,
            "metadata": self.metadata,
            "error": self.error,
        }


def wall_clock_timing(
    name: str,
    started_at: Optional[str],
    completed_at: Optional[str] = None,
    status: str = "running",
) -> Dict[str, Any]:
    """Build a timing summary from persisted ISO timestamps."""
    duration = 0.0
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(completed_at) if completed_at else datetime.now()
            duration = max(0.0, (end_dt - start_dt).total_seconds())
        except ValueError:
            duration = 0.0

    phase = {
        "name": name,
        "label": name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": _round_seconds(duration),
        "metadata": {},
        "status": status,
    }
    return {
        "name": name,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_seconds": phase["duration_seconds"],
        "duration_seconds": phase["duration_seconds"],
        "phase_totals": {name: phase["duration_seconds"]},
        "phases": [phase],
        "active_phases": [] if completed_at else [phase],
        "metadata": {},
        "error": None,
    }
