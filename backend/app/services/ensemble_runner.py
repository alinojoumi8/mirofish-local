"""Run a prepared simulation multiple times with different seeds and aggregate the
grounded forecast signal into a calibrated distribution.

A single simulation run is one sample of a stochastic process (agent activation,
ordering, LLM sampling). Reporting one run's numbers as "the forecast" overstates
certainty. Running N seeded runs and aggregating the per-run grounded signals
(see ForecastSynthesizer.build_simulation_signal) yields a mean distribution plus
a spread that becomes the forecast's confidence. The result is persisted to
``ensemble_signal.json`` in the simulation directory, which ForecastSynthesizer
prefers over a single run when present.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from .forecasting import (
    DEFAULT_OUTCOMES,
    ForecastSynthesizer,
    net_to_probabilities,
    normalize_forecast_settings,
)
from .simulation_runner import RunnerStatus, SimulationRunner

logger = get_logger("mirofish.ensemble")


class EnsembleRunner:
    DEFAULT_RUNS = 3
    MAX_RUNS = 10

    @classmethod
    def run_ensemble(
        cls,
        simulation_id: str,
        runs: Optional[int] = None,
        max_rounds: Optional[int] = None,
        platform: str = "parallel",
        storage: Any = None,
        memory_mode: str = "practical",
        economy: Optional[Dict[str, Any]] = None,
        per_run_timeout: float = 1200.0,
        progress_callback: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        config = cls._load_config(simulation_id)
        if not config:
            raise ValueError(f"Simulation config not found for {simulation_id}; call /prepare first")

        settings = normalize_forecast_settings(
            config, simulation_requirement=config.get("simulation_requirement", "")
        )
        mode = settings.forecast_mode
        outcomes = list(settings.prediction_target.get("outcomes") or DEFAULT_OUTCOMES[mode])[:5]
        runs = max(1, min(int(runs or settings.ensemble_runs or cls.DEFAULT_RUNS), cls.MAX_RUNS))
        graph_id = config.get("graph_id")
        economy = config.get("economy") if economy is None else economy

        per_run_net: List[float] = []
        per_run_probs: List[List[float]] = []
        per_run_signals: List[Optional[Dict[str, Any]]] = []

        for i in range(runs):
            seed = 1000 + i * 97
            cls._reset(simulation_id)
            try:
                SimulationRunner.start_simulation(
                    simulation_id=simulation_id,
                    platform=platform,
                    max_rounds=max_rounds,
                    enable_graph_memory_update=bool(storage),
                    graph_id=graph_id,
                    storage=storage,
                    scenario_id="baseline",
                    seed=seed,
                    memory_mode=memory_mode,
                    economy=economy,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Ensemble run %s/%s failed to start: %s", i + 1, runs, exc)
                if progress_callback:
                    progress_callback(i + 1, runs, None)
                continue

            completed = cls._wait_for_completion(simulation_id, per_run_timeout)
            signal = (
                ForecastSynthesizer(simulation_id, config).build_simulation_signal()
                if completed else None
            )
            try:
                net = float(signal["net_sentiment"]) if signal else float("nan")
                usable = bool(signal and signal.get("total_actions", 0) > 0 and math.isfinite(net))
            except (KeyError, TypeError, ValueError):
                usable = False
            if not usable:
                logger.warning("Ensemble run %s/%s produced no usable completed sample", i + 1, runs)
                if progress_callback:
                    progress_callback(i + 1, runs, None)
                continue
            per_run_signals.append(signal)
            per_run_net.append(net)
            per_run_probs.append(net_to_probabilities(net, outcomes, mode, span=0.5))
            logger.info(
                "Ensemble run %s/%s seed=%s completed=%s net=%.3f actions=%s",
                i + 1, runs, seed, completed, net,
                (signal or {}).get("total_actions", 0),
            )
            if progress_callback:
                progress_callback(i + 1, runs, signal)

        # Stop the final environment without deleting its source evidence. Destructive
        # cleanup belongs before each sample, before the aggregate is persisted.
        cls._shutdown(simulation_id)
        if not per_run_signals:
            raise RuntimeError("Ensemble produced no usable completed simulation runs")
        aggregate = cls._aggregate(runs, outcomes, mode, per_run_net, per_run_probs, per_run_signals)
        cls._persist(simulation_id, aggregate)
        return aggregate

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_config(simulation_id: str) -> Dict[str, Any]:
        path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "simulation_config.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _reset(cls, simulation_id: str) -> None:
        """Best-effort: gracefully close any live env, stop a running process, and
        clear runtime logs so the next seeded run starts clean."""
        cls._shutdown(simulation_id)
        result = SimulationRunner.cleanup_simulation_logs(simulation_id)
        if result and not result.get("success", False):
            raise RuntimeError(f"Cannot isolate ensemble sample: {result.get('errors')}")

    @classmethod
    def _shutdown(cls, simulation_id: str) -> None:
        """Close runtime processes while preserving the final sample evidence."""
        try:
            SimulationRunner.close_simulation_env(simulation_id, timeout=20)
        except Exception as exc:
            logger.debug("close_simulation_env during reset: %s", exc)
        try:
            state = SimulationRunner.get_run_state(simulation_id)
            if state and state.runner_status in (RunnerStatus.RUNNING, RunnerStatus.STARTING):
                SimulationRunner.stop_simulation(simulation_id)
        except Exception as exc:
            logger.debug("stop_simulation during reset: %s", exc)

    @staticmethod
    def _wait_for_completion(simulation_id: str, timeout: float) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            state = SimulationRunner.get_run_state(simulation_id)
            if state and state.runner_status in (RunnerStatus.COMPLETED, RunnerStatus.FAILED, RunnerStatus.STOPPED):
                return state.runner_status == RunnerStatus.COMPLETED
            time.sleep(3)
        logger.warning("Ensemble run timed out after %ss for %s", timeout, simulation_id)
        return False

    @staticmethod
    def _aggregate(
        runs: int,
        outcomes: List[str],
        mode: str,
        per_run_net: List[float],
        per_run_probs: List[List[float]],
        per_run_signals: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        completed = len(per_run_net)
        mean_net = round(statistics.fmean(per_run_net), 3) if per_run_net else 0.0
        std_net = round(statistics.pstdev(per_run_net), 3) if len(per_run_net) > 1 else 0.0

        n_out = len(outcomes)
        mean_probs: List[float] = []
        prob_std: List[float] = []
        for k in range(n_out):
            column = [probs[k] for probs in per_run_probs if k < len(probs)]
            if column:
                mean_probs.append(statistics.fmean(column))
                prob_std.append(round(statistics.pstdev(column), 4) if len(column) > 1 else 0.0)
            else:
                mean_probs.append(0.0)
                prob_std.append(0.0)
        norm = sum(mean_probs) or 1.0
        mean_probs = [round(p / norm, 4) for p in mean_probs]

        total_actions = sum((sig or {}).get("total_actions", 0) for sig in per_run_signals)
        active_agents = max(((sig or {}).get("active_agents", 0) for sig in per_run_signals), default=0)
        agents_supporting = max(((sig or {}).get("agents_supporting", 0) for sig in per_run_signals), default=0)
        agents_opposing = max(((sig or {}).get("agents_opposing", 0) for sig in per_run_signals), default=0)

        return {
            "runs": runs,
            "completed_runs": completed,
            "forecast_mode": mode,
            "outcomes": outcomes,
            "per_run_net": [round(n, 3) for n in per_run_net],
            "mean_net": mean_net,
            "std_net": std_net,
            "mean_probabilities": mean_probs,
            "probability_std": prob_std,
            "total_actions": total_actions,
            "active_agents": active_agents,
            "agents_supporting": agents_supporting,
            "agents_opposing": agents_opposing,
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _persist(simulation_id: str, aggregate: Dict[str, Any]) -> None:
        path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "ensemble_signal.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error("Failed to persist ensemble signal for %s: %s", simulation_id, exc)
            raise
