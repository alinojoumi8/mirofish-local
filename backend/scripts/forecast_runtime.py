"""Runtime helpers for forecast-aware OASIS simulations."""

from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole
except Exception:  # pragma: no cover - scripts may import before camel is installed
    BaseMessage = None
    OpenAIBackendRole = None


SHORT_MEMORY_FILE = "agent_short_memory.json"
SUMMARY_MEMORY_FILE = "agent_memory_summaries.json"
MAX_MEMORY_ITEMS = 8
SUMMARY_INTERVAL_ROUNDS = 5


def _memory_path(simulation_dir: str) -> str:
    return os.path.join(simulation_dir, SHORT_MEMORY_FILE)


def load_short_memory(simulation_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    path = _memory_path(simulation_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_short_memory(simulation_dir: str, memory: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        with open(_memory_path(simulation_dir), "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _summary_path(simulation_dir: str) -> str:
    return os.path.join(simulation_dir, SUMMARY_MEMORY_FILE)


def load_memory_summaries(simulation_dir: str) -> Dict[str, Dict[str, Any]]:
    path = _summary_path(simulation_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_memory_summaries(simulation_dir: str, summaries: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(_summary_path(simulation_dir), "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def record_actions_to_short_memory(
    simulation_dir: str,
    platform: str,
    round_num: int,
    actions: List[Dict[str, Any]],
) -> None:
    if not actions:
        return
    memory = load_short_memory(simulation_dir)
    for action in actions:
        agent_id = str(action.get("agent_id"))
        if agent_id == "None":
            continue
        args = action.get("action_args") or {}
        content = (
            args.get("content")
            or args.get("quote_content")
            or args.get("original_content")
            or args.get("post_content")
            or args.get("query")
            or ""
        )
        item = {
            "round": round_num,
            "platform": platform,
            "action_type": action.get("action_type"),
            "content": str(content)[:500],
            "recorded_at": datetime.now().isoformat(),
        }
        memory.setdefault(agent_id, []).append(item)
        memory[agent_id] = memory[agent_id][-MAX_MEMORY_ITEMS:]
    save_short_memory(simulation_dir, memory)


def update_memory_summaries(simulation_dir: str, round_num: int) -> None:
    """Compact short-term memories every few rounds to avoid prompt bloat."""
    if round_num <= 0 or round_num % SUMMARY_INTERVAL_ROUNDS != 0:
        return
    short_memory = load_short_memory(simulation_dir)
    summaries = load_memory_summaries(simulation_dir)
    for agent_id, items in short_memory.items():
        if not items:
            continue
        recent = items[-MAX_MEMORY_ITEMS:]
        action_types = {}
        signals = []
        for item in recent:
            action_type = item.get("action_type") or "ACTION"
            action_types[action_type] = action_types.get(action_type, 0) + 1
            content = (item.get("content") or "").strip()
            if content:
                signals.append(content[:140])
        summary_text = (
            f"Recent behavior through round {round_num}: "
            + ", ".join(f"{count} {action}" for action, count in sorted(action_types.items()))
        )
        if signals:
            summary_text += ". Signals: " + " | ".join(signals[-3:])
        summaries[agent_id] = {
            "round": round_num,
            "summary": summary_text[:800],
            "updated_at": datetime.now().isoformat(),
        }
    save_memory_summaries(simulation_dir, summaries)


def select_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int,
) -> List[Tuple[int, Any]]:
    """Choose active agents using time, activity, post/comment rates, and influence."""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])

    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])

    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0

    target_count = max(1, int(random.uniform(base_min, base_max) * multiplier))
    weighted_candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        if current_hour not in cfg.get("active_hours", list(range(8, 23))):
            continue

        activity = float(cfg.get("activity_level", 0.5))
        posts = float(cfg.get("posts_per_hour", 0.5))
        comments = float(cfg.get("comments_per_hour", 1.0))
        influence = min(float(cfg.get("influence_weight", 1.0)), 4.0)
        response_boost = _recent_event_response_boost(config, cfg, round_num)
        activation_probability = min(
            0.98,
            activity * (0.55 + min(posts + comments, 4.0) * 0.12) * (0.8 + influence * 0.08) * response_boost,
        )
        if random.random() < activation_probability:
            weighted_candidates.append((agent_id, max(0.01, influence * activation_probability)))

    selected_ids = _weighted_sample_without_replacement(weighted_candidates, target_count)
    active_agents = []
    for agent_id in selected_ids:
        try:
            active_agents.append((agent_id, env.agent_graph.get_agent(agent_id)))
        except Exception:
            pass
    return active_agents


def _recent_event_response_boost(config: Dict[str, Any], cfg: Dict[str, Any], round_num: int) -> float:
    events = config.get("event_config", {}).get("scheduled_events", [])
    if not events:
        return 1.0
    response_delay_min = int(cfg.get("response_delay_min", 5))
    response_delay_max = int(cfg.get("response_delay_max", 60))
    minutes_per_round = max(1, int(config.get("time_config", {}).get("minutes_per_round", 60)))
    for event in events:
        event_round = int(event.get("round") or event.get("trigger_round") or -999)
        if event_round <= round_num:
            delta_minutes = (round_num - event_round) * minutes_per_round
            if response_delay_min <= delta_minutes <= max(response_delay_max, response_delay_min):
                return 1.35
    return 1.0


def _weighted_sample_without_replacement(weighted: List[Tuple[int, float]], count: int) -> List[int]:
    pool = list(weighted)
    selected = []
    while pool and len(selected) < count:
        total = sum(weight for _, weight in pool)
        pick = random.random() * total
        running = 0.0
        chosen_idx = 0
        for idx, (_, weight) in enumerate(pool):
            running += weight
            if running >= pick:
                chosen_idx = idx
                break
        selected.append(pool.pop(chosen_idx)[0])
    return selected


def due_scheduled_events(
    config: Dict[str, Any],
    round_num: int,
    simulated_hour: int,
    platform: str,
) -> List[Dict[str, Any]]:
    events = config.get("event_config", {}).get("scheduled_events", [])
    due = []
    for event in events:
        event_round = event.get("round", event.get("trigger_round"))
        event_hour = event.get("hour", event.get("trigger_hour"))
        if event_round is not None and int(event_round) != round_num:
            continue
        if event_round is None and event_hour is not None and int(event_hour) != simulated_hour:
            continue
        if event.get("platform") and event["platform"] not in {platform, "both", "parallel"}:
            continue
        content = event.get("content") or event.get("title") or event.get("description")
        if content:
            due.append(dict(event, content=content))
    return due


def inject_agent_forecast_memory(
    agent: Any,
    agent_id: int,
    config: Dict[str, Any],
    simulation_dir: str,
    platform: str,
    round_num: int,
    simulated_hour: int,
    event_context: Optional[str] = None,
) -> None:
    """Inject concise forecast and memory context into an agent before LLM action."""
    if BaseMessage is None or OpenAIBackendRole is None:
        return
    memory_mode = config.get("memory_mode") or config.get("forecast_memory_mode") or "practical"
    economy_enabled = bool((config.get("economy") or {}).get("enabled", False))
    if memory_mode in {"off", "none", "disabled"} and not economy_enabled:
        return

    agent_config = _agent_config(config, agent_id)
    forecast_context = _forecast_context(config)
    memories = [] if memory_mode in {"off", "none", "disabled"} else load_short_memory(simulation_dir).get(str(agent_id), [])[-4:]
    summary = {} if memory_mode in {"off", "none", "disabled"} else load_memory_summaries(simulation_dir).get(str(agent_id), {})
    economy_context = _economy_context(simulation_dir, agent_id) if economy_enabled else ""
    memory_lines = []
    for item in memories:
        content = item.get("content") or ""
        if content:
            memory_lines.append(f"R{item.get('round')} {item.get('action_type')}: {content[:180]}")

    content = "\n".join([
        "[Forecast Runtime Context]",
        forecast_context,
        f"Current platform: {platform}; round: {round_num}; simulated hour: {simulated_hour}.",
        f"Your stance: {agent_config.get('stance', 'neutral')}; sentiment bias: {agent_config.get('sentiment_bias', 0)}; influence weight: {agent_config.get('influence_weight', 1)}.",
        f"Expected behavior: posts/hour={agent_config.get('posts_per_hour', 0.5)}, comments/hour={agent_config.get('comments_per_hour', 1.0)}, response delay={agent_config.get('response_delay_min', 5)}-{agent_config.get('response_delay_max', 60)} minutes.",
        "Longer memory summary: " + (summary.get("summary") or "No compacted summary yet."),
        "Short-term memory: " + (" | ".join(memory_lines) if memory_lines else "No recent actions yet."),
        economy_context,
        f"Current event shock: {event_context}" if event_context else "",
        "Act only if you can add forecast signal. Avoid generic greetings or public-relations filler.",
    ]).strip()

    try:
        message = BaseMessage.make_user_message(role_name="system", content=content)
        agent.update_memory(message=message, role=OpenAIBackendRole.SYSTEM)
    except Exception:
        pass


def _economy_context(simulation_dir: str, agent_id: int) -> str:
    """Load the canonical wallet, jobs, and events for the next social action."""
    db_path = os.path.join(simulation_dir, "economy.db")
    if not os.path.isfile(db_path):
        return ""
    try:
        with sqlite3.connect(db_path, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            identity = conn.execute(
                "SELECT economic_agent_id FROM economic_agents WHERE oasis_agent_id=?",
                (int(agent_id),),
            ).fetchone()
            if not identity:
                return ""
            economic_id = identity["economic_agent_id"]
            balance = conn.execute(
                """SELECT COALESCE(SUM(l.amount_cents), 0) AS balance
                   FROM accounts a LEFT JOIN ledger_entries l ON l.account_id=a.account_id
                   WHERE a.owner_agent_id=?""",
                (economic_id,),
            ).fetchone()["balance"]
            jobs = conn.execute(
                """SELECT title, status, wage_cents FROM jobs
                   WHERE employer_id=? OR worker_id=? ORDER BY updated_tick DESC LIMIT 3""",
                (economic_id, economic_id),
            ).fetchall()
            events = conn.execute(
                """SELECT event_type, amount_cents FROM economy_events
                   WHERE actor_id=? OR counterparty_id=? ORDER BY tick DESC, created_at DESC LIMIT 3""",
                (economic_id, economic_id),
            ).fetchall()
        job_text = "; ".join(
            f"{row['title']} ({row['status']}, {row['wage_cents']} cents)" for row in jobs
        ) or "none"
        event_text = "; ".join(
            f"{row['event_type']} ({row['amount_cents']} cents)" for row in events
        ) or "none"
        return (
            f"[Economic Twin] Canonical balance: {balance} cents. "
            f"Your jobs: {job_text}. Recent economic events: {event_text}. "
            "Keep social claims consistent with these settled facts."
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return ""


def _agent_config(config: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    for cfg in config.get("agent_configs", []):
        if cfg.get("agent_id") == agent_id:
            return cfg
    return {}


def _forecast_context(config: Dict[str, Any]) -> str:
    mode = config.get("forecast_mode", "general")
    horizon = config.get("forecast_horizon", "medium_term")
    target = config.get("prediction_target") or {}
    outcomes = target.get("outcomes") or []
    lines = [
        f"Mode={mode}; horizon={horizon}.",
        f"Prediction target={target.get('question', config.get('simulation_requirement', 'future outcome'))}.",
    ]
    if outcomes:
        lines.append("Outcomes=" + "; ".join(str(outcome) for outcome in outcomes))
    return " ".join(lines)
