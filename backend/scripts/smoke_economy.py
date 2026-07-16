"""Live Ollama smoke gate for Economic Twin v1."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shutil
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.config import Config
from app.services.economy import EconomyRuntime, EconomyTickCoordinator


PROFILES = [
    {
        "agent_id": 1,
        "entity_uuid": "smoke-employer",
        "entity_name": "Smoke Employer",
        "entity_type": "company",
    },
    {
        "agent_id": 2,
        "entity_uuid": "smoke-worker",
        "entity_name": "Smoke Worker",
        "entity_type": "person",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "ollama"))
    return parser.parse_args()


async def run_smoke(args: argparse.Namespace, root: str) -> dict:
    Config.OASIS_SIMULATION_DATA_DIR = root
    Config.LLM_API_KEY = args.api_key
    Config.LLM_BASE_URL = args.base_url
    Config.LLM_MODEL_NAME = args.model
    config = {
        "simulation_id": "economic-twin-live-smoke",
        "llm_base_url": args.base_url,
        "llm_model": args.model,
        "time_config": {"minutes_per_round": 1440},
        "agent_configs": PROFILES,
        "economy": {
            "enabled": True,
            "rounds_per_tick": 1,
            "initial_balance_cents": 100000,
            "currency": "USD",
            "max_decisions_per_tick": len(PROFILES),
        },
    }
    runtime = EconomyRuntime("economic-twin-live-smoke", config)
    coordinator = EconomyTickCoordinator(runtime, {"twitter"})
    outcome = await coordinator.after_round("twitter", 1)
    if not outcome or outcome.get("error"):
        raise AssertionError(f"economic tick did not complete: {outcome}")
    results = outcome.get("results") or []
    if len(results) != len(PROFILES):
        raise AssertionError(
            f"expected {len(PROFILES)} decision results, received {len(results)}"
        )
    format_rejections = [
        result
        for result in results
        if "decision provider error" in str(result.get("rejection_reason") or "").lower()
        or "invalid json" in str(result.get("rejection_reason") or "").lower()
        or "no json content" in str(result.get("rejection_reason") or "").lower()
    ]
    if format_rejections:
        raise AssertionError(f"provider-format rejection: {format_rejections}")
    runtime.store.assert_invariants()
    summary = runtime.store.summary()
    if summary["ticks"]["completed"] != 1 or summary["ticks"]["running"] != 0:
        raise AssertionError(f"unexpected tick summary: {summary['ticks']}")
    return {
        "model": args.model,
        "tick": outcome["tick"],
        "results": len(results),
        "statuses": [result.get("status") for result in results],
        "rejection_reasons": [
            result.get("rejection_reason")
            for result in results
            if result.get("rejection_reason")
        ],
        "ticks": summary["ticks"],
        "rejected_intents": summary["rejected_intents"],
        "ledger_balanced": True,
    }


def main() -> int:
    args = parse_args()
    root = tempfile.mkdtemp(prefix="mirofish-economy-smoke-")
    try:
        result = asyncio.run(run_smoke(args, root))
        gc.collect()
    finally:
        shutil.rmtree(root)
    if os.path.exists(root):
        raise AssertionError(f"temporary database directory was not removed: {root}")
    result["temporary_database_cleanup"] = "passed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
