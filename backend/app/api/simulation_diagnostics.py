"""Read-only diagnostics endpoints for completed and running simulations."""

import json
import tempfile
import traceback

from flask import jsonify, request, send_file

from . import simulation_bp
from ..services.simulation_runner import SimulationRunner
from ..utils.logger import get_logger


logger = get_logger("mirofish.api.simulation.diagnostics")


def _failure(message: str, exc: Exception):
    logger.error("%s: %s", message, exc)
    return jsonify({
        "success": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }), 500


@simulation_bp.route('/<simulation_id>/diagnostics', methods=['GET'])
def get_run_diagnostics(simulation_id: str):
    try:
        return jsonify({
            "success": True,
            "data": SimulationRunner.get_run_diagnostics(simulation_id),
        })
    except Exception as exc:
        return _failure("Failed to get run diagnostics", exc)


@simulation_bp.route('/<simulation_id>/diagnostics/agents', methods=['GET'])
def get_agent_diagnostics(simulation_id: str):
    try:
        diagnostics = SimulationRunner.get_run_diagnostics(simulation_id)
        agents = diagnostics.get("agents", [])
        if request.args.get("high_impact", "false").lower() == "true":
            agents = diagnostics.get("high_impact_agents", [])
        if request.args.get("off_track", "false").lower() == "true":
            agents = diagnostics.get("off_track_agents", [])
        return jsonify({"success": True, "data": {"agents": agents, "count": len(agents)}})
    except Exception as exc:
        return _failure("Failed to get agent diagnostics", exc)


@simulation_bp.route('/<simulation_id>/diagnostics/topics', methods=['GET'])
def get_topic_diagnostics(simulation_id: str):
    try:
        diagnostics = SimulationRunner.get_run_diagnostics(simulation_id)
        return jsonify({"success": True, "data": diagnostics.get("topics", {})})
    except Exception as exc:
        return _failure("Failed to get topic diagnostics", exc)


@simulation_bp.route('/<simulation_id>/diagnostics/off-track', methods=['GET'])
def get_off_track_agents(simulation_id: str):
    try:
        diagnostics = SimulationRunner.get_run_diagnostics(simulation_id)
        agents = diagnostics.get("off_track_agents", [])
        return jsonify({"success": True, "data": {"agents": agents, "count": len(agents)}})
    except Exception as exc:
        return _failure("Failed to get off-track agents", exc)


@simulation_bp.route('/<simulation_id>/diagnostics/export', methods=['GET'])
def export_run_diagnostics(simulation_id: str):
    try:
        diagnostics = SimulationRunner.get_run_diagnostics(simulation_id)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as handle:
            json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
            path = handle.name
        return send_file(path, as_attachment=True, download_name=f"{simulation_id}-diagnostics.json")
    except Exception as exc:
        return _failure("Failed to export diagnostics", exc)
