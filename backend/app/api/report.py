"""
Report API Routes
Provides interfaces for simulation report generation, retrieval, and conversation
"""

import os
import traceback
import threading
from flask import request, jsonify, send_file, current_app

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus, normalize_report_mode, normalize_report_depth
from ..services.forecasting import (
    infer_report_mode_from_forecast,
    load_simulation_config,
    normalize_forecast_settings,
)
from ..services.simulation_manager import SimulationManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..services.graph_tools import GraphToolsService
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.report')


def _json_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _record_case_snapshot(report):
    """Persist a prediction snapshot onto the report's case version (if the report's
    simulation belongs to a case) so cross-version prediction comparison has real data.

    Best-effort: a storage error here must never fail an otherwise-successful report.
    """
    try:
        if not report or not getattr(report, "forecast", None):
            return
        state = SimulationManager().get_simulation(report.simulation_id)
        if not state or not getattr(state, "project_id", None):
            return
        project = ProjectManager.get_project(state.project_id)
        if not project or not project.case_id or not project.case_version_id:
            return
        from ..models.case import CaseManager, build_prediction_snapshot
        snapshot = build_prediction_snapshot(
            report.forecast,
            report_id=report.report_id,
            simulation_id=report.simulation_id,
        )
        CaseManager.record_prediction_snapshot(
            project.case_id,
            project.case_version_id,
            snapshot,
            report_id=report.report_id,
            simulation_id=report.simulation_id,
        )
        logger.info(
            "Recorded prediction snapshot: case=%s version=%s report=%s",
            project.case_id, project.case_version_id, report.report_id,
        )
    except Exception as exc:
        logger.warning("Failed to record case prediction snapshot: %s", exc)


# ============== Report Generation Interface ==============

@report_bp.route('/generate', methods=['POST'])
def generate_report():
    try:
        data = request.get_json() or {}
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({"success": False, "error": "Please provide simulation_id"}), 400

        force_regenerate = data.get('force_regenerate', False)
        requested_report_mode = data.get('report_mode')
        report_mode = normalize_report_mode(requested_report_mode)
        requested_report_depth = data.get('report_depth')
        report_depth = normalize_report_depth(requested_report_depth)
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": f"Simulation does not exist: {simulation_id}"}), 404

        simulation_config = load_simulation_config(simulation_id)
        if not requested_report_depth:
            report_depth = normalize_report_depth(
                simulation_config.get("report_depth")
                or (simulation_config.get("prediction_settings") or {}).get("report_depth")
            )
        if not requested_report_mode:
            forecast_settings = normalize_forecast_settings(
                simulation_config,
                simulation_requirement=simulation_config.get("simulation_requirement", ""),
            )
            report_mode = normalize_report_mode(
                infer_report_mode_from_forecast(forecast_settings.forecast_mode)
            )

        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if (
                existing_report
                and existing_report.report_mode == report_mode
                and existing_report.report_depth == report_depth
                and existing_report.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]
            ):
                existing_issues = ReportManager.validate_report_output(existing_report)
                blocking_issues = [issue for issue in existing_issues if issue.get("blocking")]
                if blocking_issues:
                    return jsonify({"success": False, "error": "Existing report has validation issues. Regenerate it with force_regenerate=true.", "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "validation_issues": existing_issues,
                    }}), 409
                return jsonify({"success": True, "data": {
                    "simulation_id": simulation_id,
                    "report_id": existing_report.report_id,
                    "status": existing_report.status.value,
                    "report_mode": existing_report.report_mode,
                    "report_depth": existing_report.report_depth,
                    "message": "Report already exists",
                    "already_generated": True
                }})

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Project does not exist: {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "Missing graph ID, please ensure graph is built"}), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({"success": False, "error": "Missing simulation requirement description"}), 400

        import uuid
        report_id = f"report_{uuid.uuid4().hex[:12]}"

        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id,
                "report_mode": report_mode,
                "report_depth": report_depth,
            }
        )

        # Initialize graph_tools in Flask context BEFORE spawning thread
        # (current_app is not available inside background threads)
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            return jsonify({"success": False, "error": "GraphStorage not initialized - check Neo4j connection"}), 500

        embedding_status = storage.get_embedding_status(graph_id)
        allow_degraded = _json_bool(data.get("allow_degraded"), False)
        if not embedding_status.get("safe_to_report") and not allow_degraded:
            return jsonify({
                "success": False,
                "error": "Graph is not ready for semantic report generation. Rebuild or re-embed the graph first.",
                "data": {
                    "graph_id": graph_id,
                    "embedding_status": embedding_status,
                    "hint": f"POST /api/graph/{graph_id}/reembed or run scripts/reembed_graph.py --graph-id {graph_id}"
                }
            }), 409

        graph_tools = GraphToolsService(storage=storage)

        def run_generate():
            try:
                task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0, message="Initializing Report Agent...")
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    graph_tools=graph_tools,
                    disable_interviews=_json_bool(data.get("disable_interviews"), False),
                    strict_antirepetition=_json_bool(data.get("strict_antirepetition"), False),
                    report_mode=report_mode,
                    report_depth=report_depth,
                )
                def progress_callback(stage, progress, message):
                    task_manager.update_task(task_id, progress=progress, message=f"[{stage}] {message}")
                report = agent.generate_report(progress_callback=progress_callback, report_id=report_id)
                ReportManager.save_report(report)
                if report.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]:
                    _record_case_snapshot(report)
                    task_manager.complete_task(task_id, result={
                        "report_id": report.report_id,
                        "simulation_id": simulation_id,
                        "status": report.status.value,
                        "forecast": report.forecast,
                        "timing": report.timing,
                    })
                else:
                    task_manager.fail_task(task_id, report.error or "Report generation failed")
            except Exception as e:
                logger.error(f"Report generation failed: {str(e)}")
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "report_id": report_id,
            "task_id": task_id,
            "status": "generating",
            "report_mode": report_mode,
            "report_depth": report_depth,
            "message": "Report generation task started. Query progress via /api/report/generate/status",
            "already_generated": False
        }})

    except Exception as e:
        logger.error(f"Failed to start report generation task: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/generate/status', methods=['POST'])
def get_generate_status():
    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')

        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]:
                return jsonify({"success": True, "data": {
                    "simulation_id": simulation_id,
                    "report_id": existing_report.report_id,
                    "status": existing_report.status.value,
                    "progress": 100,
                    "message": "Report generated",
                    "already_completed": True
                }})

        if not task_id:
            return jsonify({"success": False, "error": "Please provide task_id or simulation_id"}), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({"success": False, "error": f"Task does not exist: {task_id}"}), 404

        return jsonify({"success": True, "data": task.to_dict()})

    except Exception as e:
        logger.error(f"Failed to query task status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Report Retrieval Interface ==============

@report_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Report does not exist: {report_id}"}), 404
        return jsonify({"success": True, "data": report.to_dict()})
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/by-simulation/<simulation_id>', methods=['GET'])
def get_report_by_simulation(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        if not report:
            return jsonify({"success": False, "error": f"No report available for this simulation: {simulation_id}", "has_report": False}), 404
        return jsonify({"success": True, "data": report.to_dict()})
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/list', methods=['GET'])
def list_reports():
    try:
        simulation_id = request.args.get('simulation_id')
        limit = request.args.get('limit', 50, type=int)
        reports = ReportManager.list_reports(simulation_id=simulation_id, limit=limit)
        return jsonify({"success": True, "data": [r.to_dict() for r in reports], "count": len(reports)})
    except Exception as e:
        logger.error(f"Failed to list reports: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Report does not exist: {report_id}"}), 404

        md_path = ReportManager._get_report_markdown_path(report_id)
        if not os.path.exists(md_path):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name
            return send_file(temp_path, as_attachment=True, download_name=f"{report_id}.md")

        return send_file(md_path, as_attachment=True, download_name=f"{report_id}.md")

    except Exception as e:
        logger.error(f"Failed to download report: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id: str):
    try:
        success = ReportManager.delete_report(report_id)
        if not success:
            return jsonify({"success": False, "error": f"Report does not exist: {report_id}"}), 404
        return jsonify({"success": True, "message": f"Report deleted: {report_id}"})
    except Exception as e:
        logger.error(f"Failed to delete report: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Report Agent Chat Interface ==============

@report_bp.route('/chat', methods=['POST'])
def chat_with_report_agent():
    try:
        data = request.get_json() or {}
        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])

        if not simulation_id:
            return jsonify({"success": False, "error": "Please provide simulation_id"}), 400
        if not message:
            return jsonify({"success": False, "error": "Please provide message"}), 400

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": f"Simulation does not exist: {simulation_id}"}), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Project does not exist: {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "Missing graph ID"}), 400

        simulation_requirement = project.simulation_requirement or ""

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized - check Neo4j connection")
        graph_tools = GraphToolsService(storage=storage)
        existing_report = ReportManager.get_report_by_simulation(simulation_id)

        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
            graph_tools=graph_tools,
            report_mode=existing_report.report_mode if existing_report else "prediction",
        )

        result = agent.chat(message=message, chat_history=chat_history)
        return jsonify({"success": True, "data": {"response": result, "simulation_id": simulation_id}})

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Report Progress and Section Retrieval Interface ==============

@report_bp.route('/<report_id>/progress', methods=['GET'])
def get_report_progress(report_id: str):
    try:
        progress = ReportManager.get_progress(report_id)
        if not progress:
            return jsonify({"success": False, "error": f"Report does not exist or progress info unavailable: {report_id}"}), 404
        return jsonify({"success": True, "data": progress})
    except Exception as e:
        logger.error(f"Failed to get report progress: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/sections', methods=['GET'])
def get_report_sections(report_id: str):
    try:
        sections = ReportManager.get_generated_sections(report_id)
        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]
        return jsonify({"success": True, "data": {
            "report_id": report_id,
            "sections": sections,
            "total": len(sections),
            "is_complete": is_complete
        }})
    except Exception as e:
        logger.error(f"Failed to get section list: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/section/<int:section_index>', methods=['GET'])
def get_single_section(report_id: str, section_index: int):
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)
        if not os.path.exists(section_path):
            return jsonify({"success": False, "error": f"Section does not exist: section_{section_index:02d}.md"}), 404
        with open(section_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"success": True, "data": {"filename": f"section_{section_index:02d}.md", "content": content}})
    except Exception as e:
        logger.error(f"Failed to get section content: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/section/<int:section_index>/regenerate', methods=['POST'])
def regenerate_report_section(report_id: str, section_index: int):
    try:
        data = request.get_json() or {}
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Report does not exist: {report_id}"}), 404

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized - check Neo4j connection")

        disable_interviews = _json_bool(data.get("disable_interviews"), False)
        strict_antirepetition = _json_bool(data.get("strict_antirepetition"), True)

        agent = ReportAgent(
            graph_id=report.graph_id,
            simulation_id=report.simulation_id,
            simulation_requirement=report.simulation_requirement,
            graph_tools=GraphToolsService(storage=storage),
            disable_interviews=disable_interviews,
            strict_antirepetition=strict_antirepetition,
            report_mode=report.report_mode,
        )
        updated_report = agent.regenerate_section(
            report=report,
            section_index=section_index,
            disable_interviews=disable_interviews,
            strict_antirepetition=strict_antirepetition,
        )
        return jsonify({"success": True, "data": updated_report.to_dict()})
    except Exception as e:
        logger.error(f"Failed to regenerate section: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/resume', methods=['POST'])
def resume_report(report_id: str):
    try:
        data = request.get_json() or {}
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Report does not exist: {report_id}"}), 404
        progress = ReportManager.get_progress(report_id) or {}
        resumable_statuses = [
            ReportStatus.FAILED,
            ReportStatus.GENERATING,
            ReportStatus.PENDING,
            ReportStatus.INTERRUPTED,
            ReportStatus.STALE,
        ]
        if report.status not in resumable_statuses and not progress.get("is_stale"):
            return jsonify({"success": False, "error": f"Report cannot be resumed from status: {report.status.value}"}), 409

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized - check Neo4j connection")

        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_resume",
            metadata={"simulation_id": report.simulation_id, "graph_id": report.graph_id, "report_id": report_id}
        )

        def run_resume():
            try:
                task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0, message="Resuming report...")
                agent = ReportAgent(
                    graph_id=report.graph_id,
                    simulation_id=report.simulation_id,
                    simulation_requirement=report.simulation_requirement,
                    graph_tools=GraphToolsService(storage=storage),
                    disable_interviews=_json_bool(data.get("disable_interviews"), False),
                    strict_antirepetition=_json_bool(data.get("strict_antirepetition"), True),
                    report_mode=report.report_mode,
                )

                def progress_callback(stage, progress, message):
                    task_manager.update_task(task_id, progress=progress, message=f"[{stage}] {message}")

                resumed = agent.resume_report(report, progress_callback=progress_callback)
                if resumed.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]:
                    _record_case_snapshot(resumed)
                    task_manager.complete_task(task_id, result={
                        "report_id": resumed.report_id,
                        "simulation_id": resumed.simulation_id,
                        "status": resumed.status.value,
                        "forecast": resumed.forecast,
                    })
                else:
                    task_manager.fail_task(task_id, resumed.error or "Report resume failed")
            except Exception as e:
                logger.error(f"Report resume failed: {str(e)}")
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=run_resume, daemon=True)
        thread.start()

        return jsonify({"success": True, "data": {
            "simulation_id": report.simulation_id,
            "report_id": report_id,
            "task_id": task_id,
            "status": "generating",
            "message": "Report resume task started"
        }})
    except Exception as e:
        logger.error(f"Failed to resume report: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Report Status Check Interface ==============

@report_bp.route('/check/<simulation_id>', methods=['GET'])
def check_report_status(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        has_report = report is not None
        report_status = report.status.value if report and hasattr(report.status, 'value') else (report.status if report else None)
        report_id = report.report_id if report else None
        interview_unlocked = has_report and report.status in [ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW]
        progress = ReportManager.get_progress(report_id) if report_id else None
        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "has_report": has_report,
            "report_id": report_id,
            "report_status": report_status,
            "interview_unlocked": interview_unlocked,
            "progress": progress
        }})
    except Exception as e:
        logger.error(f"Failed to check report status: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Agent Log Interface ==============

@report_bp.route('/<report_id>/agent-log', methods=['GET'])
def get_agent_log(report_id: str):
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)
        return jsonify({"success": True, "data": log_data})
    except Exception as e:
        logger.error(f"Failed to get agent log: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/agent-log/stream', methods=['GET'])
def stream_agent_log(report_id: str):
    try:
        logs = ReportManager.get_agent_log_stream(report_id)
        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})
    except Exception as e:
        logger.error(f"Failed to get agent log: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Console Log Interface ==============

@report_bp.route('/<report_id>/console-log', methods=['GET'])
def get_console_log(report_id: str):
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_console_log(report_id, from_line=from_line)
        return jsonify({"success": True, "data": log_data})
    except Exception as e:
        logger.error(f"Failed to get console log: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/console-log/stream', methods=['GET'])
def stream_console_log(report_id: str):
    try:
        logs = ReportManager.get_console_log_stream(report_id)
        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})
    except Exception as e:
        logger.error(f"Failed to get console log: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Tool Call Interface (For Debugging) ==============

@report_bp.route('/tools/search', methods=['POST'])
def search_graph_tool():
    try:
        data = request.get_json() or {}
        graph_id = data.get('graph_id')
        query = data.get('query')
        limit = data.get('limit', 10)
        if not graph_id or not query:
            return jsonify({"success": False, "error": "Please provide graph_id and query"}), 400
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized — check Neo4j connection")
        tools = GraphToolsService(storage=storage)
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)
        return jsonify({"success": True, "data": result.to_dict()})
    except Exception as e:
        logger.error(f"Graph search failed: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/tools/statistics', methods=['POST'])
def get_graph_statistics_tool():
    try:
        data = request.get_json() or {}
        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({"success": False, "error": "Please provide graph_id"}), 400
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized — check Neo4j connection")
        tools = GraphToolsService(storage=storage)
        result = tools.get_graph_statistics(graph_id)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500
