"""
Graph-related API Routes
Uses project context mechanism with server-side state persistence
"""

import os
import traceback
import threading
import time
from flask import request, jsonify, current_app

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services.embedding_benchmark import DEFAULT_QUERIES, run_embedding_benchmark
from ..services.text_processor import TextProcessor
from ..storage.extraction_cache import ExtractionCache
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..utils.timing import PhaseTimer
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus
from ..models.case import (
    CaseManager,
    classify_document_type,
    normalize_graph_build_settings,
    normalize_prediction_settings,
)

# Get logger
logger = get_logger('mirofish.api')


def _get_storage():
    """Get Neo4jStorage from Flask app extensions."""
    storage = current_app.extensions.get('neo4j_storage')
    if not storage:
        raise ValueError("GraphStorage not initialized — check Neo4j connection")
    return storage


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


# ============== Project Management Interface ==============

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    Get project details
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Project does not exist: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    List all projects
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    Delete project
    """
    success = ProjectManager.delete_project(project_id)

    if not success:
        return jsonify({
            "success": False,
            "error": f"Project does not exist or deletion failed: {project_id}"
        }), 404

    return jsonify({
        "success": True,
        "message": f"Project deleted: {project_id}"
    })


# ============== Case Management Interface ==============

@graph_bp.route('/case/create', methods=['POST'])
def create_case():
    """Create a top-level litigation case."""
    try:
        data = request.get_json(silent=True) or {}
        name = data.get("name") or data.get("case_name") or "Untitled Case"
        simulation_requirement = data.get("simulation_requirement", "")
        case = CaseManager.create_case(
            name=name,
            simulation_requirement=simulation_requirement,
            tags=data.get("tags") or [],
        )
        return jsonify({"success": True, "data": case.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@graph_bp.route('/case/list', methods=['GET'])
def list_cases():
    """List cases."""
    try:
        limit = request.args.get('limit', 50, type=int)
        cases = CaseManager.list_cases(limit=limit)
        return jsonify({
            "success": True,
            "data": [case.to_dict() for case in cases],
            "count": len(cases),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@graph_bp.route('/case/<case_id>', methods=['GET'])
def get_case(case_id: str):
    """Get case details."""
    case = CaseManager.get_case(case_id)
    if not case:
        return jsonify({"success": False, "error": f"Case does not exist: {case_id}"}), 404
    return jsonify({"success": True, "data": case.to_dict()})


@graph_bp.route('/case/<case_id>/versions', methods=['GET'])
def list_case_versions(case_id: str):
    """List versions for a case."""
    case = CaseManager.get_case(case_id)
    if not case:
        return jsonify({"success": False, "error": f"Case does not exist: {case_id}"}), 404
    return jsonify({
        "success": True,
        "data": [version.to_dict() for version in case.versions],
        "count": len(case.versions),
    })


@graph_bp.route('/case/<case_id>/compare', methods=['POST'])
def compare_case_versions(case_id: str):
    """Compare prediction snapshots across two case versions."""
    try:
        data = request.get_json(silent=True) or {}
        from_version_id = data.get("from_version_id")
        to_version_id = data.get("to_version_id")
        if not from_version_id or not to_version_id:
            return jsonify({
                "success": False,
                "error": "Please provide from_version_id and to_version_id",
            }), 400
        return jsonify({
            "success": True,
            "data": CaseManager.compare_prediction_versions(case_id, from_version_id, to_version_id),
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@graph_bp.route('/case/<case_id>/documents', methods=['POST'])
def add_documents_to_case(case_id: str):
    """
    Add a document batch to an existing case and create a new case version.

    This endpoint records the version and extracted text. The existing
    ontology/generate endpoint remains the full analyze-and-build entry point.
    """
    try:
        case = CaseManager.get_case(case_id)
        if not case:
            return jsonify({"success": False, "error": f"Case does not exist: {case_id}"}), 404

        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({"success": False, "error": "Please upload at least one document file"}), 400

        project = ProjectManager.create_project(name=request.form.get("project_name", case.name))
        project.case_id = case_id
        project.simulation_requirement = request.form.get("simulation_requirement") or case.simulation_requirement

        document_texts = []
        all_text = ""
        for file in uploaded_files:
            if not file or not file.filename or not allowed_file(file.filename):
                continue
            file_info = ProjectManager.save_file_to_project(project.project_id, file, file.filename)
            text = TextProcessor.preprocess_text(FileParser.extract_text(file_info["path"]))
            document_texts.append({
                "filename": file_info["original_filename"],
                "path": file_info["path"],
                "size": file_info["size"],
                "text": text,
                "document_type": classify_document_type(file_info["original_filename"], text).value,
            })
            all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"

        if not document_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify({"success": False, "error": "No documents successfully processed"}), 400

        version = CaseManager.create_version(
            case_id=case_id,
            project_id=project.project_id,
            documents=document_texts,
            build_settings=normalize_graph_build_settings(request.form.to_dict()),
            prediction_settings=normalize_prediction_settings(request.form.to_dict()),
        )

        stored_docs = []
        for raw, case_doc in zip(document_texts, version.documents):
            stored_docs.append({
                **raw,
                "document_id": case_doc.document_id,
                "case_id": case_id,
                "version_id": version.version_id,
                "hash": case_doc.hash,
            })

        project.documents = [
            {key: value for key, value in item.items() if key != "text"}
            for item in stored_docs
        ]
        project.files = [
            {"filename": item["filename"], "path": item.get("path"), "size": item.get("size", 0)}
            for item in stored_docs
        ]
        project.total_text_length = len(all_text)
        project.case_version_id = version.version_id
        project.case_version_number = version.version_number
        project.graph_build_settings = version.build_settings
        project.prediction_settings = version.prediction_settings
        ProjectManager.save_extracted_text(project.project_id, all_text)
        ProjectManager.save_document_texts(project.project_id, stored_docs)
        ProjectManager.save_project(project)

        return jsonify({
            "success": True,
            "data": {
                "case": CaseManager.get_case(case_id).to_dict(),
                "version": version.to_dict(),
                "project": project.to_dict(),
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    Reset project status (for rebuilding graph)
    """
    project = ProjectManager.get_project(project_id)

    if not project:
        return jsonify({
            "success": False,
            "error": f"Project does not exist: {project_id}"
        }), 404

    # Reset to ontology generated state
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED

    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)

    return jsonify({
        "success": True,
        "message": f"Project reset: {project_id}",
        "data": project.to_dict()
    })


# ============== Interface 1: Upload Files and Generate Ontology ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    Interface 1: Upload files and analyze to generate ontology definition

    Request method: multipart/form-data

    Parameters:
        files: Uploaded files (PDF/MD/TXT), multiple allowed
        simulation_requirement: Simulation requirement description (required)
        project_name: Project name (optional)
        additional_context: Additional notes (optional)

    Response:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {
                    "entity_types": [...],
                    "edge_types": [...],
                    "analysis_summary": "..."
                },
                "files": [...],
                "total_text_length": 12345
            }
        }
    """
    try:
        logger.info("=== Starting ontology generation ===")
        timing = PhaseTimer("ontology_generation")
        phase_profile = {
            "pdf_extraction_seconds": 0.0,
            "ontology_seconds": 0.0,
            "total_seconds": 0.0,
        }

        # Get parameters
        simulation_requirement = request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Unnamed Project')
        case_id = request.form.get('case_id')
        additional_context = request.form.get('additional_context', '')

        logger.debug(f"Project name: {project_name}")
        logger.debug(f"Simulation requirement: {simulation_requirement[:100]}...")

        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Please provide simulation requirement description (simulation_requirement)"
            }), 400

        # Get uploaded files
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({
                "success": False,
                "error": "Please upload at least one document file"
            }), 400

        # Create or load case
        case = CaseManager.get_case(case_id) if case_id else None
        if case_id and not case:
            return jsonify({
                "success": False,
                "error": f"Case does not exist: {case_id}"
            }), 404
        if not case:
            case = CaseManager.create_case(
                name=project_name,
                simulation_requirement=simulation_requirement,
            )
        case_id = case.case_id

        # Create project
        project = ProjectManager.create_project(name=project_name)
        project.case_id = case_id
        project.simulation_requirement = simulation_requirement
        logger.info(f"Project created: {project.project_id}")
        
        # Save files and extract text
        ontology_texts = []
        document_payloads = []
        all_text = ""

        with timing.time("document_extraction", label="Save files and extract text", metadata={"file_count": len(uploaded_files)}):
            for file in uploaded_files:
                if file and file.filename and allowed_file(file.filename):
                    # Save file to project directory
                    file_info = ProjectManager.save_file_to_project(
                        project.project_id,
                        file,
                        file.filename
                    )
                    # Extract text
                    text = FileParser.extract_text(file_info["path"])
                    text = TextProcessor.preprocess_text(text)
                    document_type = classify_document_type(file_info["original_filename"], text).value
                    ontology_texts.append(text)
                    document_payloads.append({
                        "filename": file_info["original_filename"],
                        "path": file_info["path"],
                        "size": file_info["size"],
                        "text": text,
                        "document_type": document_type,
                    })
                    all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"
        phase_profile["pdf_extraction_seconds"] = timing.snapshot()["phase_totals"].get("document_extraction", 0.0)

        if not ontology_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": "No documents successfully processed. Please check file format"
            }), 400

        with timing.time("case_version", label="Create case version", metadata={"document_count": len(document_payloads)}):
            version = CaseManager.create_version(
                case_id=case_id,
                project_id=project.project_id,
                documents=document_payloads,
                build_settings=normalize_graph_build_settings(request.form.to_dict()),
                prediction_settings=normalize_prediction_settings(request.form.to_dict()),
            )
        stored_documents = []
        for raw, case_doc in zip(document_payloads, version.documents):
            stored_documents.append({
                **raw,
                "document_id": case_doc.document_id,
                "case_id": case_id,
                "version_id": version.version_id,
                "hash": case_doc.hash,
            })
        project.case_version_id = version.version_id
        project.case_version_number = version.version_number
        project.documents = [
            {key: value for key, value in item.items() if key != "text"}
            for item in stored_documents
        ]
        project.files = [
            {"filename": item["filename"], "path": item.get("path"), "size": item.get("size", 0)}
            for item in stored_documents
        ]
        project.graph_build_settings = version.build_settings
        project.prediction_settings = version.prediction_settings

        # Save extracted text
        project.total_text_length = len(all_text)
        with timing.time("persist_documents", label="Persist extracted document text"):
            ProjectManager.save_extracted_text(project.project_id, all_text)
            ProjectManager.save_document_texts(project.project_id, stored_documents)
        logger.info(f"Text extraction completed, total {len(all_text)} characters")

        # Generate ontology
        logger.info("Calling LLM to generate ontology definition...")
        generator = OntologyGenerator()
        with timing.time("ontology_llm", label="Generate ontology with LLM", metadata={"document_count": len(ontology_texts)}):
            ontology = generator.generate(
                document_texts=ontology_texts,
                simulation_requirement=simulation_requirement,
                additional_context=additional_context if additional_context else None
            )
        phase_profile["ontology_seconds"] = timing.snapshot()["phase_totals"].get("ontology_llm", 0.0)

        # Save ontology to project
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(f"Ontology generation completed: {entity_count} entity types, {edge_count} relation types")
        
        with timing.time("persist_ontology", label="Persist ontology metadata"):
            project.ontology = {
                "entity_types": ontology.get("entity_types", []),
                "edge_types": ontology.get("edge_types", [])
            }
            project.analysis_summary = ontology.get("analysis_summary", "")
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            ProjectManager.save_project(project)
            CaseManager.update_version(
                project.case_id,
                project.case_version_id,
                status="ontology_generated",
                prediction_settings=project.prediction_settings,
                build_settings=project.graph_build_settings,
            )
        timing_summary = timing.complete(metadata={
            "project_id": project.project_id,
            "case_id": project.case_id,
            "case_version_id": project.case_version_id,
            "document_count": len(stored_documents),
            "entity_type_count": entity_count,
            "edge_type_count": edge_count,
        })
        phase_profile["total_seconds"] = timing_summary["total_seconds"]
        phase_profile["timing"] = timing_summary
        project.timings = {**(project.timings or {}), "ontology_generation": timing_summary}
        ProjectManager.save_project(project)
        logger.info(f"=== Ontology generation completed === Project ID: {project.project_id}")
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "project_name": project.name,
                "case_id": project.case_id,
                "case_version_id": project.case_version_id,
                "case_version_number": project.case_version_number,
                "ontology": project.ontology,
                "analysis_summary": project.analysis_summary,
                "files": project.files,
                "documents": project.documents,
                "total_text_length": project.total_text_length,
                "profile": phase_profile,
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface 2: Build Graph ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """
    Interface 2: Build graph based on project_id

    Request (JSON):
        {
            "project_id": "proj_xxxx",  // Required: from interface 1
            "graph_name": "Graph name",    // Optional
            "chunk_size": 3000,         // Optional, default 3000
            "chunk_overlap": 200        // Optional, default 200
        }

    Response:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "Graph build task started"
            }
        }
    """
    try:
        logger.info("=== Starting graph build ===")

        # Parse request
        data = request.get_json() or {}
        project_id = data.get('project_id')
        logger.debug(f"Request parameters: project_id={project_id}")
        
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Please provide project_id"
            }), 400

        # Get project
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Project does not exist: {project_id}"
            }), 404

        # Check project status
        force = data.get('force', False)  # Force rebuild

        if project.status == ProjectStatus.CREATED:
            return jsonify({
                "success": False,
                "error": "Project has not generated ontology yet. Please call /ontology/generate first"
            }), 400

        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify({
                "success": False,
                "error": "Graph is being built. Do not submit repeatedly. To force rebuild, add force: true",
                "task_id": project.graph_build_task_id
            }), 400

        # If force rebuild, reset status
        if force and project.status in [ProjectStatus.GRAPH_BUILDING, ProjectStatus.FAILED, ProjectStatus.GRAPH_COMPLETED]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            project.error = None

        # Get configuration
        graph_name = data.get('graph_name', project.name or 'MiroFish Graph')
        request_settings = {**(project.graph_build_settings or {}), **data}
        build_settings = GraphBuilderService.resolve_build_settings(request_settings)
        chunk_size = build_settings["chunk_size"]
        chunk_overlap = build_settings["chunk_overlap"]
        batch_size = build_settings["batch_size"]
        build_mode = build_settings["build_mode"]
        llm_concurrency_override = build_settings.get("llm_concurrency")
        incremental_mode = bool(build_settings.get("incremental"))

        # Update project configuration
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap
        project.graph_build_settings = build_settings

        # Get extracted text
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify({
                "success": False,
                "error": "Extracted text not found"
            }), 400
        document_texts = ProjectManager.get_document_texts(project_id)
        if document_texts:
            for document in document_texts:
                document.setdefault("case_id", project.case_id)
                document.setdefault("version_id", project.case_version_id)
                document.setdefault("case_version_id", project.case_version_id)

        # Get ontology
        ontology = project.ontology
        if not ontology:
            return jsonify({
                "success": False,
                "error": "Ontology definition not found"
            }), 400

        # Get storage in request context (background thread cannot access current_app)
        storage = _get_storage()
        use_existing_graph = bool(incremental_mode and project.graph_id and not force)
        built_hashes = set(build_settings.get("built_document_hashes") or [])
        documents_for_build = document_texts
        if use_existing_graph and document_texts:
            documents_for_build = [
                document for document in document_texts
                if document.get("hash") not in built_hashes
            ]

        # Create async task
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Build graph: {graph_name}")
        logger.info(f"Graph build task created: task_id={task_id}, project_id={project_id}")
        
        # Update project status
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        ProjectManager.save_project(project)

        # Start background task
        def build_task():
            build_logger = get_logger('mirofish.build')
            build_started = time.perf_counter()
            timing = PhaseTimer("graph_build", metadata={
                "project_id": project_id,
                "task_id": task_id,
                "graph_name": graph_name,
                "build_preset": build_settings.get("build_preset"),
                "build_mode": build_mode,
            })
            build_profile = {
                "chunking_seconds": 0.0,
                "graph_create_seconds": 0.0,
                "ontology_seconds": 0.0,
                "llm_extraction_seconds": 0.0,
                "embedding_seconds": 0.0,
                "neo4j_write_seconds": 0.0,
                "graph_data_seconds": 0.0,
                "total_seconds": 0.0,
            }
            try:
                build_logger.info(f"[{task_id}] Starting graph build...")
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    message="Initializing graph build service..."
                )

                # Create graph builder service (storage passed from outer closure)
                with timing.time("initialize_service", label="Initialize graph builder"):
                    builder = GraphBuilderService(storage=storage)

                # Chunk text, preserving document boundaries when available.
                task_manager.update_task(
                    task_id,
                    message="Chunking text...",
                    progress=5
                )
                per_document_profile = []
                chunk_metadata = None
                with timing.time("chunking", label="Split documents into chunks", metadata={"document_count": len(documents_for_build)}):
                    if documents_for_build:
                        chunk_rows = GraphBuilderService.split_documents_with_provenance(
                            documents_for_build,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        chunks = [row["text"] for row in chunk_rows]
                        chunk_metadata = [row["metadata"] for row in chunk_rows]
                        for document in documents_for_build:
                            per_document_profile.append({
                                "document_id": document.get("document_id"),
                                "filename": document.get("filename"),
                                "document_type": document.get("document_type"),
                                "hash": document.get("hash"),
                                "text_length": len(document.get("text") or ""),
                                "chunk_count": sum(
                                    1 for row in chunk_rows
                                    if row["metadata"].get("document_id") == document.get("document_id")
                                ),
                                "cache_status": "new" if document.get("hash") not in built_hashes else "reused",
                            })
                    else:
                        chunks = TextProcessor.split_text(
                            text,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap
                        )
                build_profile["chunking_seconds"] = timing.snapshot()["phase_totals"].get("chunking", 0.0)
                build_profile["per_document"] = per_document_profile
                total_chunks = len(chunks)
                selected_text = "\n".join(chunks) if chunks else text
                file_hash = ExtractionCache.hash_text(selected_text)
                ontology_hash = ExtractionCache.hash_ontology(ontology)
                progress_detail = {
                    "current_chunk": 0,
                    "total_chunks": total_chunks,
                    "document_count": len(document_texts),
                    "documents_built": len(documents_for_build),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "batch_size": batch_size,
                    "build_preset": build_settings.get("build_preset"),
                    "build_mode": build_mode,
                    "incremental": incremental_mode,
                    "using_existing_graph": use_existing_graph,
                    "max_batch_chars": Config.GRAPH_BUILD_MAX_BATCH_CHARS,
                    "llm_provider": Config.LLM_DEFAULT_PROVIDER,
                    "llm_model": Config.LLM_MODEL_NAME,
                    "embedding_provider": Config.EMBEDDING_PROVIDER,
                    "embedding_model": Config.EMBEDDING_MODEL if Config.EMBEDDING_PROVIDER == "ollama" else Config.GEMINI_EMBEDDING_MODEL,
                    "file_hash": file_hash,
                    "eta_seconds": None,
                    "profile": build_profile,
                    "timing": timing.snapshot(),
                }

                if use_existing_graph and not chunks:
                    graph_id = project.graph_id
                    with timing.time("retrieve_graph_data", label="Retrieve existing graph data"):
                        graph_data = builder.get_graph_data(graph_id)
                    quality_summary = {}
                    with timing.time("graph_quality", label="Summarize graph quality"):
                        try:
                            quality_summary = storage.get_graph_quality(graph_id)
                        except Exception as quality_exc:
                            build_logger.warning(f"[{task_id}] Graph quality summary unavailable: {quality_exc}")
                    timing_summary = timing.complete(metadata={
                        "graph_id": graph_id,
                        "node_count": graph_data.get("node_count", 0),
                        "edge_count": graph_data.get("edge_count", 0),
                        "chunk_count": 0,
                    })
                    build_profile["total_seconds"] = timing_summary["total_seconds"]
                    build_profile["timing"] = timing_summary
                    project.status = ProjectStatus.GRAPH_COMPLETED
                    project.graph_build_settings = {
                        **build_settings,
                        "built_document_hashes": sorted(built_hashes),
                        "last_graph_quality": quality_summary,
                        "last_build_profile": build_profile,
                        "last_build_timing": timing_summary,
                    }
                    project.timings = {**(project.timings or {}), "graph_build": timing_summary}
                    ProjectManager.save_project(project)
                    if project.case_id and project.case_version_id:
                        CaseManager.update_version(
                            project.case_id,
                            project.case_version_id,
                            graph_id=graph_id,
                            build_profile=build_profile,
                            graph_quality=quality_summary,
                            status="graph_completed",
                        )
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.COMPLETED,
                        message="Graph already up to date",
                        progress=100,
                        progress_detail={**progress_detail, "profile": build_profile, "graph_quality": quality_summary, "timing": timing_summary},
                        result={
                            "project_id": project_id,
                            "graph_id": graph_id,
                            "node_count": graph_data.get("node_count", 0),
                            "edge_count": graph_data.get("edge_count", 0),
                            "chunk_count": 0,
                            "graph_quality": quality_summary,
                            "profile": build_profile,
                            "timing": timing_summary,
                            "settings": build_settings,
                        },
                    )
                    return

                # Create or reuse graph
                task_manager.update_task(
                    task_id,
                    message=(
                        f"Text split into {total_chunks} chunks. "
                        f"{'Using existing Neo4j graph' if use_existing_graph else 'Creating Neo4j graph'}..."
                    ),
                    progress=10,
                    progress_detail=progress_detail
                )
                with timing.time("create_graph", label="Create or reuse Neo4j graph"):
                    graph_id = project.graph_id if use_existing_graph else builder.create_graph(name=graph_name)
                build_profile["graph_create_seconds"] = timing.snapshot()["phase_totals"].get("create_graph", 0.0)

                # Update project graph_id
                project.graph_id = graph_id
                ProjectManager.save_project(project)

                # Set ontology
                task_manager.update_task(
                    task_id,
                    message="Setting ontology definition...",
                    progress=15
                )
                with timing.time("set_ontology", label="Store ontology on graph"):
                    builder.set_ontology(graph_id, ontology)
                build_profile["ontology_seconds"] = timing.snapshot()["phase_totals"].get("set_ontology", 0.0)
                
                # Add text (progress_callback signature is (msg, progress_ratio))
                def add_progress_callback(msg, progress_ratio, detail=None):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    latest_profile = {
                        **build_profile,
                        **((detail or {}).get("profile") or {}),
                    }
                    merged_detail = {
                        **progress_detail,
                        **(detail or {}),
                        "profile": latest_profile,
                        "timing": timing.snapshot(),
                    }
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress,
                        progress_detail=merged_detail,
                    )

                task_manager.update_task(
                    task_id,
                    message=f"Starting graph extraction for {total_chunks} chunks...",
                    progress=15,
                    progress_detail=progress_detail,
                )

                with timing.time("extract_and_write", label="Extract entities and write graph", metadata={"chunk_count": total_chunks}):
                    episode_uuids = builder.add_text_batches(
                        graph_id,
                        chunks,
                        batch_size=batch_size,
                        progress_callback=add_progress_callback,
                        cache_context={
                            "file_hash": file_hash,
                            "ontology_hash": ontology_hash,
                        },
                        llm_concurrency_override=llm_concurrency_override,
                        chunk_metadata=chunk_metadata,
                    )
                build_profile.update({
                    "llm_extraction_seconds": builder.last_build_profile.get("llm_extraction_seconds", 0.0),
                    "embedding_seconds": builder.last_build_profile.get("embedding_seconds", 0.0),
                    "neo4j_write_seconds": builder.last_build_profile.get("neo4j_write_seconds", 0.0),
                    "batch_wall_seconds": builder.last_build_profile.get("batch_wall_seconds", 0.0),
                    "cache_hits": builder.last_build_profile.get("cache_hits", 0),
                    "cache_misses": builder.last_build_profile.get("cache_misses", 0),
                    "llm_concurrency": builder.last_build_profile.get("llm_concurrency"),
                    "total_batches": builder.last_build_profile.get("total_batches"),
                })
                build_profile["timing"] = timing.snapshot()

                # Neo4j processing is synchronous, no need to wait
                task_manager.update_task(
                    task_id,
                    message="Text processing completed, generating graph data...",
                    progress=90,
                    progress_detail={
                        **progress_detail,
                        "current_chunk": total_chunks,
                        "total_chunks": total_chunks,
                        "profile": build_profile,
                        "timing": timing.snapshot(),
                    },
                )

                # Get graph data
                task_manager.update_task(
                    task_id,
                    message="Retrieving graph data...",
                    progress=95,
                    progress_detail={
                        **progress_detail,
                        "current_chunk": total_chunks,
                        "total_chunks": total_chunks,
                        "profile": build_profile,
                        "timing": timing.snapshot(),
                    },
                )
                with timing.time("retrieve_graph_data", label="Retrieve graph data"):
                    graph_data = builder.get_graph_data(graph_id)
                build_profile["graph_data_seconds"] = timing.snapshot()["phase_totals"].get("retrieve_graph_data", 0.0)
                quality_summary = {}
                with timing.time("graph_quality", label="Summarize graph quality"):
                    try:
                        quality_summary = storage.get_graph_quality(graph_id)
                    except Exception as quality_exc:
                        build_logger.warning(f"[{task_id}] Graph quality summary unavailable: {quality_exc}")
                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                timing_summary = timing.complete(metadata={
                    "graph_id": graph_id,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "chunk_count": total_chunks,
                    "episode_count": len(episode_uuids),
                })
                build_profile["total_seconds"] = timing_summary["total_seconds"]
                build_profile["timing"] = timing_summary

                # Update project status
                project.status = ProjectStatus.GRAPH_COMPLETED
                new_hashes = {document.get("hash") for document in documents_for_build if document.get("hash")}
                all_built_hashes = sorted((built_hashes | new_hashes) or {
                    document.get("hash") for document in document_texts if document.get("hash")
                })
                project.graph_build_settings = {
                    **build_settings,
                    "built_document_hashes": all_built_hashes,
                    "last_graph_quality": quality_summary,
                    "last_build_profile": build_profile,
                    "last_build_timing": timing_summary,
                }
                project.timings = {**(project.timings or {}), "graph_build": timing_summary}
                ProjectManager.save_project(project)
                if project.case_id and project.case_version_id:
                    CaseManager.update_version(
                        project.case_id,
                        project.case_version_id,
                        graph_id=graph_id,
                        build_profile=build_profile,
                        graph_quality=quality_summary,
                        build_settings=project.graph_build_settings,
                        status="graph_completed",
                    )

                build_logger.info(f"[{task_id}] Graph build completed: graph_id={graph_id}, nodes={node_count}, edges={edge_count}")

                # Complete
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="Graph build completed",
                    progress=100,
                    progress_detail={
                        **progress_detail,
                        "current_chunk": total_chunks,
                        "total_chunks": total_chunks,
                        "profile": build_profile,
                        "graph_quality": quality_summary,
                        "timing": timing_summary,
                    },
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "batch_size": batch_size,
                        "build_mode": build_mode,
                        "llm_provider": Config.LLM_DEFAULT_PROVIDER,
                        "llm_model": Config.LLM_MODEL_NAME,
                        "embedding_provider": Config.EMBEDDING_PROVIDER,
                        "profile": build_profile,
                        "graph_quality": quality_summary,
                        "timing": timing_summary,
                        "settings": project.graph_build_settings,
                    }
                )

            except Exception as e:
                # Update project status to failed
                build_logger.error(f"[{task_id}] Graph build failed: {str(e)}")
                build_logger.debug(traceback.format_exc())

                project.status = ProjectStatus.FAILED
                project.error = str(e)
                timing_summary = timing.fail(str(e))
                build_profile["timing"] = timing_summary
                build_profile["total_seconds"] = timing_summary["total_seconds"]
                project.timings = {**(project.timings or {}), "graph_build": timing_summary}
                ProjectManager.save_project(project)
                if project.case_id and project.case_version_id:
                    CaseManager.update_version(
                        project.case_id,
                        project.case_version_id,
                        status="failed",
                        build_profile=build_profile,
                    )

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Build failed: {str(e)}",
                    error=traceback.format_exc(),
                    progress_detail={"profile": build_profile, "timing": timing_summary},
                )

        # Start background thread
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "case_id": project.case_id,
                "case_version_id": project.case_version_id,
                "task_id": task_id,
                "settings": build_settings,
                "message": "Graph build task started. Query progress via /task/{task_id}"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Task Query Interface ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Query task status
    """
    task = TaskManager().get_task(task_id)

    if not task:
        return jsonify({
            "success": False,
            "error": f"Task does not exist: {task_id}"
        }), 404

    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all tasks
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== Graph Data Interface ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    Get graph data (nodes and edges)
    """
    try:
        storage = _get_storage()
        builder = GraphBuilderService(storage=storage)
        graph_data = builder.get_graph_data(graph_id)

        return jsonify({
            "success": True,
            "data": graph_data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/<graph_id>/embedding-status', methods=['GET'])
def get_graph_embedding_status(graph_id: str):
    """Return embedding coverage and report readiness for a graph."""
    try:
        storage = _get_storage()
        return jsonify({
            "success": True,
            "data": storage.get_embedding_status(graph_id)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/<graph_id>/quality', methods=['GET'])
def get_graph_quality(graph_id: str):
    """Return structural graph quality and synthesis readiness."""
    try:
        storage = _get_storage()
        return jsonify({
            "success": True,
            "data": storage.get_graph_quality(graph_id)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/<graph_id>/benchmark-embeddings', methods=['POST'])
def benchmark_graph_embeddings(graph_id: str):
    """Benchmark embedding providers against the same query set."""
    try:
        data = request.get_json(silent=True) or {}
        providers = data.get("providers") or ["ollama", "gemini"]
        providers = [str(provider).strip().lower() for provider in providers if str(provider).strip()]
        invalid = [provider for provider in providers if provider not in {"ollama", "gemini"}]
        if invalid:
            return jsonify({"success": False, "error": f"Unsupported providers: {', '.join(invalid)}"}), 400
        if not providers:
            return jsonify({"success": False, "error": "At least one provider is required"}), 400

        queries = data.get("queries") or DEFAULT_QUERIES
        queries = [str(query).strip() for query in queries if str(query).strip()]
        if not queries:
            return jsonify({"success": False, "error": "At least one benchmark query is required"}), 400
        queries = queries[:25]
        limit = max(1, min(int(data.get("limit", 10)), 25))

        return jsonify({
            "success": True,
            "data": run_embedding_benchmark(
                graph_id=graph_id,
                providers=providers,
                queries=queries,
                limit=limit,
            )
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/<graph_id>/reembed', methods=['POST'])
def reembed_graph(graph_id: str):
    """Backfill missing graph embeddings with the active embedding provider."""
    try:
        data = request.get_json(silent=True) or {}
        batch_size = int(data.get('batch_size', 32))
        storage = _get_storage()
        return jsonify({
            "success": True,
            "data": storage.reembed_graph(graph_id, batch_size=batch_size)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    Delete graph
    """
    try:
        storage = _get_storage()
        builder = GraphBuilderService(storage=storage)
        builder.delete_graph(graph_id)

        return jsonify({
            "success": True,
            "message": f"Graph deleted: {graph_id}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
