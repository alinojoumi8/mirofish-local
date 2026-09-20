"""
Case-level litigation workflow models.

Cases group one or more upload batches. Each upload batch becomes a case
version with its own project, documents, graph, simulations, reports, and
prediction snapshots.
"""

import hashlib
import json
import os
import re
import threading
import uuid
from functools import wraps
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.case")
_CASE_MUTATION_LOCK = threading.RLock()


def _serialized_case_mutation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _CASE_MUTATION_LOCK:
            return function(*args, **kwargs)
    return wrapped


class DocumentType(str, Enum):
    PLEADING = "pleading"
    MOTION_RECORD = "motion_record"
    FACTUM = "factum"
    AFFIDAVIT = "affidavit"
    EXHIBIT = "exhibit"
    ORDER = "order"
    TRANSCRIPT = "transcript"
    EXPERT_REPORT = "expert_report"
    CORRESPONDENCE = "correspondence"
    OTHER = "other"


GRAPH_BUILD_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast_scan": {
        "build_preset": "fast_scan",
        "build_mode": "fast_scan",
        "chunk_size": 4500,
        "chunk_overlap": 100,
        "batch_size": max(8, Config.GRAPH_BUILD_BATCH_SIZE),
        "llm_concurrency": None,
        "embedding_provider": Config.EMBEDDING_PROVIDER,
        "cache_reuse": True,
        "incremental": True,
        "extraction_depth": "low",
        "relation_density": "low",
    },
    "balanced": {
        "build_preset": "balanced",
        "build_mode": "accurate_fast",
        "chunk_size": Config.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
        "batch_size": Config.GRAPH_BUILD_BATCH_SIZE,
        "llm_concurrency": None,
        "embedding_provider": Config.EMBEDDING_PROVIDER,
        "cache_reuse": True,
        "incremental": True,
        "extraction_depth": "medium",
        "relation_density": "medium",
    },
    "deep_evidence": {
        "build_preset": "deep_evidence",
        "build_mode": "deep_evidence",
        "chunk_size": 1600,
        "chunk_overlap": 250,
        "batch_size": max(1, min(Config.GRAPH_BUILD_BATCH_SIZE, 4)),
        "llm_concurrency": 1,
        "embedding_provider": Config.EMBEDDING_PROVIDER,
        "cache_reuse": True,
        "incremental": True,
        "extraction_depth": "high",
        "relation_density": "high",
    },
}


PREDICTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick_legal_read": {
        "preset": "quick_legal_read",
        "agent_count": 24,
        "entity_types_include": [],
        "entity_types_exclude": [],
        "profile_parallelism": 4,
        "simulation_duration": 24,
        "minutes_per_round": 60,
        "active_agents_per_round": 8,
        "active_agents_per_hour": 8,
        "ensemble_runs": 1,
        "scenario_pack": "baseline",
        "memory_mode": "off",
        "report_mode": "litigation_case",
        "report_depth": "fast",
        "max_rounds": 24,
        "seed": None,
    },
    "standard_litigation_prediction": {
        "preset": "standard_litigation_prediction",
        "agent_count": 48,
        "entity_types_include": [],
        "entity_types_exclude": [],
        "profile_parallelism": 5,
        "simulation_duration": 72,
        "minutes_per_round": 60,
        "active_agents_per_round": 16,
        "active_agents_per_hour": 16,
        "ensemble_runs": 5,
        "scenario_pack": "baseline_adverse_favorable",
        "memory_mode": "practical",
        "report_mode": "litigation_case",
        "report_depth": "standard",
        "max_rounds": 72,
        "seed": None,
    },
    "deep_adversarial_run": {
        "preset": "deep_adversarial_run",
        "agent_count": 96,
        "entity_types_include": [],
        "entity_types_exclude": [],
        "profile_parallelism": 8,
        "simulation_duration": 120,
        "minutes_per_round": 30,
        "active_agents_per_round": 24,
        "active_agents_per_hour": 24,
        "ensemble_runs": 10,
        "scenario_pack": "baseline,plaintiff-favorable,defendant-favorable,procedural-delay,settlement-pressure",
        "memory_mode": "full",
        "report_mode": "litigation_case",
        "report_depth": "deep",
        "max_rounds": 240,
        "seed": None,
    },
}


def _now() -> str:
    return datetime.now().isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def classify_document_type(filename: str = "", text: str = "") -> DocumentType:
    haystack = f"{filename or ''}\n{text[:2000] if text else ''}".lower()
    rules = [
        (DocumentType.MOTION_RECORD, r"motion\s+record|notice\s+of\s+motion|memorandum\s+of\s+motion"),
        (DocumentType.FACTUM, r"factum|memorandum\s+of\s+argument|written\s+submissions"),
        (DocumentType.AFFIDAVIT, r"affidavit|sworn|deponent"),
        (DocumentType.EXPERT_REPORT, r"expert\s+report|expert\s+opinion|curriculum\s+vitae"),
        (DocumentType.TRANSCRIPT, r"transcript|cross[-\s]?examination|examination\s+for\s+discovery"),
        (DocumentType.ORDER, r"\border\b|judgment|reasons\s+for\s+decision|endorsement"),
        (DocumentType.EXHIBIT, r"exhibit|schedule|appendix"),
        (DocumentType.CORRESPONDENCE, r"correspondence|letter|email|e-mail"),
        (DocumentType.PLEADING, r"statement[-\s]+of[-\s]+claim|statement[-\s]+of[-\s]+defen[cs]e|complaint|answer|reply|pleading"),
    ]
    for doc_type, pattern in rules:
        if re.search(pattern, haystack):
            return doc_type
    return DocumentType.OTHER


def normalize_graph_build_settings(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = raw or {}
    preset_name = str(raw.get("build_preset") or raw.get("preset") or raw.get("build_mode") or "fast_scan")
    preset_name = preset_name.strip().lower().replace("-", "_")
    if preset_name in {"accurate_fast", "balanced"}:
        preset_name = "balanced"
    if preset_name not in GRAPH_BUILD_PRESETS:
        preset_name = "fast_scan"

    settings = dict(GRAPH_BUILD_PRESETS[preset_name])
    aliases = {
        "incremental_mode": "incremental",
        "cache": "cache_reuse",
        "reuse_cache": "cache_reuse",
    }
    for key, value in raw.items():
        normalized_key = aliases.get(key, key)
        if normalized_key in settings or normalized_key in {
            "chunk_size",
            "chunk_overlap",
            "batch_size",
            "llm_concurrency",
            "build_mode",
            "embedding_provider",
            "incremental",
            "cache_reuse",
        }:
            settings[normalized_key] = value

    settings["build_preset"] = preset_name
    settings["chunk_size"] = max(1, int(settings.get("chunk_size") or GRAPH_BUILD_PRESETS[preset_name]["chunk_size"]))
    settings["chunk_overlap"] = max(0, min(int(settings.get("chunk_overlap") or 0), settings["chunk_size"] - 1))
    settings["batch_size"] = max(1, int(settings.get("batch_size") or 1))
    if settings.get("llm_concurrency") in ("", None):
        settings["llm_concurrency"] = None
    else:
        settings["llm_concurrency"] = max(1, int(settings["llm_concurrency"]))
    settings["incremental"] = bool(settings.get("incremental", True))
    settings["cache_reuse"] = bool(settings.get("cache_reuse", True))
    return settings


def normalize_prediction_settings(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = raw or {}
    preset_name = str(raw.get("preset") or raw.get("prediction_preset") or "standard_litigation_prediction")
    preset_name = preset_name.strip().lower().replace("-", "_").replace(" ", "_")
    preset_aliases = {
        "quick": "quick_legal_read",
        "standard": "standard_litigation_prediction",
        "deep": "deep_adversarial_run",
    }
    preset_name = preset_aliases.get(preset_name, preset_name)
    if preset_name not in PREDICTION_PRESETS:
        preset_name = "standard_litigation_prediction"

    settings = dict(PREDICTION_PRESETS[preset_name])
    for key, value in raw.items():
        if key in settings or key in {
            "agent_count",
            "entity_types_include",
            "entity_types_exclude",
            "profile_parallelism",
            "simulation_duration",
            "minutes_per_round",
            "active_agents_per_round",
            "active_agents_per_hour",
            "ensemble_runs",
            "scenario_pack",
            "memory_mode",
            "report_mode",
            "report_depth",
            "max_rounds",
            "seed",
            "prediction_target",
        }:
            settings[key] = value

    settings["preset"] = preset_name
    for key in (
        "agent_count",
        "profile_parallelism",
        "simulation_duration",
        "minutes_per_round",
        "active_agents_per_round",
        "active_agents_per_hour",
        "ensemble_runs",
        "max_rounds",
    ):
        value = settings.get(key)
        if value in ("", None):
            continue
        settings[key] = max(1, int(value))
    for key in ("entity_types_include", "entity_types_exclude"):
        value = settings.get(key)
        if isinstance(value, str):
            settings[key] = [part.strip() for part in value.split(",") if part.strip()]
        elif not isinstance(value, list):
            settings[key] = []
    settings["memory_mode"] = str(settings.get("memory_mode") or "practical").strip().lower()
    if settings["memory_mode"] not in {"off", "practical", "full"}:
        settings["memory_mode"] = "practical"
    settings["report_mode"] = str(settings.get("report_mode") or "litigation_case").strip().lower().replace("-", "_")
    settings["report_depth"] = str(settings.get("report_depth") or "standard").strip().lower()
    return settings


@dataclass
class CaseDocument:
    document_id: str
    filename: str
    document_type: DocumentType
    hash: str
    size: int = 0
    path: Optional[str] = None
    text_length: int = 0
    uploaded_at: str = field(default_factory=_now)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["document_type"] = self.document_type.value if isinstance(self.document_type, DocumentType) else self.document_type
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseDocument":
        doc_type = data.get("document_type", DocumentType.OTHER)
        if isinstance(doc_type, str):
            doc_type = DocumentType(doc_type) if doc_type in {item.value for item in DocumentType} else DocumentType.OTHER
        return cls(
            document_id=data["document_id"],
            filename=data.get("filename", ""),
            document_type=doc_type,
            hash=data.get("hash", ""),
            size=int(data.get("size", 0) or 0),
            path=data.get("path"),
            text_length=int(data.get("text_length", 0) or 0),
            uploaded_at=data.get("uploaded_at", _now()),
            provenance=data.get("provenance", {}),
        )


@dataclass
class CaseVersion:
    version_id: str
    version_number: int
    project_id: str
    created_at: str
    documents: List[CaseDocument] = field(default_factory=list)
    graph_id: Optional[str] = None
    simulation_ids: List[str] = field(default_factory=list)
    report_ids: List[str] = field(default_factory=list)
    prediction_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    build_settings: Dict[str, Any] = field(default_factory=dict)
    prediction_settings: Dict[str, Any] = field(default_factory=dict)
    graph_quality: Dict[str, Any] = field(default_factory=dict)
    build_profile: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["documents"] = [doc.to_dict() for doc in self.documents]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseVersion":
        return cls(
            version_id=data["version_id"],
            version_number=int(data.get("version_number", 1) or 1),
            project_id=data.get("project_id", ""),
            created_at=data.get("created_at", _now()),
            documents=[CaseDocument.from_dict(item) for item in data.get("documents", [])],
            graph_id=data.get("graph_id"),
            simulation_ids=data.get("simulation_ids", []),
            report_ids=data.get("report_ids", []),
            prediction_snapshots=data.get("prediction_snapshots", []),
            build_settings=data.get("build_settings", {}),
            prediction_settings=data.get("prediction_settings", {}),
            graph_quality=data.get("graph_quality", {}),
            build_profile=data.get("build_profile", {}),
            status=data.get("status", "created"),
        )


@dataclass
class Case:
    case_id: str
    name: str
    simulation_requirement: str
    created_at: str
    updated_at: str
    versions: List[CaseVersion] = field(default_factory=list)
    graph_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "simulation_requirement": self.simulation_requirement,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": [version.to_dict() for version in self.versions],
            "graph_ids": self.graph_ids,
            "tags": self.tags,
            "latest_version": self.versions[-1].to_dict() if self.versions else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        return cls(
            case_id=data["case_id"],
            name=data.get("name", "Untitled Case"),
            simulation_requirement=data.get("simulation_requirement", ""),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            versions=[CaseVersion.from_dict(item) for item in data.get("versions", [])],
            graph_ids=data.get("graph_ids", []),
            tags=data.get("tags", []),
        )


def build_prediction_snapshot(
    forecast: Optional[Dict[str, Any]],
    report_id: Optional[str] = None,
    simulation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten a report's `forecast` dict into a diff-friendly snapshot.

    Top-level keys are per-outcome probabilities plus `point_estimate`/`confidence`,
    so `compare_prediction_versions` produces clean numeric deltas. Non-comparable
    context lives under the reserved `_meta` key (skipped by the comparison).
    """
    forecast = forecast or {}
    snapshot: Dict[str, Any] = {}
    for row in forecast.get("probabilities") or []:
        label = str(row.get("outcome") or "").strip()
        if not label:
            continue
        try:
            snapshot[label] = round(float(row.get("probability") or 0.0), 4)
        except (TypeError, ValueError):
            continue
    numeric = forecast.get("numeric") or {}
    if numeric.get("point_estimate") is not None:
        snapshot["point_estimate"] = numeric.get("point_estimate")
    if forecast.get("confidence"):
        snapshot["confidence"] = forecast.get("confidence")
    snapshot["_meta"] = {
        "report_id": report_id,
        "simulation_id": simulation_id,
        "forecast_mode": forecast.get("forecast_mode"),
        "forecast_horizon": forecast.get("forecast_horizon"),
        "created_at": _now(),
        "probabilities": forecast.get("probabilities") or [],
    }
    return snapshot


class CaseManager:
    CASES_DIR = os.path.join(Config.UPLOAD_FOLDER, "cases")
    _repository = None

    @classmethod
    def configure_repository(cls, repository) -> None:
        cls._repository = repository

    @classmethod
    def _ensure_cases_dir(cls) -> None:
        os.makedirs(cls.CASES_DIR, exist_ok=True)

    @classmethod
    def _get_case_dir(cls, case_id: str) -> str:
        return os.path.join(cls.CASES_DIR, case_id)

    @classmethod
    def _get_case_path(cls, case_id: str) -> str:
        return os.path.join(cls._get_case_dir(case_id), "case.json")

    @classmethod
    @_serialized_case_mutation
    def create_case(cls, name: str, simulation_requirement: str = "", tags: Optional[List[str]] = None) -> Case:
        cls._ensure_cases_dir()
        case_id = f"case_{uuid.uuid4().hex[:12]}"
        now = _now()
        case = Case(
            case_id=case_id,
            name=name or "Untitled Case",
            simulation_requirement=simulation_requirement or "",
            created_at=now,
            updated_at=now,
            tags=tags or [],
        )
        os.makedirs(cls._get_case_dir(case_id), exist_ok=True)
        cls.save_case(case)
        return case

    @classmethod
    def save_case(cls, case: Case) -> None:
        cls._ensure_cases_dir()
        case.updated_at = _now()
        if cls._repository is not None:
            cls._repository.put_case(case.to_dict())
            return

        os.makedirs(cls._get_case_dir(case.case_id), exist_ok=True)
        with open(cls._get_case_path(case.case_id), "w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def get_case(cls, case_id: str) -> Optional[Case]:
        if cls._repository is not None:
            payload = cls._repository.get_case(case_id)
            return Case.from_dict(payload) if payload else None

        path = cls._get_case_path(case_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return Case.from_dict(json.load(f))

    @classmethod
    def list_cases(cls, limit: int = 50) -> List[Case]:
        if cls._repository is not None:
            return [Case.from_dict(item) for item in cls._repository.list_cases(limit)]

        cls._ensure_cases_dir()
        cases: List[Case] = []
        for case_id in os.listdir(cls.CASES_DIR):
            case = cls.get_case(case_id)
            if case:
                cases.append(case)
        cases.sort(key=lambda item: item.updated_at, reverse=True)
        return cases[:limit]

    @classmethod
    @_serialized_case_mutation
    def create_version(
        cls,
        case_id: str,
        project_id: str,
        graph_id: Optional[str] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        build_settings: Optional[Dict[str, Any]] = None,
        prediction_settings: Optional[Dict[str, Any]] = None,
        prediction_snapshot: Optional[Dict[str, Any]] = None,
    ) -> CaseVersion:
        case = cls.get_case(case_id)
        if not case:
            raise ValueError(f"Case does not exist: {case_id}")

        version_id = f"casever_{uuid.uuid4().hex[:12]}"
        version_number = len(case.versions) + 1
        case_documents: List[CaseDocument] = []
        for raw in documents or []:
            text = raw.get("text") or ""
            document_id = raw.get("document_id") or f"doc_{uuid.uuid4().hex[:12]}"
            doc_type = raw.get("document_type") or classify_document_type(raw.get("filename", ""), text).value
            if isinstance(doc_type, DocumentType):
                doc_type_enum = doc_type
            else:
                doc_type_enum = DocumentType(doc_type) if doc_type in {item.value for item in DocumentType} else DocumentType.OTHER
            provenance = {
                "case_id": case_id,
                "version_id": version_id,
                "document_id": document_id,
                "filename": raw.get("filename", ""),
                "document_type": doc_type_enum.value,
            }
            case_documents.append(
                CaseDocument(
                    document_id=document_id,
                    filename=raw.get("filename", ""),
                    document_type=doc_type_enum,
                    hash=raw.get("hash") or _hash_text(text or raw.get("filename", "")),
                    size=int(raw.get("size", 0) or 0),
                    path=raw.get("path"),
                    text_length=len(text),
                    provenance={**provenance, **(raw.get("provenance") or {})},
                )
            )

        version = CaseVersion(
            version_id=version_id,
            version_number=version_number,
            project_id=project_id,
            graph_id=graph_id,
            created_at=_now(),
            documents=case_documents,
            build_settings=normalize_graph_build_settings(build_settings),
            prediction_settings=normalize_prediction_settings(prediction_settings),
            prediction_snapshots=[prediction_snapshot] if prediction_snapshot else [],
        )
        case.versions.append(version)
        if graph_id and graph_id not in case.graph_ids:
            case.graph_ids.append(graph_id)
        cls.save_case(case)
        return version

    @classmethod
    @_serialized_case_mutation
    def update_version(cls, case_id: str, version_id: str, **updates: Any) -> Optional[CaseVersion]:
        case = cls.get_case(case_id)
        if not case:
            logger.warning(
                "update_version skipped: case not found case_id=%s version_id=%s fields=%s",
                case_id, version_id, ",".join(sorted(updates)) or "-",
            )
            return None
        target = None
        for version in case.versions:
            if version.version_id == version_id:
                target = version
                break
        if not target:
            logger.warning(
                "update_version skipped: version not found case_id=%s version_id=%s fields=%s",
                case_id, version_id, ",".join(sorted(updates)) or "-",
            )
            return None

        applied = []
        for key, value in updates.items():
            if not hasattr(target, key):
                logger.warning(
                    "update_version ignoring unknown field: case_id=%s version_id=%s field=%s",
                    case_id, version_id, key,
                )
                continue
            setattr(target, key, value)
            applied.append(key)
        if target.graph_id and target.graph_id not in case.graph_ids:
            case.graph_ids.append(target.graph_id)
        cls.save_case(case)
        logger.debug(
            "Updated case version: case_id=%s version_id=%s fields=%s",
            case_id, version_id, ",".join(applied) or "-",
        )
        return target

    @classmethod
    def get_version(cls, case_id: str, version_id: str) -> Optional[CaseVersion]:
        case = cls.get_case(case_id)
        if not case:
            return None
        for version in case.versions:
            if version.version_id == version_id:
                return version
        return None

    @classmethod
    @_serialized_case_mutation
    def record_prediction_snapshot(
        cls,
        case_id: str,
        version_id: str,
        snapshot: Dict[str, Any],
        report_id: Optional[str] = None,
        simulation_id: Optional[str] = None,
    ) -> Optional[CaseVersion]:
        """Append a prediction snapshot to a case version (unlike update_version,
        which replaces attributes) and link the report/simulation that produced it."""
        case = cls.get_case(case_id)
        if not case:
            logger.warning(
                "record_prediction_snapshot skipped: case not found case_id=%s version_id=%s report_id=%s",
                case_id, version_id, report_id,
            )
            return None
        target = None
        for version in case.versions:
            if version.version_id == version_id:
                target = version
                break
        if not target:
            logger.warning(
                "record_prediction_snapshot skipped: version not found case_id=%s version_id=%s report_id=%s",
                case_id, version_id, report_id,
            )
            return None

        target.prediction_snapshots.append(snapshot)
        if report_id and report_id not in target.report_ids:
            target.report_ids.append(report_id)
        if simulation_id and simulation_id not in target.simulation_ids:
            target.simulation_ids.append(simulation_id)
        target.status = "reported"
        cls.save_case(case)
        logger.info(
            "Recorded prediction snapshot: case_id=%s version_id=%s report_id=%s outcomes=%d",
            case_id, version_id, report_id,
            sum(1 for key in snapshot if not key.startswith("_")),
        )
        return target

    @classmethod
    def compare_prediction_versions(cls, case_id: str, from_version_id: str, to_version_id: str) -> Dict[str, Any]:
        from_version = cls.get_version(case_id, from_version_id)
        to_version = cls.get_version(case_id, to_version_id)
        if not from_version or not to_version:
            raise ValueError("Both case versions must exist")

        from_snapshot = from_version.prediction_snapshots[-1] if from_version.prediction_snapshots else {}
        to_snapshot = to_version.prediction_snapshots[-1] if to_version.prediction_snapshots else {}
        changed: Dict[str, Dict[str, Any]] = {}
        for key in sorted(set(from_snapshot) | set(to_snapshot)):
            if key.startswith("_"):
                continue
            old = from_snapshot.get(key)
            new = to_snapshot.get(key)
            if old == new:
                continue
            row = {"from": old, "to": new}
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                row["delta"] = round(new - old, 4)
            changed[key] = row

        return {
            "case_id": case_id,
            "from_version_id": from_version_id,
            "to_version_id": to_version_id,
            "from_snapshot": from_snapshot,
            "to_snapshot": to_snapshot,
            "changed_predictions": changed,
        }
