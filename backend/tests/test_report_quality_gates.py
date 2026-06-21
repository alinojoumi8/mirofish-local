from datetime import datetime, timedelta

from app.services.report_agent import Report, ReportManager, ReportSection, ReportStatus, normalize_report_mode
from app.services.report_agent import ReportAgent, ReportOutline
from app.config import Config


def test_validate_report_output_blocks_raw_tool_call():
    report = Report(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        markdown_content="## Section\n\n<tool_call>{}</tool_call>",
    )

    issues = ReportManager.validate_report_output(report)

    assert any(issue["code"] == "raw_tool_call" and issue["blocking"] for issue in issues)


def test_report_mode_is_serialized_and_normalized():
    report = Report(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        report_mode="legal_case",
    )

    assert report.to_dict()["report_mode"] == "legal_case"
    assert normalize_report_mode("legal-case") == "legal_case"
    assert normalize_report_mode("unsupported") == "prediction"


def test_validate_report_output_warns_about_failed_interviews():
    report = Report(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        markdown_content="Interview API call failed: No successful interviews",
    )

    issues = ReportManager.validate_report_output(report)

    assert any(issue["code"] == "failed_interview" and not issue["blocking"] for issue in issues)


def test_validate_report_output_warns_about_repeated_section_facts_and_meta_commentary():
    repeated_fact = "TechCrunch published an internal memo showing Duolingo expected 88% of users to defect."
    report = Report(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        markdown_content=(
            "# Report\n\n"
            "## Section One\n\n"
            "The simulation captured early pressure from competitors. "
            f"{repeated_fact}\n\n"
            "## Section Two\n\n"
            "The simulation recorded analyst skepticism. "
            f"{repeated_fact}\n"
        ),
    )

    issues = ReportManager.validate_report_output(report)

    assert any(issue["code"] == "repeated_fact" for issue in issues)
    assert any(issue["code"] == "meta_commentary" for issue in issues)


def test_clean_section_content_removes_stray_bold_artifact():
    cleaned = ReportManager._clean_section_content(
        "**\n\nThe adoption story starts with pricing pressure.",
        "Market Adoption",
    )

    assert cleaned == "The adoption story starts with pricing pressure."


def test_quality_score_marks_warning_heavy_report_for_review():
    repeated_fact = "TechCrunch published an internal memo showing Duolingo expected 88% of users to defect."
    report = Report(
        report_id="report_1",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.COMPLETED,
        markdown_content=(
            "# Report\n\n"
            "## Section One\n\n"
            "The simulation captured this result. "
            f"{repeated_fact} Interview API call failed: No successful interviews.\n\n"
            "## Section Two\n\n"
            "The simulation recorded the same result. "
            f"{repeated_fact}\n"
        ),
    )

    report.validation_issues = ReportManager.validate_report_output(report)
    report.quality_score = ReportManager.evaluate_report_quality(report)
    ReportManager.apply_quality_status(report)

    assert report.quality_score["score"] < ReportManager.QUALITY_REVIEW_THRESHOLD
    assert report.status == ReportStatus.NEEDS_REVIEW


def test_enrich_progress_marks_old_generating_report_stale():
    old_timestamp = (datetime.now() - timedelta(minutes=20)).isoformat()
    progress = {
        "status": "generating",
        "progress": 37,
        "message": "Generating section",
        "updated_at": old_timestamp,
    }

    enriched = ReportManager.enrich_progress(progress, stale_after_seconds=60)

    assert enriched["is_stale"] is True
    assert enriched["effective_status"] == "stale"


def test_extract_evidence_cards_dedupes_numbered_facts():
    tool_result = """
### Key Facts
1. "TechCrunch published an internal memo showing Duolingo expected 88% of users to defect."
2. "TechCrunch published an internal memo showing Duolingo expected 88% of users to defect."
3. Lingua set its replacement plan at $14.99 per month after the shutdown.
"""

    cards = ReportManager.extract_evidence_cards(
        text=tool_result,
        tool_name="insight_forge",
        query="Duolingo migration risk",
        section_title="Competitive Response",
    )

    assert len(cards) == 2
    assert cards[0]["tool_name"] == "insight_forge"
    assert cards[0]["query"] == "Duolingo migration risk"
    assert cards[0]["section_title"] == "Competitive Response"
    assert cards[0]["usage"] == "retrieved"


def test_report_section_serializes_evidence_cards():
    section = ReportSection(
        title="Competitive Response",
        evidence_cards=[{"fact": "Duolingo expected 88% migration.", "tool_name": "quick_search"}],
    )

    assert section.to_dict()["evidence_cards"][0]["fact"] == "Duolingo expected 88% migration."


class ResumeAgent(ReportAgent):
    def __init__(self):
        self.graph_id = "graph_1"
        self.simulation_id = "sim_1"
        self.simulation_requirement = "test"
        self.report_mode = "prediction"
        self.report_logger = None
        self.console_logger = None
        self.disable_interviews = False
        self.strict_antirepetition = False
        self.tools = []
        self.generated = []

    def _generate_section_react(self, section, outline, previous_sections, progress_callback=None, section_index=1):
        self.generated.append(section_index)
        return f"generated section {section_index}"

    def _apply_forecast_synthesis(self, report, outline):
        report.forecast = {"confidence": "test"}


def test_resume_report_continues_from_first_missing_section(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))
    agent = ResumeAgent()
    report = Report(
        report_id="report_resume",
        simulation_id="sim_1",
        graph_id="graph_1",
        simulation_requirement="test",
        status=ReportStatus.FAILED,
        outline=ReportOutline(
            title="Report",
            summary="Summary",
            sections=[
                ReportSection(title="One"),
                ReportSection(title="Two"),
                ReportSection(title="Three"),
            ],
        ),
        error="temporary DNS failure",
    )
    ReportManager.save_report(report)
    report.outline.sections[0].content = "already generated"
    ReportManager.save_section(report.report_id, 1, report.outline.sections[0])
    ReportManager.update_progress(
        report.report_id,
        "failed",
        -1,
        "failed on section 2",
        completed_sections=["One"],
        current_section="Two",
        failed_section_index=2,
        failed_section_title="Two",
    )

    resumed = agent.resume_report(report)

    assert agent.generated == [2, 3]
    assert resumed.status in {ReportStatus.COMPLETED, ReportStatus.NEEDS_REVIEW}
    sections = ReportManager.get_generated_sections(report.report_id)
    assert [section["section_index"] for section in sections] == [1, 2, 3]


def test_report_progress_records_failed_section_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path / "reports"))

    ReportManager.update_progress(
        "report_failed",
        "failed",
        -1,
        "section failed",
        current_section="Risk",
        completed_sections=["Intro"],
        failed_section_index=2,
        failed_section_title="Risk",
    )

    progress = ReportManager.get_progress("report_failed")

    assert progress["failed_section_index"] == 2
    assert progress["failed_section_title"] == "Risk"
