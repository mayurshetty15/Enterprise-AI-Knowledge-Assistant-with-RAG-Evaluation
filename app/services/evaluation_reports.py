from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from evaluation.runner import DEFAULT_REPORTS_DIR

logger = logging.getLogger(__name__)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,120}$")

_CASE_FIELDS = (
    "case_id",
    "category",
    "retrieval_metrics",
    "answer_quality",
    "performance",
    "expected",
    "actual",
    "failure_kinds",
)


class EvaluationReportService:
    """Read only generated, content-redacted reports from a fixed local directory."""

    def __init__(self, reports_dir: Path | str = DEFAULT_REPORTS_DIR):
        self.reports_dir = Path(reports_dir).resolve()

    def _resolved_report_path(self, run_id: str) -> Path | None:
        try:
            run_directory = (self.reports_dir / run_id).resolve(strict=True)
            report_path = (run_directory / "report.json").resolve(strict=True)
        except FileNotFoundError:
            return None
        if (
            run_directory.parent != self.reports_dir
            or report_path.parent != run_directory
            or not report_path.is_file()
        ):
            return None
        return report_path

    @staticmethod
    def _safe_report(report: dict[str, Any]) -> dict[str, Any]:
        metadata = report.get("metadata", {})
        return {
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "run_id",
                    "dataset_version",
                    "dataset_sha256",
                    "configuration_sha256",
                    "generated_at",
                    "case_count",
                    "embedding_model",
                    "generation_model",
                    "configuration",
                    "local_judge_is_subjective",
                    "cost",
                )
            },
            "aggregates": report.get("aggregates", {}),
            "cases": [
                {key: case.get(key) for key in _CASE_FIELDS}
                for case in report.get("cases", [])
            ],
            "failures": [
                {
                    key: failure.get(key)
                    for key in (
                        "case_id",
                        "failure_kinds",
                        "expected",
                        "actual",
                        "answer_quality",
                    )
                }
                for failure in report.get("failures", [])
            ],
        }

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.reports_dir.exists():
            return []
        runs = []
        for run_directory in sorted(
            self.reports_dir.iterdir(), key=lambda item: item.name, reverse=True
        ):
            run_id = run_directory.name
            if not _RUN_ID_PATTERN.fullmatch(run_id):
                continue
            report_path = self._resolved_report_path(run_id)
            if report_path is None:
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable evaluation report %s", run_id)
                continue
            if report.get("metadata", {}).get("run_id") != run_id:
                continue
            safe = self._safe_report(report)
            runs.append(
                {
                    "metadata": safe["metadata"],
                    "aggregates": safe["aggregates"],
                    "failures": safe["failures"],
                }
            )
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return None
        report_path = self._resolved_report_path(run_id)
        if report_path is None:
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read evaluation report %s", run_id)
            raise
        if report.get("metadata", {}).get("run_id") != run_id:
            return None
        return self._safe_report(report)
