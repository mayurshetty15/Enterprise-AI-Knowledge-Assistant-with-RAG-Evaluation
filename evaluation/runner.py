from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.config import Settings
from pydantic import ValidationError

from app.config import (
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    INSUFFICIENT_EVIDENCE_DISTANCE,
    MAX_CHUNK_CHARS,
    MAX_RETRIEVAL_RESULTS,
)
from app.generation import (
    GENERATION_NUM_PREDICT,
    GENERATION_TEMPERATURE,
    GENERATION_TIMEOUT_SECONDS,
    generate_answer,
)
from app.ingestion import chunk_text, stable_chunk_id
from app.retrieval import retrieve_relevant_chunks
from app.vector_store import generate_embedding
from evaluation.metrics import (
    citation_f1,
    lexical_answer_relevance,
    lexical_faithfulness,
    retrieval_metrics,
    token_f1,
)
from evaluation.schemas import EvaluationDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "synthetic-v1.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
K_VALUES = (1, 3, 5)


def load_dataset(path: Path | str) -> EvaluationDataset:
    """Load and validate a JSON evaluation dataset."""
    dataset_path = Path(path)
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read evaluation dataset: {dataset_path.name}") from error
    try:
        return EvaluationDataset.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"Invalid evaluation dataset: {error}") from error


def _index_corpus(dataset: EvaluationDataset) -> tuple[Any, set[str]]:
    client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.create_collection(
        name=f"rag_eval_{uuid.uuid4().hex[:16]}",
        metadata={"hnsw:space": "cosine"},
    )
    chunk_ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for corpus_document in dataset.corpus:
        for page in corpus_document.pages:
            for chunk_index, text in enumerate(
                chunk_text(page.text, MAX_CHUNK_CHARS, CHUNK_OVERLAP)
            ):
                chunk_id = stable_chunk_id(
                    corpus_document.filename, page.page, chunk_index, text
                )
                chunk_ids.append(chunk_id)
                embeddings.append(generate_embedding(text))
                documents.append(text)
                metadatas.append(
                    {
                        "document_id": corpus_document.document_id,
                        "source_name": corpus_document.filename,
                        "page_number": page.page,
                        "chunk_index": chunk_index,
                    }
                )
    if not chunk_ids:
        raise ValueError("Evaluation corpus produced no indexable chunks.")
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection, set(chunk_ids)


def _citation_sources(
    citations: list[dict[str, Any]], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    citation_keys = {
        (item.get("source_name"), int(item.get("page_number", 0)))
        for item in citations
    }
    sources: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in retrieved:
        key = (chunk.get("source_name"), int(chunk.get("page_number", 0)))
        if key in citation_keys:
            source_key = (str(chunk.get("document_id")), key[1])
            sources[source_key] = {"document_id": source_key[0], "page": key[1]}
    return list(sources.values())


def _failure_kinds(
    error_kind: str | None,
    answerable: bool,
    status: str | None,
    answer_quality: dict[str, Any],
) -> list[str]:
    failures = [error_kind] if error_kind else []
    if status and ((answerable and status != "answer_ready") or (not answerable and status != "insufficient_evidence")):
        failures.append("abstention_mismatch")
    if answer_quality.get("citation_f1") is not None and answer_quality["citation_f1"] < 1:
        failures.append("citation_mismatch")
    if answer_quality.get("answer_correctness") is not None and answer_quality["answer_correctness"] < 1:
        failures.append("reference_answer_mismatch")
    return failures


def _aggregate_mean(values: list[float | None]) -> dict[str, float | int | None]:
    available = [value for value in values if value is not None]
    return {
        "mean": statistics.fmean(available) if available else None,
        "n": len(available),
    }


def _latency_stats(values: list[float | None]) -> dict[str, float | int | None]:
    available = sorted(value for value in values if value is not None)
    return {
        "n": len(available),
        "median_ms": statistics.median(available) if available else None,
        "p95_ms": (
            available[math.ceil(0.95 * len(available)) - 1]
            if len(available) >= 20
            else None
        ),
        "p95_note": None if len(available) >= 20 else "Unavailable for fewer than 20 samples.",
    }


def _case_csv_row(case: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "status": case["actual"]["status"] or "",
        "failure_kinds": "|".join(case["failure_kinds"]),
        "retrieval_count": case["performance"]["retrieval_count"],
        "retrieval_latency_ms": case["performance"]["retrieval_latency_ms"],
        "generation_latency_ms": case["performance"]["generation_latency_ms"],
        "total_latency_ms": case["performance"]["total_latency_ms"],
    }
    for metric, value in case["answer_quality"].items():
        if metric == "local_judge" and isinstance(value, dict):
            for judge_metric, score in value.items():
                row[f"local_judge_{judge_metric}"] = score
        elif metric != "local_judge":
            row[metric] = value
    for metric, value in case["retrieval_metrics"].items():
        row[metric.replace("@", "_at_")] = value
    return row


def _markdown_report(report: dict[str, Any]) -> str:
    retrieval_lines = [
        f"| {name} | {value['mean'] if value['mean'] is not None else 'n/a'} | {value['n']} |"
        for name, value in report["aggregates"]["retrieval"].items()
    ]
    quality_lines = [
        f"| {name} | {value['mean'] if value['mean'] is not None else 'n/a'} | {value['n']} |"
        for name, value in report["aggregates"]["answer_quality"].items()
    ]
    latency_lines = []
    for name, value in report["aggregates"]["latency"].items():
        p95 = value["p95_ms"] if value["p95_ms"] is not None else "n/a"
        latency_lines.append(
            f"| {name} | {value['median_ms'] if value['median_ms'] is not None else 'n/a'} | {p95} | {value['n']} |"
        )
    failures = report["failures"]
    failure_lines = (
        [
            f"| {item['case_id']} | {', '.join(item['failure_kinds'])} | "
            f"{item['expected']['pages']} | {item['actual']['pages']} | "
            f"{item['actual']['status'] or 'error'} |"
            for item in failures
        ]
        or ["| None | No recorded failures | — | — | — |"]
    )
    metadata = report["metadata"]
    return "\n".join(
        [
            f"# RAG evaluation: {metadata['run_id']}",
            "",
            f"- Dataset: `{metadata['dataset_version']}` ({metadata['case_count']} cases)",
            f"- Timestamp: `{metadata['generated_at']}`",
            f"- Models: embedding `{metadata['embedding_model']}`, generation `{metadata['generation_model']}`",
            f"- Cost: not applicable (local Ollama inference).",
            "",
            "## Retrieval metrics",
            "",
            "| Metric | Mean | Cases scored |",
            "|---|---:|---:|",
            *retrieval_lines,
            "",
            "## Answer quality metrics",
            "",
            "| Metric | Mean | Cases scored |",
            "|---|---:|---:|",
            *quality_lines,
            "",
            "## Latency",
            "",
            "| Stage | Median ms | p95 ms | Samples |",
            "|---|---:|---:|---:|",
            *latency_lines,
            "",
            "p95 is omitted for fewer than 20 samples. Timings use a monotonic clock.",
            "",
            "## Failure examples",
            "",
            "| Case | Failure kinds | Expected pages | Retrieved/cited pages | Status |",
            "|---|---|---|---|---|",
            *failure_lines,
            "",
            "Reports intentionally exclude question text, document text, reference answers, and generated answer text.",
            "Lexical answer metrics are proxies, not semantic ground truth. Retrieval metrics use source pages unless exact chunk labels were supplied.",
            "",
        ]
    )


def _write_reports(
    report: dict[str, Any], reports_dir: Path | str
) -> Path:
    root = Path(reports_dir)
    run_directory = root / report["metadata"]["run_id"]
    run_directory.mkdir(parents=True, exist_ok=False)
    json_path = run_directory / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = run_directory / "cases.csv"
    rows = [_case_csv_row(case) for case in report["cases"]]
    fieldnames = list(rows[0]) if rows else ["case_id"]
    with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = run_directory / "report.md"
    with markdown_path.open("x", encoding="utf-8") as markdown_file:
        markdown_file.write(_markdown_report(report))
    return run_directory


def run_evaluation(
    dataset_path: Path | str = DEFAULT_DATASET,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    *,
    llm_judge: bool = False,
) -> dict[str, Any]:
    """Run the dataset against an isolated ephemeral Chroma index and save reports."""
    dataset = load_dataset(dataset_path)
    collection, indexed_chunk_ids = _index_corpus(dataset)
    for case in dataset.cases:
        unknown_ids = set(case.relevant_chunk_ids) - indexed_chunk_ids
        if unknown_ids:
            raise ValueError(
                f"Case {case.case_id!r} references unknown relevant chunk IDs."
            )

    from evaluation.judge import (
        JUDGE_NUM_PREDICT,
        JUDGE_SEED,
        JUDGE_TEMPERATURE,
        JUDGE_TIMEOUT_SECONDS,
        judge_answer,
    )

    case_results: list[dict[str, Any]] = []
    for case in dataset.cases:
        total_start = time.perf_counter()
        retrieval_ms: float | None = None
        generation_ms: float | None = None
        retrieved: list[dict[str, Any]] = []
        answer_result: dict[str, Any] = {}
        judge_scores: dict[str, float] | None = None
        judge_error_kind: str | None = None
        error_kind: str | None = None
        context = ""
        answer_text = ""

        retrieval_start = time.perf_counter()
        try:
            retrieved = retrieve_relevant_chunks(
                case.question,
                n_results=MAX_RETRIEVAL_RESULTS,
                collection=collection,
            )
        except requests.Timeout:
            error_kind = "retrieval_timeout"
        except Exception:
            error_kind = "retrieval_error"
        finally:
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if error_kind is None:
            context = "\n".join(item.get("text", "") for item in retrieved)
            generation_start = time.perf_counter()
            try:
                answer_result = generate_answer(case.question, retrieved)
                answer_text = answer_result.get("answer", "")
            except requests.Timeout:
                error_kind = "generation_timeout"
            except Exception:
                error_kind = "generation_error"
            finally:
                generation_ms = (time.perf_counter() - generation_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        status = answer_result.get("status") if error_kind is None else None
        actual_citations = _citation_sources(
            answer_result.get("citations", []), retrieved
        )
        expected_sources = [source.model_dump() for source in case.expected_sources]
        answer_quality: dict[str, Any] = {
            "answer_correctness": (
                token_f1(case.reference_answer, answer_text)
                if case.reference_answer is not None and error_kind is None
                else None
            ),
            "lexical_faithfulness": (
                lexical_faithfulness(answer_text, context)
                if answer_text and error_kind is None
                else None
            ),
            "lexical_answer_relevance": (
                lexical_answer_relevance(case.question, answer_text)
                if answer_text and error_kind is None
                else None
            ),
            "abstention_correct": (
                float(
                    status == ("answer_ready" if case.answerable else "insufficient_evidence")
                )
                if status is not None
                else None
            ),
            "citation_f1": (
                citation_f1(expected_sources, actual_citations)
                if error_kind is None
                else None
            ),
        }
        if llm_judge and error_kind is None and case.reference_answer is not None:
            try:
                judge_scores = judge_answer(
                    case.question, case.reference_answer, answer_text, context
                )
            except requests.Timeout:
                judge_error_kind = "local_judge_timeout"
            except Exception:
                judge_error_kind = "local_judge_error"
        answer_quality["local_judge"] = judge_scores

        retrieved_for_metrics = [
            {
                **item,
                "page": int(item["page_number"]),
            }
            for item in retrieved
        ]
        metrics = retrieval_metrics(
            retrieved_for_metrics,
            expected_sources,
            case.relevant_chunk_ids,
            K_VALUES,
        )
        actual_pages = sorted(
            {int(item["page"]) for item in actual_citations}
        )
        expected_pages = sorted({source["page"] for source in expected_sources})
        failures = _failure_kinds(
            error_kind, case.answerable, status, answer_quality
        )
        if judge_error_kind:
            failures.append(judge_error_kind)
        case_results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "retrieval_metrics": metrics,
                "answer_quality": answer_quality,
                "performance": {
                    "retrieval_latency_ms": retrieval_ms,
                    "generation_latency_ms": generation_ms,
                    "total_latency_ms": total_ms,
                    "retrieval_count": len(retrieved),
                    "error_kind": error_kind,
                    "judge_error_kind": judge_error_kind,
                },
                "expected": {
                    "answerable": case.answerable,
                    "source_count": len(expected_sources),
                    "pages": expected_pages,
                },
                "actual": {
                    "status": status,
                    "source_count": len(actual_citations),
                    "pages": actual_pages,
                },
                "failure_kinds": failures,
            }
        )

    retrieval_metric_names = sorted(
        {
            metric
            for case in case_results
            for metric in case["retrieval_metrics"]
        }
    )
    answer_metric_names = sorted(
        {
            aggregate_metric
            for case in case_results
            for metric, value in case["answer_quality"].items()
            for aggregate_metric in (
                [
                    f"local_judge_{judge_metric}"
                    for judge_metric in value
                ]
                if metric == "local_judge" and isinstance(value, dict)
                else [metric]
            )
            if (
                metric == "local_judge" and isinstance(value, dict)
                or value is not None and isinstance(value, (int, float))
            )
        }
    )
    retrieval_aggregates = {
        name: _aggregate_mean(
            [case["retrieval_metrics"][name] for case in case_results]
        )
        for name in retrieval_metric_names
    }
    answer_aggregates = {
        name: _aggregate_mean(
            [
                (
                    case["answer_quality"].get("local_judge", {}).get(
                        name.removeprefix("local_judge_")
                    )
                    if name.startswith("local_judge_")
                    else case["answer_quality"].get(name)
                )
                for case in case_results
            ]
        )
        for name in answer_metric_names
    }
    retrieval_latencies = [
        case["performance"]["retrieval_latency_ms"] for case in case_results
    ]
    generation_latencies = [
        case["performance"]["generation_latency_ms"] for case in case_results
    ]
    total_latencies = [
        case["performance"]["total_latency_ms"] for case in case_results
    ]

    dataset_bytes = Path(dataset_path).read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()[:10]
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_retrieval_results": MAX_RETRIEVAL_RESULTS,
        "insufficient_evidence_distance": INSUFFICIENT_EVIDENCE_DISTANCE,
        "k_values": list(K_VALUES),
        "llm_judge": llm_judge,
        "generation_temperature": GENERATION_TEMPERATURE,
        "generation_num_predict": GENERATION_NUM_PREDICT,
        "generation_timeout_seconds": GENERATION_TIMEOUT_SECONDS,
        "judge_temperature": JUDGE_TEMPERATURE if llm_judge else None,
        "judge_seed": JUDGE_SEED if llm_judge else None,
        "judge_num_predict": JUDGE_NUM_PREDICT if llm_judge else None,
        "judge_timeout_seconds": JUDGE_TIMEOUT_SECONDS if llm_judge else None,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    now = datetime.now(timezone.utc)
    run_id = (
        f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{dataset_hash}_{config_hash[:8]}"
    )
    report = {
        "metadata": {
            "run_id": run_id,
            "dataset_version": dataset.dataset_version,
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "configuration_sha256": config_hash,
            "generated_at": now.isoformat(),
            "case_count": len(case_results),
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "configuration": config,
            "local_judge_is_subjective": llm_judge,
            "cost": "not_applicable_local_ollama",
        },
        "aggregates": {
            "retrieval": retrieval_aggregates,
            "answer_quality": answer_aggregates,
            "latency": {
                "retrieval": _latency_stats(retrieval_latencies),
                "generation": _latency_stats(generation_latencies),
                "total": _latency_stats(total_latencies),
            },
            "failures": {
                "case_errors": sum(
                    case["performance"]["error_kind"] is not None
                    for case in case_results
                ),
                "timeouts": sum(
                    case["performance"]["error_kind"] in (
                        "retrieval_timeout",
                        "generation_timeout",
                    )
                    or case["performance"]["judge_error_kind"] == "local_judge_timeout"
                    for case in case_results
                ),
                "judge_errors": sum(
                    case["performance"]["judge_error_kind"] is not None
                    for case in case_results
                ),
                "cases_with_failure": sum(
                    bool(case["failure_kinds"]) for case in case_results
                ),
            },
        },
        "cases": case_results,
        "failures": [
            {
                "case_id": case["case_id"],
                "failure_kinds": case["failure_kinds"],
                "expected": case["expected"],
                "actual": case["actual"],
                "answer_quality": {
                    key: case["answer_quality"][key]
                    for key in (
                        "answer_correctness",
                        "abstention_correct",
                        "citation_f1",
                    )
                },
            }
            for case in case_results
            if case["failure_kinds"]
        ],
    }
    report_directory = _write_reports(report, reports_dir)
    report["metadata"]["report_directory"] = report_directory.name
    return report
