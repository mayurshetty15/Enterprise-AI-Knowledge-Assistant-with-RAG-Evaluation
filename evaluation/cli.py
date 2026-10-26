from __future__ import annotations

import argparse

from evaluation.runner import (
    DEFAULT_DATASET,
    DEFAULT_REPORTS_DIR,
    run_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation dataset against an isolated temporary index."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Evaluation dataset JSON path (default: bundled synthetic dataset).",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory where a unique run directory will be created.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Add optional subjective semantic scores using the configured local Ollama model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_evaluation(
        dataset_path=args.dataset,
        reports_dir=args.reports_dir,
        llm_judge=args.llm_judge,
    )
    print(f"Evaluation run: {report['metadata']['run_id']}")
    print(f"Reports: {args.reports_dir}\\{report['metadata']['report_directory']}")
    print(f"Cases: {report['metadata']['case_count']}")
    print(f"Failures: {report['aggregates']['failures']['cases_with_failure']}")


if __name__ == "__main__":
    main()
