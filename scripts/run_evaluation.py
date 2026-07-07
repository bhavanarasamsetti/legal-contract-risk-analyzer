"""CLI entry point for RAGAS evaluation.

Usage::

    python scripts/run_evaluation.py

Custom dataset::

    python scripts/run_evaluation.py --dataset evaluation/datasets/sample.json

Save results::

    python scripts/run_evaluation.py --output evaluation/results/latest.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import RagEvaluator

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "sample.json"
_SEP = "─" * 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the legal contract RAG pipeline with RAGAS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="INT",
        help="Number of chunks to retrieve per question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional path to save the full EvalResult as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"\n[1/2] Running evaluation  {_SEP}")
    print(f"  Dataset : {args.dataset}")
    print(f"  top_k   : {args.top_k}")

    evaluator = RagEvaluator(top_k=args.top_k)

    try:
        result = evaluator.evaluate(args.dataset)
    except Exception as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)

    print(f"\n[2/2] Results  {_SEP}")
    print()
    print(result.to_readme_markdown())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(result.model_dump(), fh, ensure_ascii=False, indent=2)
        print(f"\n  ✓  Saved full results to {args.output}")

    print(f"\n{'─' * 56}")
    print("  Evaluation complete.")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    main()
