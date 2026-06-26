from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import lm_eval

from benchmarks._utils import save_results
from benchmarks.lm_eval_adapter import BitNetLM

TASKS = [
    "meddialog_raw_dialogues",
    "meddialog_qsumm",
    "meddialog_qsumm_perplexity",
    "meddialog_raw_perplexity",
]


def run(limit: int | None = None, model: str | None = None) -> dict | None:
    lm = BitNetLM()
    results = None

    for task in TASKS:
        print(f"\n{'=' * 60}")
        print(f"Running: {task}")
        print(f"{'=' * 60}")

        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                batch_size=1,
                log_samples=True,
                limit=limit,
            )
        except NotImplementedError as e:
            print(f"  SKIP ({e})")
            continue
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}: {e})")
            traceback.print_exc()
            continue

        json_path, summary_path = save_results(
            results, stem=f"results_{task}",
            output_dir=Path("logs/benchmarks") / "meddialog",
            model_name=model,
        )
        print(summary_path.read_text(encoding="utf-8"))
        print(f"Results saved to {json_path.parent}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedDialog benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    args = parser.parse_args()
    run(limit=args.limit, model=args.model)
