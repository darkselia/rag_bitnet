from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import save_results
from benchmarks.lm_eval_adapter import BitNetLM


def run(limit: int | None = None, model: str | None = None) -> dict:
    lm = BitNetLM()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["ifeval"],
        batch_size=1,
        log_samples=True,
        limit=limit,
    )

    json_path, summary_path = save_results(results, stem="results", output_dir=Path("logs/benchmarks") / "ifeval",
                                           model_name=model)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Results saved to {json_path.parent}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IFEval benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    args = parser.parse_args()
    run(limit=args.limit, model=args.model)
