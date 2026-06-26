from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import save_results
from benchmarks.stub_lm import StubLM


def run(limit: int | None = None) -> None:
    lm = StubLM()

    tasks_run = {
        "medqa": ["medqa_4options"],
        "mmlu_pro": ["mmlu_pro_math"],
        "mmlu_prox_ru": ["mmlu_prox_ru_health"],
        "ifeval": ["ifeval"],
    }

    for bench_name, tasks in tasks_run.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {bench_name} with StubLM")
        print(f"{'=' * 60}")

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size=1,
            log_samples=True,
            limit=limit,
        )

        json_path, summary_path = save_results(
            results, stem=f"stub_{bench_name}",
            output_dir=Path("logs/benchmarks") / "stub",
            model_name="stub",
        )
        print(summary_path.read_text(encoding="utf-8"))
        print(f"Saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test StubLM on benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit per task")
    args = parser.parse_args()
    run(limit=args.limit)
