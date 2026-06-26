from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import save_results
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.stub_lm import StubLM

SUBJECTS = [
    "mmlu_prox_ru_biology",
    "mmlu_prox_ru_business",
    "mmlu_prox_ru_chemistry",
    "mmlu_prox_ru_computer_science",
    "mmlu_prox_ru_economics",
    "mmlu_prox_ru_engineering",
    "mmlu_prox_ru_health",
    "mmlu_prox_ru_history",
    "mmlu_prox_ru_law",
    "mmlu_prox_ru_math",
    "mmlu_prox_ru_other",
    "mmlu_prox_ru_philosophy",
    "mmlu_prox_ru_physics",
    "mmlu_prox_ru_psychology",
]


def run(limit: int | None = None, subjects: list[str] | None = None,
        model: str | None = None, stub: bool = False) -> dict | None:
    lm = StubLM() if stub else BitNetLM()

    if subjects:
        tasks = subjects
    else:
        tasks = list(SUBJECTS)

    results = None

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"Running: {task}")
        print(f"{'=' * 60}")

        task_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=1,
            log_samples=True,
            limit=limit,
        )

        json_path, summary_path = save_results(
            task_results, stem=f"results_{task}",
            output_dir=Path("logs/benchmarks") / "mmlu_prox_ru",
            model_name=model,
        )
        print(summary_path.read_text(encoding="utf-8"))
        print(f"Results saved to {json_path.parent}")

        samples = (task_results or {}).get("samples", {}).get(task, [])
        if samples:
            print(f"Per‑sample predictions ({task}, first 5):")
            for i, s in enumerate(samples[:5]):
                gold = s["doc"].get("answer", "?")
                pred = s.get("filtered_resps", [None])[0] if s.get("filtered_resps") else "?"
                ok = "✓" if s.get("exact_match", 0) else "✗"
                print(f"  #{i}: gold={gold!r}  pred={pred!r}  {ok}")

        results = task_results

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU-ProX Russian benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per subject")
    parser.add_argument(
        "--subjects", nargs="*", choices=SUBJECTS, default=None,
        help="One or more subjects (default: all 14)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run(limit=args.limit, subjects=args.subjects, model=args.model, stub=args.stub)