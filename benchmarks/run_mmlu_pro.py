from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import sample_window, save_results
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.stub_lm import StubLM

SUBJECTS = [
    "mmlu_pro_biology",
    "mmlu_pro_business",
    "mmlu_pro_chemistry",
    "mmlu_pro_computer_science",
    "mmlu_pro_economics",
    "mmlu_pro_engineering",
    "mmlu_pro_health",
    "mmlu_pro_history",
    "mmlu_pro_law",
    "mmlu_pro_math",
    "mmlu_pro_other",
    "mmlu_pro_philosophy",
    "mmlu_pro_physics",
    "mmlu_pro_psychology",
]


def run(limit: int | None = None, subjects: list[str] | None = None, model: str | None = None,
        stub: bool = False, start: int = 0) -> dict | None:
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

        eval_limit, samples = sample_window(task, start, limit)
        task_results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            batch_size=1,
            log_samples=True,
            limit=eval_limit,
            samples=samples,
        )

        json_path, summary_path = save_results(
            task_results, stem=f"results_{task}",
            output_dir=Path("logs/benchmarks") / "mmlu_pro",
            model_name=model, model=lm, start=start, limit=limit,
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
    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per subject")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first sample index; requires --limit")
    parser.add_argument(
        "--subjects", nargs="*", choices=SUBJECTS, default=None,
        help="One or more subjects (default: all 14)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run(limit=args.limit, subjects=args.subjects, model=args.model, stub=args.stub, start=args.start)
