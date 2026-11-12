from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import sample_window, save_results
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.stub_lm import StubLM


def run(limit: int | None = None, model: str | None = None, stub: bool = False,
        start: int = 0) -> dict:
    lm = StubLM() if stub else BitNetLM()
    eval_limit, samples = sample_window("ifeval", start, limit)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["ifeval"],
        batch_size=1,
        log_samples=True,
        limit=eval_limit,
        samples=samples,
    )

    json_path, summary_path = save_results(results, stem="results", output_dir=Path("logs/benchmarks") / "ifeval",
                                           model_name=model, model=lm, start=start, limit=limit)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Results saved to {json_path.parent}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IFEval benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first sample index; requires --limit")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run(limit=args.limit, model=args.model, stub=args.stub, start=args.start)
