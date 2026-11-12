from __future__ import annotations

import argparse
import time

from benchmarks.run_ifeval import run as run_ifeval
from benchmarks.run_medqa import run as run_medqa
from benchmarks.run_mmlu_pro import run as run_mmlu_pro
from benchmarks.run_mmlu_prox_ru import run as run_mmlu_prox_ru


def run_all(
        limit: int | None = None,
        model: str | None = None,
        stub: bool = False,
        start_index: int = 0,
        health_only: bool = False,
) -> None:
    if start_index < 0:
        raise ValueError("start must be >= 0")
    if start_index and limit is None:
        raise ValueError("--start requires --limit")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    pro_subjects = ["mmlu_pro_health"] if health_only else None
    prox_ru_subjects = ["mmlu_prox_ru_health"] if health_only else None
    runners = [
        ("MMLU-Pro", run_mmlu_pro, {"subjects": pro_subjects}),
        ("MMLU-Prox-ru", run_mmlu_prox_ru, {"subjects": prox_ru_subjects}),
        ("IFEval", run_ifeval, {}),
        ("MedQA", run_medqa, {}),
    ]

    failed = []
    for name, fn, kwargs in runners:
        print(f"\n{'#' * 60}")
        print(f"# Starting: {name}")
        print(f"{'#' * 60}")
        started_at = time.monotonic()
        try:
            fn(limit=limit, model=model, stub=stub, start=start_index, **kwargs)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        elapsed = time.monotonic() - started_at
        print(f"[DONE] {name} in {elapsed:.1f}s")

    print(f"\n{'#' * 60}")
    print("# All benchmarks finished")
    print(f"{'#' * 60}")
    if failed:
        raise RuntimeError("Failed benchmarks: " + ", ".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first sample index; requires --limit")
    parser.add_argument("--health-only", action="store_true", help="Run only health MMLU subjects")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run_all(limit=args.limit, model=args.model, stub=args.stub,
            start_index=args.start, health_only=args.health_only)
