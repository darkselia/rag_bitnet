from __future__ import annotations

import argparse
import time

from benchmarks import run_mmlu_prox_ru
from benchmarks.run_ifeval import run as run_ifeval
from benchmarks.run_meddialog import run as run_meddialog
from benchmarks.run_medqa import run as run_medqa
from benchmarks.run_mmlu_pro import run as run_mmlu_pro


def run_all(limit: int | None = None, model: str | None = None, stub: bool = False):
    runners = [
        ("MMLU-Pro", run_mmlu_pro),
        ("MMLU-Prox-ru", run_mmlu_prox_ru),
        ("IFEval", run_ifeval),
        # ("MedDialog", run_meddialog),
        ("MedQA", run_medqa),
    ]

    for name, fn in runners:
        print(f"\n{'#' * 60}")
        print(f"# Starting: {name}")
        print(f"{'#' * 60}")
        start = time.monotonic()
        try:
            fn(limit=limit, model=model, stub=stub)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        elapsed = time.monotonic() - start
        print(f"[DONE] {name} in {elapsed:.1f}s")

    print(f"\n{'#' * 60}")
    print("# All benchmarks finished")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run_all(limit=args.limit, model=args.model, stub=args.stub)
