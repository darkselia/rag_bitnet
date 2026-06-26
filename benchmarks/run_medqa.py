from __future__ import annotations

import argparse
from pathlib import Path

import lm_eval

from benchmarks._utils import save_results
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.stub_lm import StubLM


def run(limit: int | None = None, model: str | None = None, stub: bool = False) -> dict | None:
    lm = StubLM() if stub else BitNetLM()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["medqa_4options"],
        batch_size=1,
        log_samples=True,
        limit=limit,
    )

    json_path, summary_path = save_results(results, stem="results", output_dir=Path("logs/benchmarks") / "medqa",
                                           model_name=model)
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Results saved to {json_path.parent}")

    samples = (results or {}).get("samples", {}).get("medqa_4options", [])
    if samples:
        print(f"\n{'─' * 60}")
        print("Per‑sample predictions (first 5):")
        for i, s in enumerate(samples[:5]):
            doc = s["doc"]
            choices = [doc.get(f"ending{j}", "") for j in range(4)]
            gold = doc.get("label", -1)
            gold_label = choices[gold] if 0 <= gold < len(choices) else "?"
            chosen_idx = -1
            for ci, resp in enumerate(s.get("filtered_resps", s.get("resps", []))):
                if isinstance(resp, (list, tuple)) and len(resp) == 2:
                    score, is_greedy = resp
                else:
                    score, is_greedy = (0.0, False) if not resp else resp
                if is_greedy:
                    chosen_idx = ci
                    break
            chosen_label = choices[chosen_idx] if 0 <= chosen_idx < len(choices) else "?"
            ok = "✓" if s.get("acc") else "✗"
            print(
                f"  #{i}: gold={gold_label}[{gold}]  "
                f"pred={chosen_label}[{chosen_idx}]  {ok}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedQA benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default=None, help="Model name for log filenames")
    parser.add_argument("--stub", action="store_true", help="Use StubLM instead of BitNetLM")
    args = parser.parse_args()
    run(limit=args.limit, model=args.model, stub=args.stub)
