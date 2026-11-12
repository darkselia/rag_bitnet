from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset

from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.stub_lm import StubLM
from common import config


def extract_answer(text: str) -> str | None:
    match = re.fullmatch(r"\s*\(?([A-D])\)?[.]?\s*", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def run(limit: int | None = None, model: str | None = None, stub: bool = False,
        start: int = 0) -> dict:
    if start < 0 or (limit is not None and limit < 1) or (start and limit is None):
        raise ValueError("start must be non-negative and requires a positive limit")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    end = min(start + limit, len(dataset)) if limit is not None else len(dataset)
    if start >= end:
        raise ValueError("Sample range is outside the MedQA test set")
    lm = StubLM() if stub else BitNetLM()
    samples = []
    for i in range(start, end):
        doc = dataset[i]
        choices = "\n".join(f"{letter}. {doc[f'ending{j}']}" for j, letter in enumerate("ABCD"))
        prompt = f"Answer with only one letter: A, B, C, or D.\nQuestion: {doc['sent1']}\n{choices}\nAnswer:"
        request = SimpleNamespace(args=(prompt, {"max_gen_toks": 10, "do_sample": False}), doc=doc, task_name="medqa")
        if stub:
            raw = "ABCD"[doc["label"]]
        else:
            raw = lm.generate_until([request])[0]
        answer = extract_answer(raw)
        gold = "ABCD"[doc["label"]]
        samples.append({"doc_id": i, "id": doc.get("id"), "response": raw, "prediction": answer,
                        "gold": gold, "correct": answer == gold})
    identity = {"server_model": "stub", "response_model": "stub",
                "server_model_verified": True} if stub else lm.model_identity()
    results = {
        "method": "medqa_generation_v1", "dataset": "GBaker/MedQA-USMLE-4-options-hf:test",
        "prompt": "Answer with only one letter: A, B, C, or D.",
        "extraction": "entire response must be A-D, optionally parenthesized or followed by a period",
        "user_model_label": model, **identity,
        "server": None if stub else f"http://{config.BITNET_HOST}:{config.BITNET_PORT}",
        "mode": config.BITNET_MODE, "temperature": 0.0, "max_gen_toks": 10,
        "start": start, "count": len(samples),
        "accuracy": sum(s["correct"] for s in samples) / len(samples),
        "unrecognized": sum(s["prediction"] is None for s in samples), "samples": samples,
    }
    output = Path("logs/benchmarks/medqa")
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output / f"generative_medqa_{timestamp}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generative MedQA accuracy: {results['accuracy']:.4f}; unrecognized: {results['unrecognized']}; {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generative MedQA (not likelihood-based)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model", type=str, default=None, help="User label, not a model selector")
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, model=args.model, stub=args.stub, start=args.start)
