from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lm_eval.utils import make_table

from common import config


def sample_window(
        task: str,
        start: int,
        limit: int | None,
) -> tuple[int | None, dict[str, list[int]] | None]:
    if start < 0:
        raise ValueError("start must be >= 0")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if start == 0:
        return limit, None
    if limit is None:
        raise ValueError("start requires a limit")
    return None, {task: list(range(start, start + limit))}


def save_results(
        results: dict,
        *,
        output_dir: Path,
        stem: str = "results",
        model_name: str | None = None,
        model: object | None = None,
        start: int = 0,
        limit: int | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mn = f"_{model_name}" if model_name else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"{stem}{mn}_{ts}.json"
    summary_path = output_dir / f"{stem}{mn}_{ts}_summary.txt"
    latest_json = output_dir / f"{stem}{mn}_latest.json"
    latest_summary = output_dir / f"{stem}{mn}_latest_summary.txt"

    identity = model.model_identity() if hasattr(model, "model_identity") else {
        "server_model": None, "response_model": None, "server_model_verified": False,
    }
    results["run_metadata"] = {
        "user_model_label": model_name, **identity,
        "server": getattr(model, "_base_url", None), "mode": getattr(model, "_mode", None),
        "default_temperature": config.LLM_TEMPERATURE, "start": start, "limit": limit,
        "method_version": "lm_eval_adapter_v2",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    table = make_table(results)
    for path in (summary_path, latest_summary):
        with open(path, "w", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")

    return json_path, summary_path
