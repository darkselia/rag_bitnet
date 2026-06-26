from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lm_eval.utils import make_table


def save_results(
        results: dict,
        *,
        output_dir: Path,
        stem: str = "results",
        model_name: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mn = f"_{model_name}" if model_name else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{stem}{mn}_{ts}.json"
    summary_path = output_dir / f"{stem}{mn}_{ts}_summary.txt"
    latest_json = output_dir / f"{stem}{mn}_latest.json"
    latest_summary = output_dir / f"{stem}{mn}_latest_summary.txt"

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
