import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from common import config
from common.text_utils import tokenize
from rag.models import get_embed_model


@dataclass(frozen=True)
class Snapshot:
    rows: list[dict[str, Any]]
    corpus: list[str]
    index: Any
    bm25: Any


_lock = threading.RLock()
_snapshot = Snapshot([], [], None, None)
rows: list[dict[str, Any]] = []


def snapshot() -> Snapshot:
    return _snapshot


def _norm_path(path: str) -> str:
    return os.path.normcase(str(Path(path).resolve())).replace("\\", "/")


def _bm25(corpus: list[str]) -> Any:
    tokens = [tokenize(text) for text in corpus]
    return BM25Okapi(tokens) if any(tokens) else None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index() -> None:
    global _snapshot, rows
    pointer = config.DATA_DIR / "current.json"
    if not pointer.exists():
        if config.CORPUS_JSONL.exists() or config.FAISS_INDEX.exists():
            raise RuntimeError("Legacy index detected. Preserve it and reindex into a new DATA_DIR.")
        return
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    if Path(version).name != version:
        raise ValueError("Invalid index version")
    directory = config.DATA_DIR / "versions" / version
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != 2 or meta.get("version") != version or meta.get("embed_model") != config.EMBED_MODEL:
        raise ValueError("Index version or embedding model mismatch; reindex into a new DATA_DIR")
    for name, expected in meta["sha256"].items():
        if name not in {"rows.jsonl", "index.faiss"} or _digest(directory / name) != expected:
            raise ValueError("Index version checksum mismatch")
    if "rows.jsonl" not in meta["sha256"] or (meta["count"] and "index.faiss" not in meta["sha256"]):
        raise ValueError("Index version is incomplete")
    loaded_rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines() if
                   line]
    corpus = [r["text"] for r in loaded_rows]
    idx = faiss.read_index(str(directory / "index.faiss")) if corpus else None
    if len(corpus) != meta["count"] or (idx is not None and idx.ntotal != len(corpus)):
        raise ValueError("Index and corpus sizes do not match")
    new_snapshot = Snapshot(loaded_rows, corpus, idx, _bm25(corpus))
    rows = loaded_rows
    _snapshot = new_snapshot


def _commit(new_rows: list[dict[str, Any]]) -> None:
    global _snapshot, rows
    ids = [row["id"] for row in new_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    corpus = [r["text"] for r in new_rows]
    idx = None
    if corpus:
        model = get_embed_model()
        vectors = []
        for i in range(0, len(corpus), config.EMBED_BATCH):
            batch = corpus[i:i + config.EMBED_BATCH]
            vectors.append(
                model.encode(batch, batch_size=len(batch), show_progress_bar=False, normalize_embeddings=True))
        embeddings = np.vstack(vectors).astype("float32")
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        idx.add(embeddings)
    bm25 = _bm25(corpus)
    new_snapshot = Snapshot(new_rows, corpus, idx, bm25)
    version = uuid.uuid4().hex
    directory = config.DATA_DIR / "versions" / version
    directory.mkdir(parents=True)
    with (directory / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if idx is not None:
        faiss.write_index(idx, str(directory / "index.faiss"))
        if faiss.read_index(str(directory / "index.faiss")).ntotal != len(corpus):
            raise ValueError("Saved index size mismatch")
    files = ["rows.jsonl"] + (["index.faiss"] if idx is not None else [])
    meta = {**config.meta_payload(len(corpus)), "version": version, "format": 2,
            "sha256": {name: _digest(directory / name) for name in files}}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    pointer_tmp = config.DATA_DIR / f"current.{version}.tmp"
    pointer_tmp.write_text(json.dumps({"version": version}), encoding="utf-8")
    os.replace(pointer_tmp, config.DATA_DIR / "current.json")
    rows = new_rows
    _snapshot = new_snapshot


def replace_chunks_by_paths(paths: set[str], chunks: list[dict[str, Any]]) -> int:
    with _lock:
        normalized = {_norm_path(path) for path in paths}
        kept = [r for r in rows if _norm_path(str(r.get("meta", {}).get("path", ""))) not in normalized]
        _commit(kept + chunks)
        return len(chunks)


def replace_chunks_by_documents(document_ids: set[str], chunks: list[dict[str, Any]]) -> int:
    with _lock:
        kept = [r for r in rows if r.get("meta", {}).get("document_id") not in document_ids]
        _commit(kept + chunks)
        return len(chunks)


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    with _lock:
        ids = [c["id"] for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs in ingestion")
        existing = {r["id"] for r in rows}
        new_items = [c for c in chunks if c["id"] not in existing]
        if new_items:
            _commit(rows + new_items)
        return len(new_items)


load_index()
