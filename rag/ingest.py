import glob
import re
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from rag import data_store
from common.formatters import file_to_document, site_to_documents
from common.logger import get_logger


logger = get_logger("ingest")


def _log_site(message: str) -> None:
    logger.info(message)


def _stored_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _split_long_text(text: str, max_chars: int) -> List[str]:
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue

        words = sentence.split()
        cur = ""
        for word in words:
            if len(word) > max_chars:
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
                continue
            if len(cur) + len(word) + 1 <= max_chars:
                cur = f"{cur} {word}".strip()
            else:
                if cur:
                    pieces.append(cur)
                cur = word
        if cur:
            pieces.append(cur)
    return pieces


def _word_safe_tail(text: str, overlap: int) -> str:
    if not text or overlap <= 0:
        return ""
    tail = text[-overlap:]
    idx = tail.find(" ")
    return tail if idx == -1 else tail[idx + 1:]


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    paras = []
    for p in re.split(r"\n{2,}", text):
        p = p.strip()
        if not p:
            continue
        paras.extend(_split_long_text(p, max_chars))

    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = _word_safe_tail(cur, overlap) if overlap and cur else ""
            if tail and len(tail) + len(p) + 2 > max_chars:
                tail = ""
            cur = (tail + "\n\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def ingest_folder(folder: str, replace_existing: bool = False) -> int:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    supported_suffixes = (".txt", ".md", ".html", ".htm", ".pdf")
    files = [
        Path(p)
        for p in glob.glob(str(folder_path / "**/*"), recursive=True)
        if Path(p).suffix.lower() in supported_suffixes
    ]
    new_chunks: List[Dict] = []
    for fp in files:
        document = file_to_document(fp)
        if not document:
            continue
        for i, ch in enumerate(chunk_text(document.text)):
            new_chunks.append({
                "id": f"doc:{document.source}:{i}",
                "text": ch,
                "source": document.source,
                "kind": document.kind,
                "meta": document.meta
            })
    if not new_chunks:
        return 0
    if replace_existing:
        return data_store.replace_chunks_by_paths({_stored_path(fp) for fp in files}, new_chunks)
    return data_store.add_chunks(new_chunks)


class CrawlConfig(BaseModel):
    base_url: str
    max_pages: int = 100
    timeout: int = 10
    user_agent: str = "rag-bot/1.0"


def crawl_site(cfg: CrawlConfig) -> int:
    all_chunks: List[Dict] = []
    _log_site(f"start base_url={cfg.base_url} max_pages={cfg.max_pages}")
    documents = site_to_documents(
        cfg.base_url,
        max_pages=cfg.max_pages,
        timeout=cfg.timeout,
        user_agent=cfg.user_agent,
    )
    _log_site(f"formatted documents={len(documents)}")
    for document in documents:
        chunks = chunk_text(document.text)
        _log_site(
            f"chunk source={document.source} stable_id={document.stable_id} "
            f"text_chars={len(document.text)} chunks={len(chunks)}"
        )
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "id": f"site:{document.source}:{i}",
                "text": ch,
                "source": document.source,
                "kind": document.kind,
                "meta": document.meta
            })
    if not all_chunks:
        _log_site("done added=0 reason=no_chunks")
        return 0
    before = len(data_store.rows)
    added = data_store.add_chunks(all_chunks)
    after = len(data_store.rows)
    _log_site(f"done candidate_chunks={len(all_chunks)} added={added} before={before} after={after}")
    if added == 0 and all_chunks:
        _log_site("done added=0 reason=all_candidate_chunk_ids_already_exist")
    return added
