import glob
import re
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from common import config
from common.formatters import file_to_document, site_to_documents
from common.logger import get_logger
from rag import data_store

logger = get_logger("ingest")


def _log_site(message: str) -> None:
    logger.info(message)


def _stored_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


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


def _sections(text: str) -> List[str]:
    lines = text.splitlines(keepends=True)
    sections: List[str] = []
    current: List[str] = []
    fence_char = ""
    fence_length = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("\ufeff")
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not line[fence.end():].strip():
                fence_char = ""
            current.append(line)
            continue
        blank_before = i == 0 or not lines[i - 1].strip()
        blank_after = i + 1 == len(lines) or not lines[i + 1].strip()
        markdown = bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line.lstrip("\ufeff")))
        uppercase = (
                blank_before and blank_after and 3 <= len(stripped) <= 80
                and len(stripped.split()) <= 12 and stripped.isupper()
                and all(c.isalnum() or c.isspace() or c in "-–—:&'’" for c in stripped)
        )
        if not fence_char and (markdown or uppercase) and "".join(current).strip():
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def _chunk_section(text: str, max_chars: int, overlap: int) -> List[str]:
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


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    return [chunk for section in _sections(text) for chunk in _chunk_section(section, max_chars, overlap)]


def ingest_folder(folder: str, replace_existing: bool = False) -> int:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    supported_suffixes = (".txt", ".md", ".html", ".htm", ".pdf", ".docx", ".csv")
    files = sorted(
        Path(p)
        for p in glob.glob(str(folder_path / "**/*"), recursive=True)
        if Path(p).is_file() and Path(p).suffix.lower() in supported_suffixes
    )
    total_size = 0
    for fp in files:
        size = fp.stat().st_size
        if size > config.INGEST_MAX_FILE_BYTES:
            raise ValueError(f"File size exceeds limit: {fp.name}")
        total_size += size
    if total_size > config.INGEST_MAX_BATCH_BYTES:
        raise ValueError("Total file size exceeds ingestion limit")

    new_chunks: List[Dict] = []
    processed_paths: set[str] = set()
    for fp in files:
        document = file_to_document(fp)
        if not document:
            raise ValueError(f"No extractable text in file: {fp.name}")
        chunks = chunk_text(document.text)
        if not chunks:
            raise ValueError(f"No extractable text in file: {fp.name}")
        processed_paths.add(_stored_path(fp))
        for i, ch in enumerate(chunks):
            new_chunks.append({
                "id": f"doc:{_stored_path(fp)}:{i}",
                "text": ch,
                "source": document.source,
                "kind": document.kind,
                "meta": {**document.meta, "path": _stored_path(fp)}
            })
    if replace_existing and processed_paths:
        return data_store.replace_chunks_by_paths(processed_paths, new_chunks)
    return data_store.add_chunks(new_chunks)


class CrawlConfig(BaseModel):
    base_url: str
    max_pages: int = 100
    timeout: int = 10
    user_agent: str = "rag-bot/1.0"


def crawl_site(cfg: CrawlConfig, failed_urls: Optional[list[str]] = None) -> int:
    all_chunks: List[Dict] = []
    processed_ids: set[str] = set()
    _log_site(f"start base_url={cfg.base_url} max_pages={cfg.max_pages}")
    documents = site_to_documents(
        cfg.base_url,
        max_pages=cfg.max_pages,
        timeout=cfg.timeout,
        user_agent=cfg.user_agent,
        failed_urls=failed_urls,
    )
    _log_site(f"formatted documents={len(documents)}")
    for document in documents:
        chunks = chunk_text(document.text)
        _log_site(
            f"chunk source={document.source} stable_id={document.stable_id} "
            f"text_chars={len(document.text)} chunks={len(chunks)}"
        )
        if not chunks:
            if failed_urls is not None and document.source not in failed_urls:
                failed_urls.append(document.source)
            continue
        processed_ids.add(document.stable_id)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "id": f"{document.stable_id}:{i}",
                "text": ch,
                "source": document.source,
                "kind": document.kind,
                "meta": {**document.meta, "document_id": document.stable_id}
            })
    if not all_chunks:
        _log_site("done added=0 reason=no_chunks")
        return 0
    before = len(data_store.rows)
    added = data_store.replace_chunks_by_documents(processed_ids, all_chunks)
    after = len(data_store.rows)
    _log_site(f"done candidate_chunks={len(all_chunks)} added={added} before={before} after={after}")
    if added == 0 and all_chunks:
        _log_site("done added=0 reason=all_candidate_chunk_ids_already_exist")
    return added
