import glob
import hashlib
import queue
import re
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urldefrag

import requests
import tldextract
from bs4 import BeautifulSoup
from pydantic import BaseModel

from rag import data_store


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


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
            tail = cur[-overlap:] if overlap and cur else ""
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
    files = [Path(p) for p in glob.glob(str(folder_path / "**/*"), recursive=True) if p.endswith((".txt", ".md"))]
    new_chunks: List[Dict] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        text = normalize_ws(text)
        if not text:
            continue
        for i, ch in enumerate(chunk_text(text)):
            cid = f"doc:{fp.name}:{i}"
            new_chunks.append({
                "id": cid,
                "text": ch,
                "source": str(fp.name),
                "kind": "docs",
                "meta": {"path": _stored_path(fp)}
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


def same_domain(u1: str, u2: str) -> bool:
    d1 = tldextract.extract(u1)
    d2 = tldextract.extract(u2)
    return (d1.domain, d1.suffix) == (d2.domain, d2.suffix)


def canon_url(base: str, url: str) -> Optional[str]:
    if not url:
        return None
    joined = urljoin(base, url)
    clean, _ = urldefrag(joined)
    return clean


def fetch_html(url: str, cfg: CrawlConfig) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout)
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            return None
        return r.text
    except Exception:
        return None


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def crawl_site(cfg: CrawlConfig) -> int:
    visited = set()
    q = queue.Queue()
    q.put(cfg.base_url)
    all_chunks: List[Dict] = []
    while not q.empty() and len(visited) < cfg.max_pages:
        url = q.get()
        if url in visited:
            continue
        visited.add(url)
        html = fetch_html(url, cfg)
        if not html:
            continue
        text = html_to_text(html)
        if not text or len(text) < 200:
            continue
        new_chunks = []
        for i, ch in enumerate(chunk_text(text)):
            digest = hashlib.md5((url + str(i)).encode("utf-8")).hexdigest()[:8]
            cid = f"site:{digest}:{i}"
            new_chunks.append({
                "id": cid,
                "text": ch,
                "source": url,
                "kind": "site",
                "meta": {"url": url}
            })
        all_chunks.extend(new_chunks)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            nu = canon_url(url, a.get("href"))
            if not nu:
                continue
            if not same_domain(cfg.base_url, nu):
                continue
            if nu in visited:
                continue
            if re.search(r"\.(pdf|zip|jpg|png|gif|svg|mp4|mp3)$", nu, re.I):
                continue
            q.put(nu)
    if not all_chunks:
        return 0
    return data_store.add_chunks(all_chunks)
