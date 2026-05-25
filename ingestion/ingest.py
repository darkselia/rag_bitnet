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

from core import data_store


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = (tail + "\n\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def ingest_folder(folder: str) -> int:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [Path(p) for p in glob.glob(str(folder_path / "**/*"), recursive=True) if p.endswith((".txt", ".md"))]
    added = 0
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
                "meta": {"path": str(fp)}
            })
            added += 1
    if new_chunks:
        data_store.add_chunks(new_chunks)
    return added


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
    added = 0
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
            added += 1
        if new_chunks:
            data_store.add_chunks(new_chunks)

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
    return added
