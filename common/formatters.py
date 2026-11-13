import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO
from urllib.parse import urljoin, urldefrag
from zipfile import BadZipFile, ZipFile

import requests
import tldextract
from bs4 import BeautifulSoup, NavigableString
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError

from common import config
from common.logger import get_logger

logger = get_logger("ingest")
# A CSV field may occupy the entire permitted file.
csv.field_size_limit(config.INGEST_MAX_FILE_BYTES)


@dataclass(frozen=True)
class FormattedDocument:
    text: str
    source: str
    kind: str
    meta: dict = field(default_factory=dict)
    stable_id: str = ""


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _stored_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def text_file_to_document(path: Path) -> Optional[FormattedDocument]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_ws(text)
    if not text:
        return None
    return FormattedDocument(
        text=text,
        source=path.name,
        kind="docs",
        meta={"path": _stored_path(path)},
        stable_id=f"doc:{_stored_path(path)}",
    )


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        for header in soup.find_all("header"):
            for heading in header.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                header.insert_before(heading.extract())
            header.decompose()
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            title = heading.get_text(" ", strip=True)
            heading.replace_with(NavigableString(f"\n\n{'#' * int(heading.name[1])} {title}\n\n"))
        text = re.sub(r"[^\S\n]+", " ", soup.get_text(" "))
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()
    except Exception:
        return ""


def html_file_to_document(path: Path) -> Optional[FormattedDocument]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)
    if not text:
        return None
    return FormattedDocument(
        text=text,
        source=path.name,
        kind="docs",
        meta={"path": _stored_path(path)},
        stable_id=f"doc:{_stored_path(path)}",
    )


def pdf_file_to_document(path: Path) -> Optional[FormattedDocument]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            from PyPDF2.errors import PdfReadError as PyPdfError
        except ImportError as e:
            raise RuntimeError("PDF ingest requires pypdf or PyPDF2 to be installed") from e

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    except PyPdfError as e:
        raise ValueError(f"Invalid PDF file: {path.name}") from e
    text = normalize_ws("\n\n".join(pages))
    if not text:
        return None
    return FormattedDocument(
        text=text,
        source=path.name,
        kind="docs",
        meta={"path": _stored_path(path)},
        stable_id=f"doc:{_stored_path(path)}",
    )


def _docx_blocks(container: Any) -> list[str]:
    parts = []
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                style = block.style
                heading = re.fullmatch(r"Heading ([1-9])", style.name or "") if style else None
                parts.append(f"{'#' * min(int(heading.group(1)), 6)} {text}" if heading else text)
        elif isinstance(block, Table):
            for row in block.rows:
                cells = ["\n".join(_docx_blocks(cell)) for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    return parts


def docx_file_to_document(path: Path) -> Optional[FormattedDocument]:
    try:
        with ZipFile(path) as archive:
            expanded_size = sum(item.file_size for item in archive.infolist())
        if expanded_size > config.INGEST_MAX_DOCX_UNCOMPRESSED_BYTES:
            raise ValueError(f"DOCX uncompressed size exceeds limit: {path.name}")
        text = "\n\n".join(_docx_blocks(Document(str(path))))
    except (BadZipFile, KeyError, PackageNotFoundError, XMLSyntaxError) as e:
        raise ValueError(f"Invalid DOCX file: {path.name}") from e
    if not text:
        return None
    return FormattedDocument(
        text=text,
        source=path.name,
        kind="docs",
        meta={"path": _stored_path(path)},
        stable_id=f"doc:{_stored_path(path)}",
    )


def _csv_delimiter(f: TextIO, filename: str) -> str:
    header_candidates: list[str] = []
    valid_candidates: list[str] = []
    for delimiter in (",", ";"):
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter, strict=True)
        try:
            headers = next(reader, [])
            if len(headers) < 2:
                continue
            header_candidates.append(delimiter)
            for row in reader:
                if row and any(value.strip() for value in row) and len(row) != len(headers):
                    break
            else:
                valid_candidates.append(delimiter)
        except csv.Error:
            continue
    if len(valid_candidates) == 1:
        return valid_candidates[0]
    if len(valid_candidates) > 1 or (not valid_candidates and len(header_candidates) > 1):
        raise ValueError(f"Ambiguous CSV delimiter: {filename}")
    return header_candidates[0] if header_candidates else ","


def csv_file_to_document(path: Path) -> Optional[FormattedDocument]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            if not f.read(1):
                return None
            delimiter = _csv_delimiter(f, path.name)
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter, strict=True)
            headers = [header.strip() for header in next(reader, [])]
            if not headers or any(not header for header in headers):
                raise ValueError(f"CSV must have a header for each column: {path.name}")
            records = []
            for row in reader:
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(headers):
                    raise ValueError(f"CSV column count mismatch at line {reader.line_num}: {path.name}")
                records.append("\n".join(f"{header}: {value}" for header, value in zip(headers, row)))
    except UnicodeDecodeError as e:
        raise ValueError(f"CSV must be UTF-8: {path.name}") from e
    except csv.Error as e:
        raise ValueError(f"Invalid CSV file: {path.name}") from e
    if not records:
        return None
    return FormattedDocument(
        text="\n\n".join(records),
        source=path.name,
        kind="docs",
        meta={"path": _stored_path(path)},
        stable_id=f"doc:{_stored_path(path)}",
    )


def file_to_document(path: Path) -> Optional[FormattedDocument]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return text_file_to_document(path)
    if suffix in {".html", ".htm"}:
        return html_file_to_document(path)
    if suffix == ".pdf":
        return pdf_file_to_document(path)
    if suffix == ".docx":
        return docx_file_to_document(path)
    if suffix == ".csv":
        return csv_file_to_document(path)
    return None


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


def fetch_text(url: str, *, timeout: int, user_agent: str, content_types: Iterable[str]) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
        content_type = r.headers.get("Content-Type", "")
        expected = tuple(content_types)
        logger.info(
            f"fetch status={r.status_code} "
            f"bytes={len(r.content)}"
        )
        if r.status_code != 200:
            logger.info("skip reason=status_not_200")
            return None
        if not any(t in content_type for t in expected):
            logger.info("skip reason=unexpected_content_type")
            return None
        return r.text
    except Exception as e:
        logger.info("fetch failed error_type=%s", type(e).__name__)
        return None


def _is_skipped_asset(url: str) -> bool:
    return bool(re.search(r"\.(pdf|zip|jpg|jpeg|png|gif|svg|mp4|mp3|ico|webp|css)(?:\?|$)", url, re.I))


def discover_page_links(base_url: str, page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        url = canon_url(page_url, a.get("href"))
        if url and same_domain(base_url, url) and not _is_skipped_asset(url):
            urls.append(url)
    return list(dict.fromkeys(urls))


def discover_html_assets(base_url: str, page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for tag in soup.find_all(["script", "link"]):
        raw_url = tag.get("src") or tag.get("href")
        url = canon_url(page_url, raw_url)
        if url and same_domain(base_url, url) and not _is_skipped_asset(url):
            urls.append(url)
    return list(dict.fromkeys(urls))


def discover_json_urls_from_js(base_url: str, js_url: str, js_text: str) -> list[str]:
    urls = []
    for match in re.finditer(r'["\']([^"\']+\.json(?:\?[^"\']*)?)["\']', js_text, re.I):
        url = canon_url(js_url, match.group(1))
        if url and same_domain(base_url, url):
            urls.append(url)
    return list(dict.fromkeys(urls))


def _stringify_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _format_json_value(value: Any, prefix: str = "") -> list[str]:
    scalar = _stringify_json_value(value)
    if scalar:
        return [f"{prefix}: {scalar}" if prefix else scalar]

    if isinstance(value, dict):
        lines = []
        for key, nested_value in value.items():
            key_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_format_json_value(nested_value, key_prefix))
        return lines

    if isinstance(value, list):
        lines = []
        for i, nested_value in enumerate(value):
            item_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            lines.extend(_format_json_value(nested_value, item_prefix))
        return lines

    return []


def _record_id_from_json(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        for key in ("id", "_id", "uuid", "slug", "key"):
            record_id = _stringify_json_value(value.get(key))
            if record_id:
                parent = re.sub(r"\[\d+\]$", "", fallback)
                return f"{parent}:{key}:{record_id}"
    return fallback


def _has_scalar_fields(value: dict[str, Any]) -> bool:
    return any(_stringify_json_value(nested_value) for nested_value in value.values())


def _flatten_json_records(data: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(data, dict):
        if _has_scalar_fields(data):
            lines = _format_json_value(data)
            if lines:
                return [(_record_id_from_json(data, path or "root"), "\n".join(lines))]

        records = []
        for key, value in data.items():
            next_path = f"{path}.{key}" if path else str(key)
            records.extend(_flatten_json_records(value, next_path))
        return records

    if isinstance(data, list):
        records = []
        for i, value in enumerate(data):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            if isinstance(value, dict):
                lines = _format_json_value(value)
                if lines:
                    records.append((_record_id_from_json(value, item_path), "\n".join(lines)))
                continue

            if isinstance(value, list):
                nested = _flatten_json_records(value, item_path)
                if nested:
                    records.extend(nested)
                continue

            lines = _format_json_value(value)
            if lines:
                records.append((item_path, "\n".join(lines)))
        return records

    lines = _format_json_value(data)
    if lines:
        return [(path or "root", "\n".join(lines))]
    return []


def _record_failed(failed_urls: Optional[list[str]], url: str) -> None:
    if failed_urls is not None and url not in failed_urls:
        failed_urls.append(url)


def json_to_documents(
        json_url: str, json_text: str, *, failed_urls: Optional[list[str]] = None,
) -> list[FormattedDocument]:
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.info("json parse failed error_type=%s", type(e).__name__)
        _record_failed(failed_urls, json_url)
        return []

    documents = []
    for i, (record_id, text) in enumerate(_flatten_json_records(data)):
        text = normalize_ws(text)
        if not text:
            continue
        stable_id = f"site-json:{json_url}#{record_id or i}"
        documents.append(FormattedDocument(
            text=text,
            source=json_url,
            kind="site",
            meta={"url": json_url, "record_id": record_id},
            stable_id=stable_id,
        ))
    logger.info("json formatted documents=%s", len(documents))
    if not documents:
        _record_failed(failed_urls, json_url)
    return documents


def site_to_documents(
        base_url: str,
        *,
        max_pages: int,
        timeout: int,
        user_agent: str,
        failed_urls: Optional[list[str]] = None,
) -> list[FormattedDocument]:
    visited_pages = set()
    visited_assets = set()
    queue = [base_url]
    documents: list[FormattedDocument] = []
    logger.info("start max_pages=%s", max_pages)

    while queue and len(visited_pages) < max_pages:
        url = queue.pop(0)
        if url in visited_pages:
            continue
        visited_pages.add(url)
        logger.info("page start visited_pages=%s queue=%s", len(visited_pages), len(queue))

        html = fetch_text(url, timeout=timeout, user_agent=user_agent, content_types=("text/html",))
        if not html:
            logger.info("page skip reason=no_html")
            _record_failed(failed_urls, url)
            continue

        text = html_to_text(html)
        logger.info("page text html_chars=%s text_chars=%s", len(html), len(text))
        if len(text) >= config.SITE_MIN_TEXT_CHARS:
            documents.append(FormattedDocument(
                text=text,
                source=url,
                kind="site",
                meta={"url": url},
                stable_id=f"site-html:{url}",
            ))
            logger.info("page document added")
        else:
            logger.info("page document skipped reason=text_too_short threshold=%s", config.SITE_MIN_TEXT_CHARS)
            _record_failed(failed_urls, url)

        page_links = discover_page_links(base_url, url, html)
        assets = discover_html_assets(base_url, url, html)
        logger.info("page discovered links=%s assets=%s", len(page_links), len(assets))

        for link in page_links:
            if link not in visited_pages and link not in queue:
                queue.append(link)

        for asset_url in assets:
            if asset_url in visited_assets:
                continue
            visited_assets.add(asset_url)
            logger.info("asset start")

            if re.search(r"\.json(?:\?|$)", asset_url, re.I):
                json_text = fetch_text(
                    asset_url,
                    timeout=timeout,
                    user_agent=user_agent,
                    content_types=("application/json", "text/json", "text/plain"),
                )
                if json_text:
                    json_documents = json_to_documents(asset_url, json_text, failed_urls=failed_urls)
                    documents.extend(json_documents)
                    logger.info("asset json added documents=%s", len(json_documents))
                else:
                    logger.info("asset json skipped reason=no_text")
                    _record_failed(failed_urls, asset_url)
                continue

            if re.search(r"\.js(?:\?|$)", asset_url, re.I):
                js_text = fetch_text(
                    asset_url,
                    timeout=timeout,
                    user_agent=user_agent,
                    content_types=("javascript", "ecmascript", "text/plain"),
                )
                if not js_text:
                    logger.info("asset js skipped reason=no_text")
                    _record_failed(failed_urls, asset_url)
                    continue

                json_urls = discover_json_urls_from_js(base_url, asset_url, js_text)
                logger.info("asset js scanned js_chars=%s json_urls=%s", len(js_text), len(json_urls))
                for json_url in json_urls:
                    if json_url in visited_assets:
                        continue
                    visited_assets.add(json_url)
                    logger.info("js json start")
                    json_text = fetch_text(
                        json_url,
                        timeout=timeout,
                        user_agent=user_agent,
                        content_types=("application/json", "text/json", "text/plain"),
                    )
                    if json_text:
                        json_documents = json_to_documents(json_url, json_text, failed_urls=failed_urls)
                        documents.extend(json_documents)
                        logger.info("js json added documents=%s", len(json_documents))
                    else:
                        logger.info("js json skipped reason=no_text")
                        _record_failed(failed_urls, json_url)
            else:
                logger.info("asset skipped reason=unsupported_asset_type")

    logger.info(
        f"done pages={len(visited_pages)} "
        f"assets={len(visited_assets)} documents={len(documents)}"
    )
    return documents
