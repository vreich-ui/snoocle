"""Fetch and locally extract chord-sheet text from supported resources."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import httpx

from ..config import settings
from .fetch import extract_sheet_text

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


@dataclass(frozen=True)
class FetchedResource:
    url: str
    content: bytes
    content_type: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def _google_export_url(url: str) -> str:
    """Public Google Docs pages are far cleaner through their text export."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.casefold() not in {"docs.google.com", "drive.google.com"}:
        return url
    match = re.search(r"/document/d/([^/]+)", parsed.path)
    if not match:
        return url
    return f"https://docs.google.com/document/d/{match.group(1)}/export?format=txt"


def fetch_resource(url: str) -> FetchedResource:
    requested = _google_export_url(url)
    response = httpx.get(
        requested,
        headers={"User-Agent": _UA, "Accept-Language": "en"},
        timeout=settings.fetch_timeout_seconds,
        follow_redirects=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not content_type:
        content_type = _content_type_from_url(str(response.url))
    return FetchedResource(url=url, content=response.content, content_type=content_type)


def _content_type_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.casefold()
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if path.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
        return "image/*"
    return "text/html"


def normalize_resource(value, *, url: str) -> FetchedResource:
    """Compatibility seam for existing tests/fetch functions returning text."""
    if isinstance(value, FetchedResource):
        return value
    if isinstance(value, str):
        content_type = _content_type_from_url(url)
        if content_type == "application/pdf":
            return FetchedResource(url=url, content=value.encode("latin1"), content_type=content_type)
        return FetchedResource(url=url, content=value.encode("utf-8"), content_type="text/html")
    if isinstance(value, bytes):
        return FetchedResource(url=url, content=value, content_type=_content_type_from_url(url))
    raise TypeError(f"unsupported fetch result {type(value).__name__}")


def extract_resource_text(resource: FetchedResource) -> str:
    kind = resource.content_type.casefold()
    if kind in {"application/pdf", "application/x-pdf"} or resource.content.startswith(b"%PDF"):
        return _extract_pdf(resource.content)
    if "wordprocessingml" in kind or resource.url.casefold().endswith(".docx"):
        return _extract_docx(resource.content)
    if kind.startswith("image/") or kind == "image/*":
        return _extract_image(resource.content, kind)
    text = resource.content.decode("utf-8", errors="replace")
    if kind in {"text/plain", "text/markdown"}:
        return text
    return extract_sheet_text(text)


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(namespace + "t"))
        lines.append(text)
    return "\n".join(lines)


def _extract_image(content: bytes, content_type: str) -> str:
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("image OCR unavailable: tesseract is not installed")
    suffix = ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        suffix = ".jpg"
    with tempfile.TemporaryDirectory(prefix="snoocle-source-") as temp_dir:
        path = Path(temp_dir) / ("sheet" + suffix)
        path.write_bytes(content)
        result = subprocess.run(
            [executable, str(path), "stdout"], capture_output=True, text=True, timeout=60
        )
    if result.returncode != 0:
        raise RuntimeError(f"image OCR failed: {result.stderr.strip()[:300]}")
    return result.stdout
