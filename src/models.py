"""
Shared Pydantic models and configuration for arxiv-mcp.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────


def get_download_dir() -> Path:
    """Return the current download directory, reading from env each call."""
    default_download_dir = os.getenv("ARXIV_DOWNLOAD_DIR")
    if default_download_dir:
        download_dir = Path(default_download_dir).expanduser().resolve()
    else:
        download_dir = Path(tempfile.gettempdir()) / "arxiv_mcp_downloads"

    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


DOWNLOAD_DIR = get_download_dir()

MAX_RESULTS_DEFAULT = int(os.getenv("ARXIV_MAX_RESULTS", "10"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
RATE_LIMIT_DELAY = float(
    os.getenv("ARXIV_RATE_LIMIT_DELAY", "3.0")
)  # seconds between API calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
KEEP_PDFS = os.getenv("ARXIV_KEEP_PDFS", "true").lower() in ("1", "true", "yes")

# Hardened config for script performance and safety
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("ARXIV_MAX_CONCURRENT_DOWNLOADS", "4"))
MAX_CONCURRENT_PARSERS = int(os.getenv("ARXIV_MAX_CONCURRENT_PARSERS", "2"))
ARXIV_CACHE_DB = os.getenv(
    "ARXIV_CACHE_DB", ""
)  # optional sqlite path for caching metadata
PDF_MAX_SIZE_MB = int(os.getenv("ARXIV_PDF_MAX_SIZE_MB", "100"))
PDF_MAX_PAGES = int(os.getenv("ARXIV_PDF_MAX_PAGES", "500"))


# ── Pydantic Models ───────────────────────────────────────────────────────────


class Author(BaseModel):
    name: str


class PaperMetadata(BaseModel):
    """Metadata returned from arXiv API for a single paper."""

    arxiv_id: str
    title: str
    authors: list[Author]
    abstract: str
    categories: list[str]
    primary_category: str
    published: str  # ISO date string
    updated: str  # ISO date string
    pdf_url: str
    entry_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None


class TextChunk(BaseModel):
    """A single token-aware chunk of extracted paper text."""

    chunk_index: int
    text: str
    token_count: int
    section_hint: Optional[str] = None  # e.g., "Introduction", "Abstract"


class ExtractedPaper(BaseModel):
    """Full parsed content of a downloaded PDF."""

    arxiv_id: str
    title: str
    total_pages: int
    full_text: str
    chunks: list[TextChunk]
    extraction_method: str = "pymupdf"


class PaperContext(BaseModel):
    """LLM-ready context bundle for a paper."""

    metadata: PaperMetadata
    chunks: list[TextChunk]
    total_tokens: int
    chunk_count: int
    llm_system_prompt: str
    summary_prompt: str


class SearchResult(BaseModel):
    """Lightweight result for listing search hits."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract_snippet: str
    published: str
    categories: list[str]
    pdf_url: str


class DownloadResult(BaseModel):
    """Result of a PDF download operation."""

    arxiv_id: str
    local_path: str
    file_size_bytes: int
    success: bool
    error: Optional[str] = None
