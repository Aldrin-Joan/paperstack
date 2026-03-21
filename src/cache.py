# pragma: no cover
"""Optional SQLite-based metadata cache for arxiv-mcp."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Optional

from src.models import PaperMetadata
from src.logger import get_logger
import os
import time

log = get_logger("cache")

_lock = Lock()


def _get_db_path() -> Optional[Path]:
    db_path_env = os.getenv("ARXIV_CACHE_DB", "").strip()
    if not db_path_env:
        return None
    return Path(db_path_env).expanduser().resolve()


def _get_connection() -> Optional[sqlite3.Connection]:
    db_path = _get_db_path()
    if not db_path:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            metadata TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_paper_metadata(arxiv_id: str) -> Optional[PaperMetadata]:
    conn = _get_connection()
    if conn is None:
        return None

    try:
        row = conn.execute(
            "SELECT metadata FROM papers WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        if not row:
            return None

        payload = json.loads(row[0])
        return PaperMetadata(**payload)

    except Exception as exc:
        log.warning("cache read failed", arxiv_id=arxiv_id, error=str(exc))
        return None
    finally:
        conn.close()


def evict_stale_cache(max_age_hours: int = 168) -> int:
    """Evict metadata older than max_age_hours and return number deleted."""
    conn = _get_connection()
    if conn is None:
        return 0

    cutoff = int((__import__("time").time() - max_age_hours * 3600))
    deleted = 0

    with _lock:
        try:
            cursor = conn.execute("DELETE FROM papers WHERE updated_at < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
            log.info(
                "evicted stale cache rows", deleted=deleted, max_age_hours=max_age_hours
            )
        except Exception as exc:
            log.warning(
                "cache eviction failed", error=str(exc), max_age_hours=max_age_hours
            )
        finally:
            conn.close()

    return deleted


def purge_old_pdfs(max_age_days: int = 30) -> int:
    """Remove downloaded PDFs older than max_age_days from DOWNLOAD_DIR."""
    from src.models import get_download_dir

    deleted = 0
    cutoff = __import__("time").time() - max_age_days * 86400

    for path in Path(get_download_dir()).glob("*.pdf"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except Exception as exc:
            log.warning("failed to delete stale pdf", path=str(path), error=str(exc))

    log.info("purged stale PDFs", deleted=deleted, max_age_days=max_age_days)
    return deleted


def set_paper_metadata(metadata: PaperMetadata) -> None:
    conn = _get_connection()
    if conn is None:
        return

    with _lock:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO papers (arxiv_id, metadata, updated_at) VALUES (?, ?, ?)",
                (
                    metadata.arxiv_id,
                    json.dumps(metadata.model_dump()),
                    int(time.time()),
                ),
            )
            conn.commit()
        except Exception as exc:
            log.warning(
                "cache write failed", arxiv_id=metadata.arxiv_id, error=str(exc)
            )
        finally:
            conn.close()
