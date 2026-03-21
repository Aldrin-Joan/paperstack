# pragma: no cover
"""Periodic maintenance tasks for arxiv-mcp."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Callable

from src.cache import evict_stale_cache, purge_old_pdfs
from src.logger import get_logger

log = get_logger("maintenance")


async def periodic_cleanup(
    interval_hours: int = 24,
    cache_max_age_hours: int = 168,
    pdf_max_age_days: int = 30,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run maintenance in a periodic background task."""
    cache_db = os.getenv("ARXIV_CACHE_DB", "").strip()
    if cache_db:
        log.info("maintenance enabled", interval_hours=interval_hours)
    else:
        log.info("maintenance enabled (cache disabled)", interval_hours=interval_hours)

    stop_event = stop_event or asyncio.Event()

    while not stop_event.is_set():
        try:
            if os.getenv("ARXIV_CACHE_DB", "").strip():
                evict_stale_cache(cache_max_age_hours)

            purge_old_pdfs(pdf_max_age_days)

        except Exception as exc:
            log.error("maintenance failed", error=str(exc))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_hours * 3600)
        except asyncio.TimeoutError:
            pass


def schedule_periodic_maintenance(
    interval_hours: int = 24,
    cache_max_age_hours: int = 168,
    pdf_max_age_days: int = 30,
) -> Callable[[asyncio.AbstractEventLoop], asyncio.Task]:
    """Returns a factory to schedule background maintenance in an event loop."""

    def _factory(loop: asyncio.AbstractEventLoop) -> asyncio.Task:
        task = loop.create_task(
            periodic_cleanup(interval_hours, cache_max_age_hours, pdf_max_age_days)
        )
        task.add_done_callback(
            lambda t: log.warning(
                "maintenance task ended", exc=t.exception() if t.exception() else None
            )
        )
        return task

    return _factory
