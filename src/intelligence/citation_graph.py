from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import models
from src.models import (
    CITATION_CACHE_TTL,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    S2_API_KEY,
    CitationGraph,
    CitationNode,
)
from src.logger import get_logger

log = get_logger("citation_graph")

_BASE_URL = "https://api.semanticscholar.org/graph/v1"


def _should_retry_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    return False


class SemanticScholarClient:
    """Client for Semantic Scholar graph API with rate-limiting, retry, and caching."""

    def __init__(self) -> None:
        headers: Dict[str, str] = {
            "User-Agent": "arxiv-mcp/1.0 (research tool; https://github.com/arxiv-mcp) Python/httpx"
        }
        api_key = S2_API_KEY.strip()
        if api_key:
            headers["x-api-key"] = api_key
            log.info("Using Semantic Scholar API key from env", source="S2_API_KEY")
        else:
            log.warning(
                "S2_API_KEY not set: using unauthenticated rate limit (1 qps), may hit 429",
            )

        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        self._lock = asyncio.Lock()
        self._last_call_time = 0.0
        self._min_interval = 0.1 if api_key else 1.0

    async def __aenter__(self) -> "SemanticScholarClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    async def _rate_limit(self) -> None:
        async with self._lock:
            now = time.monotonic()
            sleep = self._min_interval - (now - self._last_call_time)
            if sleep > 0:
                await asyncio.sleep(sleep)
            self._last_call_time = time.monotonic()

    @retry(
        retry=retry_if_exception(_should_retry_http_error),
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(MAX_RETRIES),
        reraise=True,
    )
    async def _http_get(
        self, path: str, params: Optional[dict] = None
    ) -> httpx.Response:
        await self._rate_limit()
        response = await self._client.get(path, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise
        return response

    async def _arxiv_to_s2_id(self, arxiv_id: str) -> Optional[str]:
        try:
            response = await self._http_get(
                f"/paper/arXiv:{arxiv_id}", params={"fields": "paperId"}
            )
            data = response.json()
            return data.get("paperId")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def _fetch_references(self, s2_id: str, limit: int) -> List[CitationNode]:
        response = await self._http_get(
            f"/paper/{s2_id}/references",
            params={
                "fields": "title,year,externalIds,citationCount,isInfluential",
                "limit": limit,
            },
        )
        data = response.json().get("data", [])
        nodes: List[CitationNode] = []

        for item in data:
            paper = item.get("citedPaper") or item.get("paper") or {}
            external_ids = paper.get("externalIds", {}) or {}
            paper_id = paper.get("paperId") or ""
            citation_count = paper.get("citationCount")
            nodes.append(
                CitationNode(
                    arxiv_id=external_ids.get("ArXiv") or external_ids.get("arXiv"),
                    s2_id=paper_id,
                    title=paper.get("title") or "",
                    year=paper.get("year"),
                    citation_count=citation_count if citation_count is not None else 0,
                    is_influential=paper.get("isInfluential") if paper.get("isInfluential") is not None else False,
                )
            )

        return nodes

    async def _fetch_citations(self, s2_id: str, limit: int) -> List[CitationNode]:
        response = await self._http_get(
            f"/paper/{s2_id}/citations",
            params={
                "fields": "title,year,externalIds,citationCount,isInfluential",
                "limit": limit,
            },
        )
        data = response.json().get("data", [])
        nodes: List[CitationNode] = []

        for item in data:
            paper = item.get("citingPaper") or item.get("paper") or {}
            external_ids = paper.get("externalIds", {}) or {}
            paper_id = paper.get("paperId") or ""
            citation_count = paper.get("citationCount")
            nodes.append(
                CitationNode(
                    arxiv_id=external_ids.get("ArXiv") or external_ids.get("arXiv"),
                    s2_id=paper_id,
                    title=paper.get("title") or "",
                    year=paper.get("year"),
                    citation_count=citation_count if citation_count is not None else 0,
                    is_influential=paper.get("isInfluential") if paper.get("isInfluential") is not None else False,
                )
            )

        return nodes

    async def get_citation_graph(
        self,
        arxiv_id: str,
        max_references: int = 50,
        max_citations: int = 50,
        influential_only: bool = False,
    ) -> CitationGraph:
        safe_id = arxiv_id.replace("/", "_")
        citations_dir = models.DOWNLOAD_DIR / "citations"
        citations_dir.mkdir(parents=True, exist_ok=True)
        cache_path = citations_dir / f"{safe_id}.json"

        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                fetched_at = datetime.fromisoformat(raw["fetched_at"])
                if (
                    datetime.utcnow() - fetched_at
                ).total_seconds() < CITATION_CACHE_TTL:
                    return CitationGraph(**raw)
            except Exception:
                pass

        s2_id = await self._arxiv_to_s2_id(arxiv_id)
        if not s2_id:
            return CitationGraph(
                root_arxiv_id=arxiv_id,
                root_title="",
                references=[],
                cited_by=[],
                reference_count=0,
                citation_count=0,
                fetched_at=datetime.utcnow().isoformat(),
            )

        try:
            paper_resp = await self._http_get(
                f"/paper/{s2_id}", params={"fields": "title,citationCount"}
            )
            paper_data = paper_resp.json()
            root_title = paper_data.get("title", "")
            root_citations = paper_data.get("citationCount", 0)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return CitationGraph(
                    root_arxiv_id=arxiv_id,
                    root_title="",
                    references=[],
                    cited_by=[],
                    reference_count=0,
                    citation_count=0,
                    fetched_at=datetime.utcnow().isoformat(),
                )
            raise

        references = await self._fetch_references(s2_id, max_references)
        cited_by = await self._fetch_citations(s2_id, max_citations)

        if influential_only:
            references = [node for node in references if node.is_influential]
            cited_by = [node for node in cited_by if node.is_influential]

        graph = CitationGraph(
            root_arxiv_id=arxiv_id,
            root_title=root_title,
            references=references,
            cited_by=cited_by,
            reference_count=len(references),
            citation_count=root_citations,
            fetched_at=datetime.utcnow().isoformat(),
        )

        cache_path.write_text(json.dumps(graph.model_dump()), encoding="utf-8")
        return graph
