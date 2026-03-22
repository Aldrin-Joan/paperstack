from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.intelligence.contribution_extractor import ContributionExtractor
from src.models import ComparisonReport, PaperContributions
from src.logger import get_logger

log = get_logger("paper_comparator")

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "comparison.txt"


class PaperComparator:
    """Compare commentary across multiple paper contributions."""

    def __init__(self, contributor: ContributionExtractor) -> None:
        self._contributor = contributor

    def _validate_comparability(
        self, contributions: List[PaperContributions]
    ) -> Optional[str]:
        domains = [
            c.task_domain.strip().lower() for c in contributions if c.task_domain
        ]
        domains = [d for d in domains if d]
        unique_domains = set(domains)

        if not unique_domains:
            return None

        if len(unique_domains) > 1:
            intersects = any(
                d1 in d2 or d2 in d1
                for d1 in unique_domains
                for d2 in unique_domains
                if d1 != d2
            )
            if not intersects:
                return (
                    "Papers are not comparable — task domains differ too significantly"
                )

        return None

    def _build_comparison_prompt(
        self, contributions: List[PaperContributions], metadatas: Dict[str, Any]
    ) -> str:
        if not _PROMPT_PATH.exists():
            raise FileNotFoundError(f"Comparison prompt not found: {_PROMPT_PATH}")

        template = _PROMPT_PATH.read_text(encoding="utf-8")

        papers_json = []
        for c in contributions:
            papers_json.append(
                {
                    "arxiv_id": c.arxiv_id,
                    "title": c.core_claim,
                    "task_domain": c.task_domain,
                    "proposed_method": c.proposed_method,
                    "key_results": c.key_results,
                    "limitations": c.limitations,
                    "datasets_used": c.datasets_used,
                    "novelty_type": c.novelty_type,
                }
            )

        payload = {
            "papers": papers_json,
            "metadata": metadatas,
        }

        return template.replace("{papers_json}", json.dumps(payload, indent=2))

    async def compare(self, arxiv_ids: List[str]) -> ComparisonReport:
        if len(arxiv_ids) < 2 or len(arxiv_ids) > 5:
            raise ValueError("PaperComparator.compare requires 2 to 5 paper IDs")

        unique_ids = list(
            dict.fromkeys(
                [arxiv_id.strip() for arxiv_id in arxiv_ids if arxiv_id.strip()]
            )
        )
        if len(unique_ids) != len(arxiv_ids):
            raise ValueError("Duplicate paper IDs are not allowed in comparison")

        import asyncio

        extraction_tasks = [
            self._contributor.extract(arxiv_id) for arxiv_id in unique_ids
        ]
        contributions = await asyncio.gather(*extraction_tasks)

        validation_error = self._validate_comparability(contributions)
        if validation_error:
            raise ValueError(validation_error)

        metadatas = {
            c.arxiv_id: {"title": c.core_claim, "task_domain": c.task_domain}
            for c in contributions
        }
        prompt = self._build_comparison_prompt(contributions, metadatas)

        try:
            lmm_response = await self._contributor._call_ollama(prompt)
            parsed = json.loads(lmm_response)
            # normalize the output
            report_data = {
                "paper_ids": parsed.get("paper_ids", unique_ids),
                "paper_titles": parsed.get(
                    "paper_titles", {c.arxiv_id: c.core_claim for c in contributions}
                ),
                "shared_task_domain": parsed.get("shared_task_domain"),
                "dimensions": parsed.get("dimensions", []),
                "conflicting_claims": parsed.get("conflicting_claims", []),
                "strongest_results": parsed.get("strongest_results", ""),
                "recommendation": parsed.get("recommendation", ""),
                "compared_at": parsed.get("compared_at", datetime.utcnow().isoformat()),
            }
            return ComparisonReport(**report_data)
        except Exception as exc:
            log.warning(
                "Comparison LLM parse failed, returning partial report",
                error=str(exc),
                arxiv_ids=arxiv_ids,
            )
            return ComparisonReport(
                paper_ids=unique_ids,
                paper_titles={c.arxiv_id: c.core_claim for c in contributions},
                shared_task_domain=None,
                dimensions=[],
                conflicting_claims=[],
                strongest_results="",
                recommendation=str(exc),
                compared_at=datetime.utcnow().isoformat(),
            )
