from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import httpx

from src import models
from src.models import (
    CONTRIBUTION_CACHE_TTL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PaperContributions,
    ExtractedPaper,
)
from src.pdf_fetcher import PDFFetcher
from src.pdf_parser import PDFParser
from src.logger import get_logger

log = get_logger("contribution_extractor")

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "contributions.txt"


class ContributionExtractor:
    """Extract paper contributions using LLM with heuristic fallback and caching."""

    def __init__(self) -> None:
        self._ollama_url = OLLAMA_BASE_URL.rstrip("/")
        self._ollama_model = OLLAMA_MODEL

    def _build_prompt(self, paper: ExtractedPaper) -> str:
        """Build prompt for contribution extraction from parsed paper text."""
        if not _PROMPT_PATH.exists():
            raise FileNotFoundError(f"Contribution prompt not found: {_PROMPT_PATH}")

        template = _PROMPT_PATH.read_text(encoding="utf-8")

        body_text = paper.full_text
        truncated_body = body_text[:3000]

        prompt_text = (
            f"Title: {paper.title}\n\n"
            f"Abstract + body (first 3000 chars):\n{truncated_body}\n"
        )

        return template.replace("{paper_text}", prompt_text)

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama service to generate JSON response."""
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._ollama_url}/api/generate", json=payload
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "response" in data:
                return str(data["response"])
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "text" in first:
                    return str(first["text"])

            raise ValueError("Unexpected Ollama response format")

    def _heuristic_extract(self, paper: ExtractedPaper) -> PaperContributions:
        """Fallback heuristic extraction for contributions when LLM fails."""
        text = paper.full_text
        sentences = re.split(r"(?<=[\.\!?])\s+", text)

        core_claim = ""
        for sentence in sentences[:20]:
            lower = sentence.lower()
            if any(
                k in lower
                for k in ["we propose", "we present", "this paper", "we introduce"]
            ):
                core_claim = sentence.strip()
                break

        if not core_claim:
            core_claim = sentences[0].strip() if sentences else ""

        key_result_patterns = re.compile(
            r"\b\d+(?:\.\d+)?%|state[- ]of[- ]the[- ]art|sota\b", re.IGNORECASE
        )
        key_results = [s.strip() for s in sentences if key_result_patterns.search(s)]
        key_results = key_results[:3] if key_results else []

        datasets = []
        for match in re.finditer(
            r"([A-Z][A-Za-z0-9_\-]+(?: dataset| benchmark)?)", text
        ):
            candidate = match.group(1).strip()
            if candidate.lower() not in (d.lower() for d in datasets):
                datasets.append(candidate)
            if len(datasets) >= 3:
                break

        if not datasets:
            m = re.search(
                r"\b(CIFAR-10|CIFAR-100|ImageNet|SQuAD|GLUE|MNIST|COCO|OpenAI*)\b", text
            )
            if m:
                datasets.append(m.group(1))

        if "\bnlp\b" in text.lower():
            task_domain = "NLP"
        elif "\bcomputer vision\b" in text.lower() or "\bvision\b" in text.lower():
            task_domain = "Computer Vision"
        elif "\brobotics\b" in text.lower():
            task_domain = "Robotics"
        else:
            task_domain = "General"

        novelty_type = "empirical"

        for nt in [
            "architectural",
            "algorithmic",
            "empirical",
            "theoretical",
            "systems",
        ]:
            if nt in text.lower():
                novelty_type = nt
                break

        return PaperContributions(
            arxiv_id=paper.arxiv_id,
            core_claim=core_claim,
            proposed_method="",
            key_results=key_results,
            baselines_compared=[],
            limitations=[],
            datasets_used=datasets,
            task_domain=task_domain,
            novelty_type=novelty_type,
            extraction_method="heuristic",
            extracted_at=datetime.utcnow().isoformat(),
        )

    def _ensure_keys(self, data: Dict[str, Any], arxiv_id: str) -> Dict[str, Any]:
        required = {
            "core_claim": "",
            "proposed_method": "",
            "key_results": [],
            "baselines_compared": [],
            "limitations": [],
            "datasets_used": [],
            "task_domain": "",
            "novelty_type": "empirical",
        }

        merged: Dict[str, Any] = {
            **required,
            **{k: data[k] for k in data if k in required},
        }
        merged["arxiv_id"] = arxiv_id
        merged["extraction_method"] = "llm"
        merged["extracted_at"] = datetime.utcnow().isoformat()
        return merged

    def _parse_llm_output(self, output_text: str, arxiv_id: str) -> PaperContributions:
        """Parse LLM text output into PaperContributions with fallback JSON substring extraction."""
        try:
            data = json.loads(output_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group(0))

        filtered = self._ensure_keys(data, arxiv_id)

        return PaperContributions(**filtered)

    async def extract(
        self, arxiv_id: str, force_refresh: bool = False
    ) -> PaperContributions:
        """Extract contributions for a paper, using cache, LLM, and heuristic fallback."""
        arxiv_id = arxiv_id.strip()
        contributions_dir = models.DOWNLOAD_DIR / "contributions"
        contributions_dir.mkdir(parents=True, exist_ok=True)
        cache_path = contributions_dir / f"{arxiv_id}.json"

        if not force_refresh and cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                fetched_at = datetime.fromisoformat(
                    raw.get("extracted_at", datetime.utcnow().isoformat())
                )
                if (
                    datetime.utcnow() - fetched_at
                ).total_seconds() < CONTRIBUTION_CACHE_TTL:
                    return PaperContributions(**raw)
            except Exception:
                pass

        async with PDFFetcher() as fetcher:
            dl_result = await fetcher.download(arxiv_id)

        if not dl_result.success:
            raise RuntimeError(f"PDF download failed for {arxiv_id}: {dl_result.error}")

        paper = PDFParser().parse(dl_result.local_path, arxiv_id)

        prompt = self._build_prompt(paper)
        contributions: PaperContributions

        try:
            llm_result = await self._call_ollama(prompt)
            contributions = self._parse_llm_output(llm_result, arxiv_id)
        except Exception as exc:
            log.warning(
                "LLM contribution extraction failed, falling back to heuristic",
                error=str(exc),
                arxiv_id=arxiv_id,
            )
            contributions = self._heuristic_extract(paper)

        cache_path.write_text(json.dumps(contributions.model_dump()), encoding="utf-8")

        return contributions
