from pathlib import Path
from unittest.mock import AsyncMock
import asyncio
import httpx

from src import models
from src.intelligence.citation_graph import SemanticScholarClient

models.DOWNLOAD_DIR = Path('C:/tmp/arxiv_mcp_test')

class DummyResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError('err', request=httpx.Request('GET','https://example.com'), response=self)

client = SemanticScholarClient()
client._client.get = AsyncMock(side_effect=[
    DummyResponse(200, {'paperId': 'S2ID'}),
    DummyResponse(200, {'title': 'Root Title', 'citationCount': 42}),
    DummyResponse(200, {'data': [{'citedPaper': {'paperId': 'S2REF', 'title': 'Ref Title', 'year': 2024, 'citationCount': 3, 'isInfluential': True, 'externalIds': {'ArXiv': '1706.00000'}}}]}),
    DummyResponse(200, {'data': [{'citingPaper': {'paperId': 'S2CITE', 'title': 'Citing Title', 'year': 2025, 'citationCount': 2, 'isInfluential': False, 'externalIds': {'ArXiv': '1706.11111'}}}]})
])

graph = asyncio.run(client.get_citation_graph('1706.03762', max_references=10, max_citations=10))
print(graph)
print('file exists', (models.DOWNLOAD_DIR / 'citations' / '1706.03762.json').exists())
