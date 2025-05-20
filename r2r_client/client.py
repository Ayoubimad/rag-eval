"""
A simple R2R Client for RAG evaluation
"""

from r2r import R2RClient as _R2RClient
from typing import List
from r2r_client.config import GenerationConfig, SearchSettings
import logging
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "\033[90m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - \033[92m%(levelname)s\033[0m - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class R2RClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = _R2RClient(base_url=self.base_url, timeout=self.timeout)

    def get_base_url(self) -> str:
        return self.base_url

    def get_client(self) -> _R2RClient:
        return self.client

    def delete_all_documents(self) -> None:
        """Delete all documents from R2R"""
        while True:
            documents = self.client.documents.list(limit=1000)
            doc_count = len(documents.results)

            if doc_count == 0:
                return

            for doc in documents.results:
                self.client.documents.delete(str(doc.id))

    def ingest_chunks(self, chunks: List[str]) -> None:
        """Ingest chunks into R2R"""
        self.client.documents.create(chunks=chunks)

    def process_rag_query(
        self,
        question: str,
        generation_config: GenerationConfig,
        search_settings: SearchSettings,
        search_mode: str,
    ) -> str:
        """Process a RAG query"""
        response = self.client.retrieval.rag(
            query=question,
            rag_generation_config=generation_config.to_dict(),
            search_mode=search_mode,
            search_settings=search_settings.to_dict(),
        )
        return response
