from typing import List
from abc import ABC, abstractmethod


class ChunkEnrichmentStrategy(ABC):
    @abstractmethod
    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """Enrich the chunks with additional information"""
        """
        Args:
            chunks: List of chunks to enrich

        Returns:
            List of enriched chunks
        """
        raise NotImplementedError("Subclasses must implement this method")
