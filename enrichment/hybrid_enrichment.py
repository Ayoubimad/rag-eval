"""
Hybrid enrichment module combining metadata and contextual enrichment strategies.

This module provides a combined strategy for enriching chunks with both metadata
and contextual information from surrounding chunks.
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from enrichment.base import ChunkEnrichmentStrategy
from enrichment.metadata_enrichment import MetadataEnrichment
from enrichment.contextual_enrichment import ContextualEnrichment


class HybridEnrichment(ChunkEnrichmentStrategy):
    """Strategy combining metadata and contextual enrichment."""

    def __init__(
        self,
        llm: ChatOpenAI,
        n_chunks: int = 2,
        include_entities: bool = True,
        include_keywords: bool = True,
        include_topic: bool = True,
        max_entities: int = 5,
        max_keywords: int = 7,
        metadata_first: bool = False,
    ):
        """
        Initialize the hybrid enrichment strategy.

        Args:
            llm: LLM model to use for enrichment
            n_chunks: Number of surrounding chunks to include as context
            include_entities: Whether to include entities in the metadata
            include_keywords: Whether to include keywords in the metadata
            include_topic: Whether to include document topic in the metadata
            max_entities: Maximum number of entities to include
            max_keywords: Maximum number of keywords to include
            metadata_first: Whether to apply metadata enrichment before contextual enrichment
        """
        self.metadata_enricher = MetadataEnrichment(
            llm=llm,
            include_entities=include_entities,
            include_keywords=include_keywords,
            include_topic=include_topic,
            max_entities=max_entities,
            max_keywords=max_keywords,
        )
        self.contextual_enricher = ContextualEnrichment(
            llm=llm,
            n_chunks=n_chunks,
        )
        self.metadata_first = metadata_first

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Enrich chunks with both metadata and contextual information.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        if not chunks:
            return chunks

        if self.metadata_first:
            metadata_enriched = self.metadata_enricher.enrich_chunks(chunks)
            return self.contextual_enricher.enrich_chunks(metadata_enriched)
        else:
            context_enriched = self.contextual_enricher.enrich_chunks(chunks)
            return self.metadata_enricher.enrich_chunks(context_enriched)
