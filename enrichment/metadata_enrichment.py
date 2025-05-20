"""
Metadata enrichment module for enhancing document chunks with metadata information.

This module provides a strategy for enriching chunks by adding metadata such as entities,
keywords, and document topic at the beginning of each chunk.
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from enrichment.base import ChunkEnrichmentStrategy


class MetadataEnrichment(ChunkEnrichmentStrategy):
    """Strategy for enriching chunks with metadata information."""

    def __init__(
        self,
        llm: ChatOpenAI,
        include_entities: bool = True,
        include_keywords: bool = True,
        include_topic: bool = True,
        max_entities: int = 10,
        max_keywords: int = 10,
    ):
        """
        Initialize the metadata enrichment strategy.

        Args:
            llm: LLM model to use for metadata extraction
            include_entities: Whether to include entities in the metadata
            include_keywords: Whether to include keywords in the metadata
            include_topic: Whether to include document topic in the metadata
            max_entities: Maximum number of entities to include
            max_keywords: Maximum number of keywords to include
        """
        self.llm = llm
        self.include_entities = include_entities
        self.include_keywords = include_keywords
        self.include_topic = include_topic
        self.max_entities = max_entities
        self.max_keywords = max_keywords

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Enrich chunks with metadata information.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks with metadata
        """
        if not chunks or not self.llm:
            return chunks

        document_topic = None
        if self.include_topic:
            sample_text = "\n\n".join(chunks)
            if len(sample_text) > 10000:
                sample_chunks = chunks[: min(10, len(chunks))]
                sample_text = "\n\n".join(sample_chunks)
            document_topic = self._extract_topic(sample_text)

        all_entities = []
        if self.include_entities:
            for chunk in chunks:
                chunk_entities = self._extract_entities(chunk)
                all_entities.extend(chunk_entities)
            seen = set()
            all_entities = [e for e in all_entities if not (e in seen or seen.add(e))]
            all_entities = all_entities[: self.max_entities]

        all_keywords = []
        if self.include_keywords:
            for chunk in chunks:
                chunk_keywords = self._extract_keywords(chunk)
                all_keywords.extend(chunk_keywords)
            seen = set()
            all_keywords = [k for k in all_keywords if not (k in seen or seen.add(k))]
            all_keywords = all_keywords[: self.max_keywords]

        enriched_chunks = []
        for chunk in chunks:
            enriched_chunk = self._add_metadata_to_chunk(
                chunk,
                all_entities if self.include_entities else None,
                all_keywords if self.include_keywords else None,
                document_topic,
            )
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the text."""
        sample = text[:5000] if len(text) > 5000 else text

        prompt = f"""You are an expert at identifying named entities in text.
        
                Task: Extract exactly {self.max_entities} most important named entities from the provided text.
                Focus on: People, organizations, locations, products, technologies, and specific terms.
                Format: Return ONLY a comma-separated list of entities with no explanations or additional text.
                Example output: "John Smith, Microsoft Corporation, San Francisco, iPhone 13, BERT"

                Text to analyze:
                {sample}
                """

        try:
            response = self.llm.invoke(prompt)
            entities = [
                entity.strip()
                for entity in response.content.split(",")
                if entity.strip()
            ]
            return entities[: self.max_entities]
        except Exception:
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the text."""
        sample = text[:5000] if len(text) > 5000 else text

        prompt = f"""You are an expert at identifying key concepts and terminology in text.
        
                    Task: Extract exactly {self.max_keywords} most important keywords or key phrases from the provided text.
                    Focus on: Technical terms, main concepts, recurring themes, and significant topics.
                    Format: Return ONLY a comma-separated list of keywords with no explanations or additional text.
                    Example output: "machine learning, climate change, neural networks, data privacy, GDP growth"

                    Text to analyze:
                    {sample}
                    """

        try:
            response = self.llm.invoke(prompt)
            keywords = [
                keyword.strip()
                for keyword in response.content.split(",")
                if keyword.strip()
            ]
            return keywords[: self.max_keywords]
        except Exception:
            return []

    def _extract_topic(self, text: str) -> str:
        """Extract the main topic of the document."""
        sample = text[:8000] if len(text) > 8000 else text

        prompt = f"""You are an expert at summarizing document topics with precision.
        
                Task: Identify the main topic or subject of the provided document in a concise phrase.
                Requirements:
                - The topic should be 3-8 words
                - Be specific rather than general
                - Capture the core subject matter, not peripheral details
                - Use formal, objective language
                Format: Return ONLY the topic phrase with no explanations, quotes, or additional text.
                Example output: "Quantum Computing Applications in Cryptography"

                Document to analyze:
                {sample}
                """

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return ""

    def _add_metadata_to_chunk(
        self,
        chunk: str,
        entities: Optional[List[str]],
        keywords: Optional[List[str]],
        topic: Optional[str],
    ) -> str:
        """Add metadata to the beginning of the chunk."""
        metadata_sections = []

        if topic:
            metadata_sections.append(f"TOPIC: {topic}")

        if entities and len(entities) > 0:
            metadata_sections.append(f"ENTITIES: {', '.join(entities)}")

        if keywords and len(keywords) > 0:
            metadata_sections.append(f"KEYWORDS: {', '.join(keywords)}")

        if metadata_sections:
            metadata_block = "\n".join(metadata_sections)
            return f"{metadata_block}\n\n{chunk}"
        else:
            return chunk
