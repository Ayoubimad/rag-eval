"""
Chunk enrichment module for enhancing document chunks with context from surrounding chunks.

This module provides a strategy for enriching chunks by incorporating relevant context
from preceding and succeeding chunks to make each chunk more self-contained and informative.
"""

from typing import List
from langchain_openai import ChatOpenAI
from enrichment.base import ChunkEnrichmentStrategy


class ContextualEnrichment(ChunkEnrichmentStrategy):
    """Strategy for enriching chunks with context from surrounding chunks."""

    def __init__(
        self,
        llm: ChatOpenAI,
        n_chunks: int = 2,
    ):
        """
        Initialize the chunk enrichment strategy.

        Args:
            n_chunks: Number of surrounding chunks to include as context
            llm: LLM model to use for enrichment as ChatOpenAI form langchain_openai
        """
        self.n_chunks = n_chunks
        self.llm = llm

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Enrich chunks with context from surrounding chunks.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        if not chunks or len(chunks) <= 1 or not self.llm:
            return chunks

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            start_idx = max(0, i - self.n_chunks)
            preceding = chunks[start_idx:i]

            end_idx = min(len(chunks), i + self.n_chunks + 1)
            succeeding = chunks[i + 1 : end_idx]

            enriched_chunk = self._enrich_chunk(chunk, preceding, succeeding)
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks

    def _enrich_chunk(
        self,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> str:
        """
        Enrich a single chunk using the LLM.

        Returns the enriched chunk.
        """
        if not self.llm:
            return chunk_content

        prompt = self._build_prompt(chunk_content, preceding_chunks, succeeding_chunks)

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return chunk_content

    def _build_prompt(
        self,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> str:
        """Build the prompt for chunk enrichment."""
        preceding_text = "\n\n".join(preceding_chunks) if preceding_chunks else ""
        succeeding_text = "\n\n".join(succeeding_chunks) if succeeding_chunks else ""
        chunk_size = len(chunk_content) + 100

        return f"""
            You are a contextual enrichment expert. Your task is to revise the MAIN CHUNK below by naturally incorporating relevant information from the PRECEDING and SUCCEEDING CONTEXTS to improve clarity, flow, and self-containment.

            MAIN CHUNK:
            {chunk_content}

            PRECEDING CONTEXT:
            {preceding_text}

            SUCCEEDING CONTEXT:
            {succeeding_text}

            Guidelines:
            1. Preserve the original meaning and technical detail of the MAIN CHUNK.
            2. Seamlessly integrate only directly relevant context from the surrounding chunks.
            3. Ensure smooth transitions; do not insert disjointed or redundant text.
            4. The result must stand alone, clear and coherent without relying on outside text.
            5. Maintain existing terminology, references, and tone.
            6. Do not introduce new information not present in the provided context.
            7. Do not remove information from the MAIN CHUNK.
            8. **Keep the final enriched chunk under {chunk_size} characters.**
            9. **Return only the enriched chunk. No explanations, comments, or formatting.**

            ENRICHED CHUNK:
    """
