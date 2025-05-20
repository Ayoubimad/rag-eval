"""
Chunk enrichment module for enhancing document chunks with context from surrounding chunks.

This module provides a strategy for enriching chunks by incorporating relevant context
from preceding and succeeding chunks to make each chunk more self-contained and informative.
"""

from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from enrichment.base import ChunkEnrichmentStrategy
from executor.executor import run_in_executor
from tqdm import tqdm


class ContextualEnrichment(ChunkEnrichmentStrategy):
    """Strategy for enriching chunks with context from surrounding chunks."""

    def __init__(
        self,
        llm: ChatOpenAI,
        n_chunks: int = 2,
        max_concurrency: int = 5,
        show_progress_bar: bool = False,
    ):
        """
        Initialize the chunk enrichment strategy.

        Args:
            n_chunks: Number of surrounding chunks to include as context
            llm: LLM model to use for enrichment as ChatOpenAI form langchain_openai
            max_concurrency: Maximum number of concurrent LLM requests
            show_progress_bar: Whether to show a progress bar during processing
        """
        self.n_chunks = n_chunks
        self.llm = llm
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self.show_progress_bar = show_progress_bar

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Enrich chunks with context from surrounding chunks.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        return asyncio.run(self.enrich_chunks_async(chunks))

    async def enrich_chunks_async(self, chunks: List[str]) -> List[str]:
        """
        Async version: Enrich chunks with context from surrounding chunks concurrently.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks
        """
        if not chunks:
            return chunks

        if len(chunks) <= 1:
            return chunks

        if not self.llm:
            return chunks

        tasks = []
        for i, chunk in enumerate(chunks):
            start_idx = max(0, i - self.n_chunks)
            preceding = chunks[start_idx:i]

            end_idx = min(len(chunks), i + self.n_chunks + 1)
            succeeding = chunks[i + 1 : end_idx]

            tasks.append(self._enrich_chunk_async(chunk, preceding, succeeding))

        if self.show_progress_bar:
            enriched_chunks = []
            for task in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Enriching chunks"
            ):
                enriched_chunks.append(await task)
            enriched_chunks = [
                x for _, x in sorted(zip(range(len(tasks)), enriched_chunks))
            ]
        else:
            enriched_chunks = await asyncio.gather(*tasks)

        return enriched_chunks

    async def _process_chunk_with_semaphore(
        self, semaphore, chunk, preceding, succeeding
    ):
        """Process a chunk while respecting the concurrency semaphore."""
        async with semaphore:
            enriched_chunk = await self._enrich_chunk_async(
                chunk, preceding, succeeding
            )
            return enriched_chunk

    async def _enrich_chunk_async(
        self,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> str:
        """
        Asynchronously enrich a single chunk using the LLM.

        Returns the enriched chunk.
        """
        if not self.llm:
            return chunk_content

        prompt = self._build_prompt(chunk_content, preceding_chunks, succeeding_chunks)

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            enriched_content = response.content.strip()
            return enriched_content
        except Exception:
            return chunk_content

    def _enrich_chunk(
        self,
        chunk_content: str,
        preceding_chunks: List[str],
        succeeding_chunks: List[str],
    ) -> str:
        """
        Synchronous version: Enrich a single chunk using the LLM.
        """
        return asyncio.run(
            self._enrich_chunk_async(chunk_content, preceding_chunks, succeeding_chunks)
        )

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
            You are an expert in document chunk enrichment for retrieval augmented generation (RAG) systems. Your task is to improve the MAIN CHUNK by incorporating relevant context from surrounding chunks to make it more self-contained, coherent, and retrievable.

            MAIN CHUNK:
            {chunk_content}

            PRECEDING CONTEXT:
            {preceding_text}

            SUCCEEDING CONTEXT:
            {succeeding_text}

            Guidelines for RAG-optimized chunk enrichment:
            1. Preserve ALL factual information, technical details, and key concepts from the MAIN CHUNK.
            2. Add important context from surrounding chunks that completes partial ideas, clarifies references, or provides missing definitions.
            3. Ensure the chunk can be understood independently while maintaining connections to the broader document.
            4. Maintain consistent terminology, including specific technical terms, acronyms, and named entities.
            5. Resolve dangling references (e.g., "as mentioned above," "this approach," "these components") by including their antecedents.
            6. If the chunk contains the start/end of an enumerated list, include relevant items from surrounding chunks.
            7. Add contextual information that would improve semantic search relevance for related queries.
            8. Preserve chronological or logical flow across content boundaries.
            9. Do not add speculative information or new content not implied by the provided chunks.
            10. **Keep the final enriched chunk under {chunk_size} characters.**
            11. **Return only the enriched chunk without any meta-commentary, explanations, or additional formatting.**

            ENRICHED CHUNK:
    """
