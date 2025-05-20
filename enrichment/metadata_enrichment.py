"""
Metadata enrichment module for enhancing document chunks with metadata information.

This module provides a strategy for enriching chunks by adding metadata such as entities,
keywords, and document topic at the beginning of each chunk.
"""

from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from enrichment.base import ChunkEnrichmentStrategy
from executor.executor import run_in_executor
from tqdm import tqdm


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
        max_concurrency: int = 5,
        show_progress_bar: bool = False,
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
            max_concurrency: Maximum number of concurrent LLM requests
            show_progress_bar: Whether to show a progress bar during processing
        """
        self.llm = llm
        self.include_entities = include_entities
        self.include_keywords = include_keywords
        self.include_topic = include_topic
        self.max_entities = max_entities
        self.max_keywords = max_keywords
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self.show_progress_bar = show_progress_bar

    def enrich_chunks(self, chunks: List[str]) -> List[str]:
        """
        Enrich chunks with metadata information using an iterative approach.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks with metadata
        """
        return asyncio.run(self.enrich_chunks_async(chunks))

    async def enrich_chunks_async(self, chunks: List[str]) -> List[str]:
        """
        Async version: Enrich chunks with metadata information concurrently.

        Args:
            chunks: List of raw text chunks to enrich

        Returns:
            List of enriched text chunks with metadata
        """
        if not chunks or not self.llm:
            return chunks

        document_topic = ""
        all_entities = []
        all_keywords = []

        # First pass - extract metadata from each chunk
        chunks_iter = (
            tqdm(enumerate(chunks), total=len(chunks), desc="Extracting metadata")
            if self.show_progress_bar
            else enumerate(chunks)
        )

        for i, chunk in chunks_iter:
            if self.include_topic:
                if i == 0:
                    document_topic = await self._extract_topic_async(chunk)
                else:
                    document_topic = await self._update_topic_async(
                        chunk, document_topic
                    )

            if self.include_entities:
                chunk_entities = await self._extract_entities_with_context_async(
                    chunk, all_entities
                )
                for entity in chunk_entities:
                    if entity not in all_entities:
                        all_entities.append(entity)
                all_entities = all_entities[: self.max_entities]

            if self.include_keywords:
                chunk_keywords = await self._extract_keywords_with_context_async(
                    chunk, all_keywords
                )
                for keyword in chunk_keywords:
                    if keyword not in all_keywords:
                        all_keywords.append(keyword)
                all_keywords = all_keywords[: self.max_keywords]

        # Second pass - enrich each chunk with the collected metadata
        # This can be done concurrently
        tasks = []
        for chunk in chunks:
            tasks.append(
                self._add_metadata_to_chunk_async(
                    chunk,
                    all_entities if self.include_entities else None,
                    all_keywords if self.include_keywords else None,
                    document_topic,
                )
            )

        if self.show_progress_bar:
            enriched_chunks = []
            for task in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Enriching chunks"
            ):
                enriched_chunks.append(await task)
            # Maintain original order
            enriched_chunks = [
                x for _, x in sorted(zip(range(len(tasks)), enriched_chunks))
            ]
        else:
            enriched_chunks = await asyncio.gather(*tasks)

        return enriched_chunks

    async def _extract_entities_async(self, text: str) -> List[str]:
        """Asynchronously extract named entities from the text."""
        prompt = f"""You are an expert at identifying named entities in text.
        
                Task: Extract exactly {self.max_entities} most important named entities from the provided text.
                Focus on: People, organizations, locations, products, technologies, and specific terms.
                Format: Return ONLY a comma-separated list of entities with no explanations or additional text.
                Example output: "John Smith, Microsoft Corporation, San Francisco, iPhone 13, BERT"

                Text to analyze:
                {text}
                """

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            entities = [
                entity.strip()
                for entity in response.content.split(",")
                if entity.strip()
            ]
            return entities[: self.max_entities]
        except Exception:
            return []

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the text (synchronous wrapper)."""
        return asyncio.run(self._extract_entities_async(text))

    async def _extract_keywords_async(self, text: str) -> List[str]:
        """Asynchronously extract important keywords from the text."""
        prompt = f"""You are an expert at identifying key concepts and terminology in text.
        
                    Task: Extract exactly {self.max_keywords} most important keywords or key phrases from the provided text.
                    Focus on: Technical terms, main concepts, recurring themes, and significant topics.
                    Format: Return ONLY a comma-separated list of keywords with no explanations or additional text.
                    Example output: "machine learning, climate change, neural networks, data privacy, GDP growth"

                    Text to analyze:
                    {text}
                    """

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            keywords = [
                keyword.strip()
                for keyword in response.content.split(",")
                if keyword.strip()
            ]
            return keywords[: self.max_keywords]
        except Exception:
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the text (synchronous wrapper)."""
        return asyncio.run(self._extract_keywords_async(text))

    async def _extract_topic_async(self, text: str) -> str:
        """Asynchronously extract the main topic of the document."""
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
                {text}
                """

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            return response.content.strip()
        except Exception:
            return ""

    def _extract_topic(self, text: str) -> str:
        """Extract the main topic of the document (synchronous wrapper)."""
        return asyncio.run(self._extract_topic_async(text))

    async def _add_metadata_to_chunk_async(
        self,
        chunk: str,
        entities: Optional[List[str]],
        keywords: Optional[List[str]],
        topic: Optional[str],
    ) -> str:
        """Add metadata to the beginning of the chunk in a format optimized for RAG systems."""
        metadata_sections = []

        if topic:
            metadata_sections.append(f"TOPIC: {topic}")

        if entities and len(entities) > 0:
            metadata_sections.append(f"ENTITIES: {', '.join(entities)}")

        if keywords and len(keywords) > 0:
            metadata_sections.append(f"KEYWORDS: {', '.join(keywords)}")

        if metadata_sections:
            metadata_block = "\n".join(metadata_sections)
            return f"""<metadata>
                    {metadata_block}
                    </metadata>
                    <content>
                    {chunk}
                    </content>"""
        else:
            return f"""<content>
                    {chunk}
                    </content>"""

    def _add_metadata_to_chunk(
        self,
        chunk: str,
        entities: Optional[List[str]],
        keywords: Optional[List[str]],
        topic: Optional[str],
    ) -> str:
        """Add metadata to the chunk (synchronous wrapper)."""
        return self._add_metadata_to_chunk_async(chunk, entities, keywords, topic)

    async def _extract_entities_with_context_async(
        self, text: str, existing_entities: List[str]
    ) -> List[str]:
        """Asynchronously extract named entities from the text, considering already identified entities."""
        sample = text
        existing_str = ", ".join(existing_entities) if existing_entities else "None yet"

        prompt = f"""You are an expert at identifying named entities in text.
    
            Task: Extract important named entities from the provided text.
            Previously identified entities: {existing_str}
            
            Instructions:
            - Confirm which previously identified entities are relevant to this text
            - Add new important entities that weren't previously identified
            - Prioritize entities that appear to be significant in the overall document
            - Return at most {self.max_entities} entities total
            
            Format: Return ONLY a comma-separated list of entities with no explanations.
            
            Text to analyze:
            {sample}
            """

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            entities = [
                entity.strip()
                for entity in response.content.split(",")
                if entity.strip()
            ]
            return entities
        except Exception:
            return existing_entities.copy() if existing_entities else []

    def _extract_entities_with_context(
        self, text: str, existing_entities: List[str]
    ) -> List[str]:
        """Extract named entities with context (synchronous wrapper)."""
        return asyncio.run(
            self._extract_entities_with_context_async(text, existing_entities)
        )

    async def _extract_keywords_with_context_async(
        self, text: str, existing_keywords: List[str]
    ) -> List[str]:
        """Asynchronously extract keywords from the text, considering already identified keywords."""
        sample = text
        existing_str = ", ".join(existing_keywords) if existing_keywords else "None yet"

        prompt = f"""You are an expert at identifying key concepts and terminology in text.
    
                Task: Extract important keywords or key phrases from the provided text.
                Previously identified keywords: {existing_str}
                
                Instructions:
                - Confirm which previously identified keywords are relevant to this text
                - Add new important keywords that weren't previously identified
                - Prioritize keywords that appear to be significant in the overall document
                - Return at most {self.max_keywords} keywords total
                
                Format: Return ONLY a comma-separated list of keywords with no explanations.
                
                Text to analyze:
                {sample}
                """

        try:
            response = await run_in_executor(self.executor, self.llm.invoke, prompt)
            keywords = [
                keyword.strip()
                for keyword in response.content.split(",")
                if keyword.strip()
            ]
            return keywords
        except Exception:
            return existing_keywords.copy() if existing_keywords else []

    def _extract_keywords_with_context(
        self, text: str, existing_keywords: List[str]
    ) -> List[str]:
        """Extract keywords with context (synchronous wrapper)."""
        return asyncio.run(
            self._extract_keywords_with_context_async(text, existing_keywords)
        )

    async def _update_topic_async(self, text: str, current_topic: str) -> str:
        """Asynchronously update the document topic based on new information."""
        prompt = f"""You are an expert at summarizing document topics with precision.
    
            Task: Refine or confirm the current topic based on new information.
            Current topic: "{current_topic}"
            
            Instructions:
            - If the new text supports the current topic, return the current topic
            - If the new text suggests a more accurate or broader topic, update it
            - The topic should remain 3-8 words and specific rather than general
            
            Format: Return ONLY the topic phrase with no explanations or additional text.
            
            New text to analyze:
            {text}
            """

        response = await run_in_executor(self.executor, self.llm.invoke, prompt)
        return response.content.strip()

    def _update_topic(self, text: str, current_topic: str) -> str:
        """Update the document topic (synchronous wrapper)."""
        return asyncio.run(self._update_topic_async(text, current_topic))
