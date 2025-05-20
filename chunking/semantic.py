from chunking.embeddings import ChonkieEmbeddings
from chunking.base import ChunkingStrategy
from chonkie.chunker import SemanticChunker as ChonkieSemanticChunker
from typing import List, Optional, Callable


class SemanticChunker(ChunkingStrategy):
    """Chunking strategy that uses semantic search to determine natural breakpoints in the text"""

    def __init__(
        self,
        embedding_model: ChonkieEmbeddings,
        chunk_size: int = 256,
        threshold: float = 0.5,
    ):
        """
        Initialize the semantic chunking strategy

        Args:
            embedding_model: Embeddings model for semantic comparison
            chunk_size: Target chunk size in words
            threshold: Similarity threshold for chunk boundaries
        """
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunker = ChonkieSemanticChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            min_chunk_size=8,
            threshold=self.threshold,
        )

    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        """Split a text into chunks according to the semantic chunking strategy.

        Args:
            text: The text to split into chunks as a string

        Returns:
            A list of strings representing the chunks
        """
        if clean_function:
            text = clean_function(text)

        if len(text) <= self.chunk_size:
            return [text]

        semantic_chunks = self.chunker.chunk(text)

        return [chunk.text for chunk in semantic_chunks]
