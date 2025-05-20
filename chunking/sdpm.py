from chunking.base import ChunkingStrategy
from chunking.embeddings import ChonkieEmbeddings
from typing import List, Optional, Callable, Any
from chonkie.chunker import SDPMChunker as ChonkieSDPMChunker


class SDPMChunker(ChunkingStrategy):
    """
    Chunking strategy that uses Semantic Double Pass Merging to determine
    natural breakpoints in the text
    """

    def __init__(
        self,
        embedding_model: ChonkieEmbeddings,
        chunk_size: int = 256,
        threshold: float = 0.5,
    ):
        """
        Initialize the SDPM chunking strategy

        Args:
            embedding_model: Embeddings model for semantic comparison
            chunk_size: Target chunk size in words
            min_chunk_size: Minimum chunk size in words
            threshold: Similarity threshold for chunk boundaries
        """
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunker = ChonkieSDPMChunker(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            threshold=self.threshold,
            min_chunk_size=8,
        )

    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        """
        Split text into chunks using SDPM to determine natural breakpoints

        Args:
            text: Text to chunk
            clean_function: A function that cleans the text before chunking

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        if clean_function:
            text = clean_function(text)

        sdpm_chunks = self.chunker.chunk(text)

        return [chunk.text for chunk in sdpm_chunks]
