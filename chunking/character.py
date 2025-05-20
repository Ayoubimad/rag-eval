from chunking.base import ChunkingStrategy
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Optional, Callable


class CharacterChunker(ChunkingStrategy):
    """Chunking strategy that splits text into chunks based on character count."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 0):
        """Initialize the character chunking strategy.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separator="",
            is_separator_regex=False,
        )

    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        """Split text into chunks based on character count.

        Args:
            text: Text to split into chunks
            clean_function: A function that cleans the text before chunking

        Returns:
            List of text chunks
        """
        if clean_function:
            text = clean_function(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.chunker.split_text(text)

        return [chunk for chunk in chunks]
