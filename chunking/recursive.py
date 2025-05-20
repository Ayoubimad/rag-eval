from chunking.base import ChunkingStrategy
from typing import List, Optional, Callable
from chonkie.chunker import RecursiveChunker as ChonkieRecursiveChunker
from chonkie.tokenizer import WordTokenizer


class RecursiveChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.chunker = ChonkieRecursiveChunker(
            tokenizer_or_token_counter=WordTokenizer(),
            chunk_size=self.chunk_size,
            min_characters_per_chunk=64,
        )

    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        if clean_function:
            text = clean_function(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.chunker.chunk(text)

        return [chunk.text for chunk in chunks]
