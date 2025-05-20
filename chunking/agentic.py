from typing import List, Optional, Callable

from chunking.base import ChunkingStrategy
from langchain_openai import ChatOpenAI


class AgenticChunker(ChunkingStrategy):
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text"""

    def __init__(self, model: ChatOpenAI, max_chunk_size: int = 1024):
        """
        Initialize the agentic chunking strategy

        Args:
            model: LLM model to use for determining breakpoints, must be a ChatOpenAI instance from langchain_openai
            max_chunk_size: Maximum chunk size in characters
        """
        self.max_chunk_size = max_chunk_size
        self.model = model

    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        """
        Split text into chunks using LLM to determine natural breakpoints based on context

        Args:
            text: Text to split into chunks
            clean_function: Optional function to clean the text before chunking

        Returns:
            List of text chunks
        """
        if clean_function:
            text = clean_function(text)

        if len(text) <= self.max_chunk_size:
            return [text]

        chunks: List[str] = []
        remaining_text = text

        while remaining_text:
            breakpoint_index = self._get_breakpoint(remaining_text)

            chunk_text = remaining_text[:breakpoint_index].strip()
            if chunk_text:
                chunks.append(chunk_text)

            remaining_text = remaining_text[breakpoint_index:].strip()

        return chunks

    def _get_breakpoint(self, text: str) -> int:
        """
        Get a natural breakpoint in text using the LLM

        Args:
            text: Text to analyze

        Returns:
            Character index where text should be split
        """
        prompt = f"""You are an expert in natural language understanding. Your task is to find the most natural breakpoint in the **given text below**, counting characters **starting from the beginning of this text only** (i.e., character 0 is the first character shown).

        A good breakpoint is one that:
        - Ends a sentence, paragraph, or thought
        - Occurs at a natural pause or topic transition
        - Maximizes semantic completeness without exceeding the text

        Return **only the character index (as an integer)** where the text should be split. Do **not** include any explanation, notes, or formatting.

        Here are some example outputs:
        100  
        219  
        320  
        450  
        512  
        634  
        789

        Now analyze the following text and return the best breakpoint index:\n\n{text[:self.max_chunk_size]}"""

        try:
            break_point = int(self.model.generate(prompt).strip())
            return min(break_point, self.max_chunk_size, len(text))
        except Exception:
            return min(self.max_chunk_size, len(text))
