from abc import ABC, abstractmethod
from typing import List, Optional, Callable


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk(
        self, text: str, clean_function: Optional[Callable[[str], str]] = None
    ) -> List[str]:
        """Split a document into chunks according to the strategy.

        Args:
            text: The text to split into chunks as a string
            clean_function: A function that cleans the text before chunking

        Returns:
            A list of strings representing the chunks
        """
        raise NotImplementedError("Subclasses must implement this method")
