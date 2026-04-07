# VectorStore ABC and SearchResult model

from abc import ABC, abstractmethod
from typing import List, Optional

class SearchResult:
    """Represents a single result from a vector search."""
    def __init__(self, tool_name: str, score: float, metadata: dict = None):
        self.tool_name = tool_name
        self.score = score

    def __repr__(self):
        return f"SearchResult(tool_name='{self.tool_name}', score={self.score:.4f})"
        
    def __str__(self):
        return self.__repr__()

class VectorStore(ABC):
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], tool_names: List[str], metadata: List[dict] = None) -> None:
        """Adds vectors to the store."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int = 5) -> List[SearchResult]:
        """Performs a similarity search."""
        pass
