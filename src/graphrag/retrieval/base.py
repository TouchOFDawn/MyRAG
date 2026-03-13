from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class RetrieverResult(BaseModel):
    """Normalized retrieval response from any backend."""
    content: str
    metadata: Dict[str, Any] = {}
    score: float = 0.0

class BaseRetriever(ABC):
    """Abstract interface for all retrievers (Vector, Graph, etc.)."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[RetrieverResult]:
        """
        Executes a search against the backing datastore.
        Returns a list of standardized RetrieverResult objects.
        """
        pass
