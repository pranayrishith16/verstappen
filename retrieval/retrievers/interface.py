from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Retriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str,Any]]) -> None:
        """Build or update the retrieval index."""
        pass
