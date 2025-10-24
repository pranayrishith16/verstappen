import os
import pickle
from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk

class BM25Retriever(Retriever):
    def __init__(self,index_path:Optional[str]=None,k1:float=1.2,b:float=0.75):
        self.index_path = index_path or "storage/bm25_index"
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []

        # Compile regex for better tokenization performance
        self.tokenizer_pattern = re.compile(r'\b\w+\b')

        # Cache for tokenized queries to avoid re-tokenizing identical queries
        self.query_cache: Dict[str, List[str]] = {}
        self.cache_size = 1000  # Limit cache size

        # Load existing index if present
        if os.path.exists(f"{self.index_path}.pkl"):
            self.load_index()

    def load_index(self) -> None:
        """Load BM25 index and chunks from disk."""
        try:
            with open(f"{self.index_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25 = data["bm25"]
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            self.bm25 = None
            self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        """Optimized tokenization using compiled regex."""
        # Convert to lowercase and extract words
        return [token.lower() for token in self.tokenizer_pattern.findall(text)]

    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """Build BM25 index from a list of Chunk objects."""
        if not chunks:
            return

        self.chunks = chunks
        # Tokenize each chunk content using optimized tokenizer
        corpus = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

        # Persist index using highest protocol for better performance
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "bm25": self.bm25}, 
                f, 
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def clear_cache(self):
        """Clear the query tokenization cache."""
        self.query_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "indexed_chunks": len(self.chunks),
            "k1": self.k1,
            "b": self.b
        }