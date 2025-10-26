import os
import pickle
from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from azure.storage.blob import BlobServiceClient
import logging

logger = logging.getLogger(__name__)

class BM25Retriever(Retriever):
    def __init__(
            self,
            index_path:Optional[str]=None,
            k1:float=1.2,
            b:float=0.75,
            use_azure=True,
            container_name: str = "bm25-indexes",
            blob_name: str = "bm25_index.pkl"):
        
        # Azure settings
        self.use_azure = use_azure
        self.container_name = container_name
        self.blob_name = blob_name
        self.blob_client = None

        
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

        if self.use_azure:
            self._init_azure_blob()
        
        # Load existing index
        self.load_index()

    def _init_azure_blob(self):
        """Initialize Azure Blob Storage client."""
        try:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                logger.warning("⚠️  AZURE_STORAGE_CONNECTION_STRING not set, using local storage")
                self.use_azure = False
                return
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Create container if it doesn't exist
            try:
                blob_service_client.create_container(name=self.container_name)
                logger.info(f"✓ Created/verified Azure container: {self.container_name}")
            except Exception as e:
                logger.debug(f"Container already exists or error: {e}")
            
            self.blob_client = blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.blob_name
            )
            logger.info(f"✓ Connected to Azure Blob Storage")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure Blob: {e}")
            self.use_azure = False

    def load_index(self) -> None:
        """Load BM25 index and chunks from disk."""
        try:
            if self.use_azure and self.blob_client:
                # Try to load from Azure
                try:
                    logger.info("Loading BM25 index from Azure Blob Storage...")
                    blob_data = self.blob_client.download_blob().readall()
                    data = pickle.loads(blob_data)
                    self.chunks = data["chunks"]
                    self.bm25 = data["bm25"]
                    logger.info(f"✓ Loaded BM25 index from Azure with {len(self.chunks)} chunks")
                    return
                except Exception as e:
                    logger.debug(f"Index not found in Azure: {e}")
            
            # Fallback: Try local disk
            if os.path.exists(f"{self.index_path}.pkl"):
                logger.info("Loading BM25 index from local disk...")
                with open(f"{self.index_path}.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.bm25 = data["bm25"]
                logger.info(f"✓ Loaded BM25 index from disk with {len(self.chunks)} chunks")
                
                # Optionally sync to Azure
                if self.use_azure:
                    logger.info("Syncing local index to Azure...")
                    self._save_to_azure()
                return
        
        except Exception as e:
            logger.error(f"❌ Failed to load BM25 index: {e}")
            self.bm25 = None
            self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        """Optimized tokenization using compiled regex."""
        # Convert to lowercase and extract words
        return [token.lower() for token in self.tokenizer_pattern.findall(text)]
    
    def _save_to_local(self) -> bool:
        """Save index to local disk (backup)."""
        try:
            logger.debug(f"Saving {len(self.chunks)} chunks to local disk...")
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(f"{self.index_path}.pkl", "wb") as f:
                pickle.dump(
                    {"chunks": self.chunks, "bm25": self.bm25}, 
                    f, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            logger.debug(f"✓ Saved BM25 index to local disk")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save to local disk: {e}")
            return False
        
    def _save_to_azure(self) -> bool:
        """Save index to Azure Blob Storage."""
        if not self.use_azure or not self.blob_client:
            return False
        
        try:
            logger.info(f"Saving {len(self.chunks)} chunks to Azure Blob Storage...")
            data = {"chunks": self.chunks, "bm25": self.bm25}
            blob_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.blob_client.upload_blob(blob_data, overwrite=True)
            size_kb = len(blob_data) / 1024
            logger.info(f"✓ Successfully saved BM25 index to Azure ({size_kb:.2f} KB)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save to Azure: {e}")
            return False

    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """
        Build or append to the BM25 index.
        
        - If index doesn't exist: Creates new BM25 index from chunks
        - If index exists: Automatically appends chunks to existing index
        """
        if not chunks:
            return
        
        # Check if index already exists
        if self.bm25 is not None and len(self.chunks) > 0:
            # Index exists - APPEND mode
            logger.info(f"✓ BM25 index exists with {len(self.chunks)} chunks. Appending {len(chunks)} new chunks...")
            return self.append(chunks, **kwargs)
        
        # Index doesn't exist - CREATE mode
        logger.info(f"Creating new BM25 index with {len(chunks)} chunks...")
        
        self.chunks = chunks
        corpus = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        
        # Save to both Azure and local
        self._save_to_azure()
        self._save_to_local()
        
        logger.info(f"✓ BM25 index created successfully with {len(chunks)} chunks")


    def append(self, chunks: List[Chunk], **kwargs) -> None:
        """Append chunks to existing BM25 index (does NOT delete existing chunks)."""
        if not chunks:
            logger.warning("No chunks provided for appending")
            return
        
        try:
            # Load if not already loaded
            if self.bm25 is None or len(self.chunks) == 0:
                self.load_index()
                if self.bm25 is None:
                    logger.info("No existing BM25 index found, creating new one")
                    return self.build_index(chunks, **kwargs)
            
            # Append new chunks
            logger.info(f"Appending {len(chunks)} new chunks to existing {len(self.chunks)} chunks...")
            old_count = len(self.chunks)
            self.chunks.extend(chunks)
            
            # Rebuild BM25 with all chunks
            corpus = [self._tokenize(c.content) for c in self.chunks]
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
            
            # Save to both Azure and local
            self._save_to_azure()
            self._save_to_local()
            
            logger.info(f"✓ Successfully appended {len(chunks)} chunks")
            logger.info(f"  Old count: {old_count} chunks")
            logger.info(f"  New count: {len(self.chunks)} chunks (total)")
        
        except Exception as e:
            logger.error(f"❌ Error appending to BM25 index: {e}")
            raise


    def clear_cache(self):
        """Clear the query tokenization cache."""
        self.query_cache.clear()
        logger.info("✓ Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 retriever statistics."""
        return {
            "chunks_count": len(self.chunks),
            "k1": self.k1,
            "b": self.b,
            "index_exists": self.bm25 is not None,
            "storage_location": "Azure Blob Storage" if self.use_azure else "Local Disk",
            "container_name": self.container_name if self.use_azure else None,
            "blob_name": self.blob_name if self.use_azure else None
        }