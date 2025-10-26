# retrieval/retrievers/qdrant_retriever.py
import os
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class QdrantDenseRetriever(Retriever):
    """Qdrant-based dense vector retriever using REST API."""
    
    def __init__(
            self, 
            collection_name='legal-docs',
            metric: str = "cosine",
            qdrant_url: Optional[str] = None,
            qdrant_api_key: Optional[str] = None,
            timeout: int = 300):
        """
        Initialize Qdrant Dense Retriever with REST API.

        Args:
            collection_name: Name of Qdrant collection
            metric: Distance metric (cosine, euclid, dot)
            qdrant_url: Qdrant service URL (HTTPS)
            qdrant_api_key: API key for Qdrant (optional)
            timeout: Request timeout in seconds (default 300s)
        """
        self.collection_name = collection_name
        self.metric = metric
        self.dimension: Optional[int] = None
        self.logger = logger
        self.timeout = timeout

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "https://qdrant-app.southcentralus.azurecontainerapps.io")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        env_timeout = os.getenv("QDRANT_TIMEOUT")
        if env_timeout:
            try:
                self.timeout = int(env_timeout)
            except ValueError:
                pass
        
        self.logger.info(f"Initializing Qdrant at {self.qdrant_url} with {self.timeout}s timeout")

        self.session = requests.Session()
        self.headers = {}
        if self.qdrant_api_key:
            self.headers['api-key'] = self.qdrant_api_key

        try:
            response = self.session.get(
                f"{self.qdrant_url}/collections",
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                self.logger.info(f"✓ Successfully connected to Qdrant")
            else:
                raise Exception(f"Qdrant returned status {response.status_code}")
        except Exception as e:
            self.logger.error(f"✗ Failed to connect to Qdrant: {e}")
            raise

        self.embedder = None

    def _get_embedder(self):
        """Get cached embedder or fetch from registry."""
        if self.embedder is None:
            self.embedder = registry.get("embedder")
        return self.embedder
    
    def _map_metric_to_distance(self, metric: str) -> str:
        """Map metric name to Qdrant distance metric."""
        metric_map = {
            "cosine": "Cosine",
            "euclid": "Euclid",
            "dot": "Dot"
        }
        return metric_map.get(metric.lower(), "Cosine")
    
    def _collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            response = self.session.get(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            return response.status_code == 200
        except:
            return False
    
    def _create_collection(self, dimension: int) -> bool:
        """Create a new collection."""
        try:
            self.logger.info(f"Creating collection with {dimension}D vectors...")
            create_payload = {
                "vectors": {
                    "size": dimension,
                    "distance": self._map_metric_to_distance(self.metric)
                }
            }
            response = self.session.put(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                json=create_payload,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to create collection: {response.status_code} - {response.text}")
            
            self.logger.info(f"✓ Created collection: {self.collection_name}")
            self.dimension = dimension
            return True
        except Exception as e:
            self.logger.error(f"✗ Failed to create collection: {e}")
            raise
    
    def _upsert_batch(self, points: List[Dict]) -> bool:
        """Upsert a batch of points."""
        try:
            upsert_payload = {"points": points}
            response = self.session.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                json=upsert_payload,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to upsert: {response.status_code} - {response.text}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to upsert batch: {e}")
            raise

    # ============= Main Methods =============

    def build_index(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> None:
        """
        Build or append to the index.
        
        - If collection doesn't exist: Creates new collection and adds chunks
        - If collection exists: Automatically appends chunks (does NOT delete)
        
        This is now the single entry point - no need to call append() separately.
        """
        if not chunks:
            self.logger.warning("No chunks provided for indexing")
            return
        
        self.logger.info(f"Building Qdrant index with {len(chunks)} chunks")

        if embeddings is None:
            self.logger.info("Generating embeddings...")
            embedder = self._get_embedder()
            embeddings = embedder.encode(chunks)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.dimension = embeddings.shape[1]

        # Check if collection already exists
        if self._collection_exists():
            # Collection exists - APPEND mode (no delete)
            self.logger.info(f"✓ Collection '{self.collection_name}' already exists")
            return self.append(chunks, embeddings)

        # Collection doesn't exist - CREATE mode
        self.logger.info(f"Collection doesn't exist, creating new collection")
        
        # Create collection
        self._create_collection(self.dimension)

        # Prepare and upsert points
        self.logger.info("Preparing points for upsert...")
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = {
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {
                    "chunk_id": chunk.id if hasattr(chunk, 'id') else str(idx),
                    "content": chunk.content if isinstance(chunk, Chunk) else str(chunk),
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
            }
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(points) - 1) // batch_size + 1
            
            self.logger.info(f"Upserting batch {batch_num}/{total_batches}...")
            self._upsert_batch(batch)
            self.logger.info(f"✓ Upserted batch {batch_num}/{total_batches} ({len(batch)} points)")

        self.logger.info(f"✓ Qdrant index built successfully with {len(points)} points")

    def append(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> None:
        """Append new chunks to existing collection (no delete, just add/update)."""
        if not chunks:
            self.logger.warning("No chunks provided for appending")
            return
        
        self.logger.info(f"Appending {len(chunks)} chunks to collection")

        # Check if collection exists
        if not self._collection_exists():
            self.logger.warning("Collection doesn't exist, creating new one")
            self.build_index(chunks, embeddings)
            return

        if embeddings is None:
            self.logger.info("Generating embeddings...")
            embedder = self._get_embedder()
            embeddings = embedder.encode(chunks)

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Get next available ID
        try:
            response = self.session.get(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            current_count = response.json().get("result", {}).get("points_count", 0)
            start_id = current_count
        except:
            start_id = 0

        # Prepare points with new IDs
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = {
                "id": start_id + idx,
                "vector": embedding.tolist(),
                "payload": {
                    "chunk_id": chunk.id if hasattr(chunk, 'id') else str(start_id + idx),
                    "content": chunk.content if isinstance(chunk, Chunk) else str(chunk),
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
            }
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(points) - 1) // batch_size + 1
            
            self.logger.info(f"Appending batch {batch_num}/{total_batches}...")
            self._upsert_batch(batch)
            self.logger.info(f"✓ Appended batch {batch_num}/{total_batches} ({len(batch)} points)")

        self.logger.info(f"✓ Successfully appended {len(points)} points to collection")

    def update(self, point_id: int, chunk: Chunk, embedding: Optional[np.ndarray] = None) -> bool:
        """Update a single point in the collection."""
        try:
            self.logger.info(f"Updating point {point_id}")
            
            if embedding is None:
                embedder = self._get_embedder()
                embedding = embedder.encode([chunk.content])[0]
            
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            
            point = {
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": {
                    "chunk_id": chunk.id if hasattr(chunk, 'id') else str(point_id),
                    "content": chunk.content if isinstance(chunk, Chunk) else str(chunk),
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
            }
            
            return self._upsert_batch([point])
        except Exception as e:
            self.logger.error(f"Failed to update point {point_id}: {e}")
            return False

    def delete(self, point_ids: List[int]) -> bool:
        """Delete points from collection by IDs."""
        try:
            self.logger.info(f"Deleting {len(point_ids)} points")
            
            delete_payload = {"points": point_ids}
            response = self.session.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/delete",
                json=delete_payload,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code not in [200, 201]:
                raise Exception(f"Failed to delete: {response.status_code} - {response.text}")
            
            self.logger.info(f"✓ Deleted {len(point_ids)} points")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete points: {e}")
            return False

    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.logger.warning(f"Deleting entire collection: {self.collection_name}")
            
            response = self.session.delete(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                self.logger.info(f"✓ Deleted collection: {self.collection_name}")
                return True
            else:
                raise Exception(f"Failed to delete collection: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k similar chunks for a query."""
        try:
            if not query or query.strip() == "":
                self.logger.warning("Empty query provided")
                return []
            
            self.logger.debug(f"Retrieving top-{k} chunks for query: {query[:100]}...")
            
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query])[0].astype(np.float32).tolist()
            
            search_payload = {
                "vector": query_embedding,
                "limit": k
            }
            
            response = self.session.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json=search_payload,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code != 200:
                self.logger.error(f"Search failed: {response.status_code} - {response.text}")
                return []
            
            search_results = response.json()
            results = []
            
            for result in search_results.get("result", []):
                r = {
                    "id": result["id"],
                    "content": result["payload"].get("content", ""),
                    "chunk_id": result["payload"].get("chunk_id", ""),
                    "metadata": result["payload"].get("metadata", {}),
                    "score": result["score"]
                }
                results.append(r)
            
            self.logger.debug(f"Retrieved {len(results)} chunks")
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving: {e}")
            return []

    def save_index(self) -> None:
        """Qdrant persists automatically - no action needed."""
        self.logger.info("Qdrant persists data automatically to disk")
    
    def load_index(self) -> None:
        """Qdrant loads automatically - no action needed."""
        self.logger.info(f"Qdrant collection '{self.collection_name}' loaded from server")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            response = self.session.get(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                info = response.json()
                stats = info.get("result", {})
                return {
                    "chunks_count": stats.get("points_count", 0),
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "collection_name": self.collection_name,
                    "qdrant_url": self.qdrant_url,
                    "status": "ready"
                }
            else:
                return {"status": "error", "error": f"Status {response.status_code}"}
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
