"""
Ingestion Service API - Dedicated FastAPI for document ingestion and monitoring.
Handles Azure Blob Storage document discovery, ingestion tracking, and comprehensive statistics.
"""

import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient
from pathlib import Path
from datetime import datetime
from loguru import logger
import dotenv

# Import your actual ingestion pipeline
from ingestion.pipelines.ingestion_pipeline import IngestionPipeline

dotenv.load_dotenv()

# Initialize FastAPI app for Ingestion Service
app = FastAPI(
    title="Ingestion Service API",
    description="Dedicated API for document ingestion, discovery, and monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Configuration ====================
AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "legal-docs")

# ==================== Global State ====================
ingestion_pipeline = IngestionPipeline()
ingestion_in_progress = False
last_ingestion_time = None
total_ingestions = 0
ingestion_errors = 0
last_error_message = None

# Track ingested files (in production, use database or persistent storage)
ingested_files = set()  # Set of blob names that have been ingested

# ==================== Pydantic Models ====================

class DocumentDiscoveryResponse(BaseModel):
    """Response containing document discovery stats with ingestion tracking"""
    total_files: int
    ingested_files: int
    not_ingested_files: int
    total_size_bytes: int
    total_size_mb: float
    files_by_type: Dict[str, int]
    directory_scanned: str
    latest_file: Optional[str] = None
    oldest_file: Optional[str] = None
    ingestion_rate: float  # Percentage of files ingested

class IngestionResult(BaseModel):
    """Result of an ingestion operation"""
    status: str
    files_processed: int
    chunks_created: int
    processing_time_seconds: float
    started_at: str
    completed_at: str

# ==================== Helper Functions ====================

def check_if_blob_ingested(blob_name: str) -> bool:
    """
    Check if a blob has been ingested by looking at its metadata.
    In production, you should store this in a database or check blob metadata.
    """
    try:
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob_client = client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)
        properties = blob_client.get_blob_properties()
        metadata = properties.metadata or {}
        # Check if metadata has 'ingested' flag set to 'true'
        return metadata.get("ingested", "false").lower() == "true"
    except Exception as e:
        logger.error(f"Error checking ingestion status for {blob_name}: {e}")
        return False

def discover_azure_documents_with_ingestion_status() -> Dict[str, Any]:
    """Discover all documents in Azure Blob Storage with size and ingestion status"""
    try:
        if not AZURE_CONN_STR or not AZURE_CONTAINER:
            raise ValueError("Azure Storage credentials not configured")

        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        container_client = client.get_container_client(AZURE_CONTAINER)

        files = []
        files_by_type = {}
        file_times = []
        total_size_bytes = 0
        ingested_count = 0
        not_ingested_count = 0

        for blob in container_client.list_blobs():
            # Check ingestion status
            is_ingested = check_if_blob_ingested(blob.name)
            
            if is_ingested:
                ingested_count += 1
            else:
                not_ingested_count += 1

            files.append({
                "name": blob.name,
                "size_bytes": blob.size if blob.size else 0,
                "size_mb": (blob.size / (1024 * 1024)) if blob.size else 0,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "ingested": is_ingested
            })
            
            ext = os.path.splitext(blob.name)[1].lower()
            if ext:
                files_by_type[ext] = files_by_type.get(ext, 0) + 1
            
            total_size_bytes += blob.size if blob.size else 0
            
            if blob.last_modified:
                file_times.append((blob.name, blob.last_modified.timestamp()))

        oldest_file = None
        latest_file = None

        if file_times:
            file_times.sort(key=lambda x: x[1])
            oldest_file = file_times[0][0]
            latest_file = file_times[-1][0]

        total_files = len(files)
        ingestion_rate = (ingested_count / total_files * 100) if total_files > 0 else 0

        return {
            "files": files,
            "total_files": total_files,
            "ingested_files": ingested_count,
            "not_ingested_files": not_ingested_count,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "files_by_type": files_by_type,
            "latest_file": latest_file,
            "oldest_file": oldest_file,
            "directory_scanned": AZURE_CONTAINER,
            "ingestion_rate": round(ingestion_rate, 2),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error discovering Azure documents: {e}")
        raise e

def check_azure_connectivity() -> bool:
    """Check if Azure Storage is accessible"""
    try:
        if not AZURE_CONN_STR:
            return False
        client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        client.get_account_information()
        return True
    except Exception as e:
        logger.error(f"Azure connectivity check failed: {e}")
        return False

async def run_ingestion_pipeline() -> Dict[str, Any]:
    """Execute the ingestion pipeline and track results"""
    global ingestion_in_progress, last_ingestion_time, total_ingestions, ingestion_errors, last_error_message
    
    try:
        ingestion_in_progress = True
        start_time = datetime.now()
        
        logger.info("Starting ingestion pipeline...")
        
        # Call your actual ingestion pipeline
        result = ingestion_pipeline.ingest_from_azure()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        last_ingestion_time = end_time
        
        files_processed = result.get("files_processed", 0)
        total_chunks = result.get("total_chunks", 0)
        
        logger.info(f"Ingestion completed: {files_processed} files, {total_chunks} chunks")
        
        return {
            "status": "success",
            "files_processed": files_processed,
            "chunks_created": total_chunks,
            "processing_time_seconds": processing_time,
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "message": f"Successfully ingested {files_processed} files into {total_chunks} chunks"
        }
        
    except Exception as e:
        ingestion_errors += 1
        last_error_message = str(e)
        logger.error(f"Ingestion pipeline error: {e}")
        raise e
    finally:
        ingestion_in_progress = False

async def run_limited_batch_ingestion(max_files: int = 1000) -> Dict[str, Any]:
    """
    Execute ingestion pipeline for a limited batch of non-ingested files.
    """
    global ingestion_in_progress, last_ingestion_time, total_ingestions, ingestion_errors, last_error_message
    
    try:
        ingestion_in_progress = True
        start_time = datetime.now()
        
        logger.info(f"Starting limited batch ingestion (max {max_files} files)...")
        
        # Call ingestion pipeline with max_files limit
        result = ingestion_pipeline.ingest_non_ingested_batch(max_files=max_files)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        last_ingestion_time = end_time
        
        files_processed = result.get("files_processed", 0)
        total_chunks = result.get("total_chunks", 0)
        
        logger.info(f"Limited batch ingestion completed: {files_processed} files, {total_chunks} chunks")
        
        return {
            "status": "success",
            "files_processed": files_processed,
            "chunks_created": total_chunks,
            "processing_time_seconds": processing_time,
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "message": f"Successfully ingested {files_processed} files (limit: {max_files})"
        }
        
    except Exception as e:
        ingestion_errors += 1
        last_error_message = str(e)
        logger.error(f"Limited batch ingestion error: {e}")
        raise e
    finally:
        ingestion_in_progress = False

# ==================== Root Endpoint ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Ingestion Service API",
        "version": "1.0.0",
        "description": "Dedicated API for document ingestion and monitoring",
        "endpoints": [
            "GET /health - Health check",
            "GET /stats - Ingestion statistics",
            "GET /discover - Discover documents with ingestion status",
            "POST /ingest/auto - Trigger auto-ingestion",
            "GET /ingest/status - Get ingestion status",
            "GET /metrics - Detailed metrics"
        ]
    }

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring service availability"""
    try:
        azure_connected = check_azure_connectivity()
        pipeline_ready = ingestion_pipeline is not None
        
        return {
            "status": "healthy" if (azure_connected and pipeline_ready) else "degraded",
            "azure_storage_connected": azure_connected,
            "ingestion_pipeline_ready": pipeline_ready,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "azure_storage_connected": False,
            "ingestion_pipeline_ready": False,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "error": str(e)
        }

# ==================== Discovery Endpoints ====================

@app.get("/discover", response_model=DocumentDiscoveryResponse)
async def discover_documents():
    """Discover all documents in Azure Blob Storage with ingestion status tracking"""
    try:
        result = discover_azure_documents_with_ingestion_status()
        
        return DocumentDiscoveryResponse(
            total_files=result["total_files"],
            ingested_files=result["ingested_files"],
            not_ingested_files=result["not_ingested_files"],
            total_size_bytes=result["total_size_bytes"],
            total_size_mb=round(result["total_size_mb"], 2),
            files_by_type=result["files_by_type"],
            directory_scanned=result["directory_scanned"],
            latest_file=result["latest_file"],
            oldest_file=result["oldest_file"],
            ingestion_rate=result["ingestion_rate"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/discover/detailed")
async def discover_documents_detailed():
    """Get detailed information about each document including ingestion status"""
    try:
        result = discover_azure_documents_with_ingestion_status()
        
        ingested_docs = [f for f in result["files"] if f["ingested"]]
        not_ingested_docs = [f for f in result["files"] if not f["ingested"]]
        
        return {
            "summary": {
                "total_files": result["total_files"],
                "ingested_files": result["ingested_files"],
                "not_ingested_files": result["not_ingested_files"],
                "total_size_mb": round(result["total_size_mb"], 2),
                "ingestion_rate": result["ingestion_rate"],
                "directory": result["directory_scanned"]
            },
            "ingested_documents": ingested_docs,
            "not_ingested_documents": not_ingested_docs,
            "files_by_type": result["files_by_type"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Ingestion Endpoints ====================

@app.post("/ingest/auto", response_model=IngestionResult)
async def auto_ingest(background_tasks: BackgroundTasks):
    """Trigger automatic ingestion of all non-ingested documents"""
    global ingestion_in_progress, total_ingestions

    if ingestion_in_progress:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    try:
        total_ingestions += 1
        
        # Run ingestion pipeline
        result = await run_ingestion_pipeline()
        
        return IngestionResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/background")
async def auto_ingest_background(background_tasks: BackgroundTasks):
    """Trigger ingestion as a background task (non-blocking)"""
    global ingestion_in_progress, total_ingestions

    if ingestion_in_progress:
        return {"status": "warning", "message": "Ingestion already in progress"}

    total_ingestions += 1
    background_tasks.add_task(run_ingestion_pipeline)
    
    return {
        "status": "started",
        "message": "Auto-ingestion started in background",
        "started_at": datetime.now().isoformat()
    }

@app.get("/ingest/status")
async def get_ingest_status():
    """Get current ingestion status"""
    return {
        "in_progress": ingestion_in_progress,
        "last_ingestion": last_ingestion_time.isoformat() if last_ingestion_time else None,
        "total_ingestions": total_ingestions,
        "failed_ingestions": ingestion_errors,
        "last_error": last_error_message
    }

@app.post("/ingest/batch")
async def ingest_batch(max_files: int = 1000):
    """
    Trigger ingestion of a limited batch of non-ingested files.
    This endpoint should be called by a scheduled Azure Container App Job.
    
    Args:
        max_files: Maximum number of files to ingest (default 1000)
    """
    global ingestion_in_progress, total_ingestions

    if ingestion_in_progress:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    try:
        total_ingestions += 1
        
        # Run limited batch ingestion
        result = await run_limited_batch_ingestion(max_files)
        
        return IngestionResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Statistics & Metrics ====================

@app.get("/stats")
async def get_statistics():
    """Get comprehensive ingestion statistics"""
    try:
        discovery = discover_azure_documents_with_ingestion_status()
        
        return {
            "ingestion_stats": {
                "total_ingestions_run": total_ingestions,
                "successful_ingestions": total_ingestions - ingestion_errors,
                "failed_ingestions": ingestion_errors,
                "currently_ingesting": ingestion_in_progress,
                "last_ingestion_time": last_ingestion_time.isoformat() if last_ingestion_time else None,
                "last_error": last_error_message
            },
            "document_stats": {
                "total_documents": discovery["total_files"],
                "ingested_documents": discovery["ingested_files"],
                "not_ingested_documents": discovery["not_ingested_files"],
                "ingestion_rate_percent": discovery["ingestion_rate"],
                "total_storage_mb": round(discovery["total_size_mb"], 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics about ingestion service"""
    try:
        discovery = discover_azure_documents_with_ingestion_status()
        
        return {
            "service_metrics": {
                "total_ingestions": total_ingestions,
                "failed_ingestions": ingestion_errors,
                "success_rate": ((total_ingestions - ingestion_errors) / total_ingestions * 100) if total_ingestions > 0 else 0,
                "currently_ingesting": ingestion_in_progress,
                "last_ingestion": last_ingestion_time.isoformat() if last_ingestion_time else None
            },
            "document_metrics": {
                "total_documents": discovery["total_files"],
                "ingested_documents": discovery["ingested_files"],
                "not_ingested_documents": discovery["not_ingested_files"],
                "ingestion_completion_rate": discovery["ingestion_rate"],
                "total_storage_mb": round(discovery["total_size_mb"], 2),
                "files_by_type": discovery["files_by_type"],
                "average_file_size_mb": round(discovery["total_size_mb"] / discovery["total_files"], 2) if discovery["total_files"] > 0 else 0
            },
            "storage_location": {
                "container": AZURE_CONTAINER,
                "latest_file": discovery["latest_file"],
                "oldest_file": discovery["oldest_file"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Configuration ====================

@app.post("/reload-config")
async def reload_config():
    """Reload service configuration from environment variables"""
    try:
        global AZURE_CONN_STR, AZURE_CONTAINER
        AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        AZURE_CONTAINER = os.getenv("AZURE_CONTAINER_NAME", "legal-docs")
        
        logger.info("Configuration reloaded successfully")
        return {
            "status": "success",
            "message": "Configuration reloaded",
            "azure_container": AZURE_CONTAINER
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Ingestion Service API starting up...")
    logger.info(f"Azure Container configured: {AZURE_CONTAINER}")
    logger.info(f"Azure Storage connected: {check_azure_connectivity()}")
    logger.info(f"Ingestion pipeline initialized: {ingestion_pipeline is not None}")
