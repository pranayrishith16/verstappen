# Verstappen - Legal Document Ingestion Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Azure Storage](https://img.shields.io/badge/Azure_Storage-Blob-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382C)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> High-performance document ingestion and indexing pipeline for legal AI. Extracts 500k+ legal documents with automated metadata extraction, semantic + lexical indexing, and streaming batch processing.

**Backend:** [hamilton](https://github.com/pranayrishith16/hamilton) | **Frontend:** [ronaldo](https://github.com/pranayrishith16/ronaldo)  
**Live at:** [https://www.veritlyai.com](https://www.veritlyai.com)

---

## 🎯 What This Does

Verstappen is the **data pipeline** that powers VERITLY AI's legal research engine. It processes massive volumes of legal documents and makes them searchable in real-time.

### What You Get

✅ **Batch Processing** - Handle 500+ PDFs/hour with intelligent batching  
✅ **Smart Metadata** - Auto-extract court, case name, docket number, dates via regex  
✅ **Dual Indexing** - BM25 (keyword search) + Qdrant (semantic search)  
✅ **Smart Chunking** - Recursive text splitting with 1500-char chunks + 150-char overlap  
✅ **Legal-BERT** - Purpose-built embeddings for legal document understanding  
✅ **Streaming Architecture** - Process files in batches without exhausting memory  
✅ **Ingestion Tracking** - Know exactly which files are processed (blob metadata flagging)  
✅ **Production Ready** - Health checks, detailed metrics, error recovery

## 🏗️ Architecture

### How It Works

```
                          INGESTION PIPELINE
                          (Orchestrator)
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
    ┌───▼───┐            ┌──────▼─────┐        ┌───────▼────┐
    │Parse  │            │   Clean    │        │ Annotate   │
    │(PDF)  │───────────▶│ (Whitespace)│───────▶│ (Metadata) │
    └───────┘            └────────────┘        └────────────┘
                                                       │
                                                  ┌────▼─────┐
                                                  │  Chunk   │
                                                  │ (1500ch) │
                                                  └────┬─────┘
                                                       │
                                                  ┌────▼─────┐
                                                  │  Embed   │
                                                  │ (768D)   │
                                                  └────┬─────┘
                                                       │
                              ┌────────────────────────┼──────────────────────┐
                              │                        │                      │
                         ┌────▼────┐            ┌──────▼────┐        ┌───────▼────┐
                         │ Qdrant  │            │   BM25    │        │ Mark as    │
                         │ (Dense) │            │ (Sparse)  │        │ Ingested   │
                         └─────────┘            └───────────┘        └────────────┘
```

### Information Flow

1. **Discover** - Scan Azure Blob for unprocessed PDFs
2. **Batch Load** - Stream files in 50-file batches (memory efficient)
3. **Extract** - PDF → raw text via pdfplumber
4. **Prepare** - Clean text, remove page numbers, normalize whitespace
5. **Extract Metadata** - Court name, case name, docket number via regex
6. **Split** - Recursive chunking: 1500 chars/chunk, 150 char overlap
7. **Vectorize** - Legal-BERT embeddings (768 dimensions)
8. **Index** - Simultaneously to:
   - **Qdrant**: Semantic search (dense vectors)
   - **BM25**: Keyword search (sparse vectors)
   - **Azure Storage**: Backup copies
9. **Track** - Mark blob as `ingested=true` for deduplication

### Processing Speed

Under production conditions:
- **150-200 files/hour** (PDF size dependent)
- **7.5-10k chunks/hour**
- **Memory**: 2-4GB per instance
- **CPU**: 2-3 cores (scales horizontally)

## 🚀 Getting Started

### 60-Second Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/pranayrishith16/verstappen.git && cd verstappen

# 2. Create environment
python3.11 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (copy template)
cp .env.example .env
# Edit .env with your Azure and Qdrant credentials

# 5. Start server
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# ✅ Ready! Visit http://localhost:8000/docs for API
```

### Prerequisites

- **Python 3.11+**
- **Azure Storage Account** (Blob Storage enabled)
- **Qdrant instance** (cloud or self-hosted)
- **Redis** (optional, for distributed locking)

### Environment Setup

Create `.env` file with these essentials:

```env
# ===== AZURE STORAGE =====
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=legal-docs

# ===== QDRANT VECTOR DB =====
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_TIMEOUT=300

# ===== REDIS (optional, for multi-instance) =====
REDIS_URL=redis://localhost:6379

# ===== OBSERVABILITY =====
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://localhost:5050
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "azure_storage_connected": true,
#   "ingestion_pipeline_ready": true,
#   "timestamp": "2025-03-22T10:30:00Z"
# }
```

## 💻 API Reference

### Core Endpoints

#### Discovery & Monitoring

```
GET  /health              → Service health check
GET  /discover            → List documents with ingestion status
GET  /discover/detailed   → Per-document details (paginated)
GET  /stats               → Ingestion statistics dashboard
GET  /metrics             → Detailed system metrics
```

#### Ingestion Control

```
POST /ingest/auto         → Full ingestion (blocking)
POST /ingest/background   → Full ingestion (async, fire-and-forget)
POST /ingest/batch        → Limited batch (up to N files)
GET  /ingest/status       → Current ingestion state
POST /reload-config       → Hot-reload configuration
```

### Real-World Examples

#### 1. Check Service Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "azure_storage_connected": true,
  "ingestion_pipeline_ready": true,
  "timestamp": "2025-03-22T10:30:00Z",
  "version": "1.0.0"
}
```

#### 2. Discover Documents

```bash
curl http://localhost:8000/discover
```

**Response:**
```json
{
  "total_files": 5240,
  "ingested_files": 5000,
  "not_ingested_files": 240,
  "total_size_mb": 1250.5,
  "files_by_type": {".pdf": 5240},
  "ingestion_rate": 95.42,
  "latest_file": "opinion_2025_03_22.pdf",
  "oldest_file": "opinion_2020_01_01.pdf"
}
```

#### 3. Start Full Ingestion (Blocking)

```bash
curl -X POST http://localhost:8000/ingest/auto
```

**Response:**
```json
{
  "status": "success",
  "files_processed": 240,
  "chunks_created": 12500,
  "processing_time_seconds": 1245,
  "started_at": "2025-03-22T09:00:00Z",
  "completed_at": "2025-03-22T09:20:45Z"
}
```

#### 4. Start Batch Ingestion (Async)

```bash
curl -X POST "http://localhost:8000/ingest/batch?max_files=1000"
```

**Response:**
```json
{
  "status": "started",
  "message": "Batch ingestion started in background",
  "timestamp": "2025-03-22T09:00:00Z"
}
```

Check status:
```bash
curl http://localhost:8000/ingest/status
```

#### 5. Get Detailed Statistics

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "ingestion_stats": {
    "total_ingestions_run": 15,
    "successful_ingestions": 14,
    "failed_ingestions": 1,
    "currently_ingesting": false,
    "last_ingestion_time": "2025-03-22T09:45:00Z"
  },
  "document_stats": {
    "total_documents": 5240,
    "ingested_documents": 5000,
    "not_ingested_documents": 240,
    "ingestion_rate_percent": 95.42,
    "total_storage_mb": 1250.5
  }
}
```

#### 6. Get Detailed Metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "service_metrics": {
    "total_ingestions": 15,
    "failed_ingestions": 1,
    "success_rate": 93.33,
    "currently_ingesting": false,
    "last_ingestion": "2025-03-22T09:45:00Z"
  },
  "document_metrics": {
    "total_documents": 5240,
    "ingested_documents": 5000,
    "average_file_size_mb": 0.238,
    "files_by_type": {".pdf": 5240}
  },
  "storage_location": {
    "container": "legal-docs",
    "latest_file": "opinion_2025_03_22.pdf",
    "oldest_file": "opinion_2020_01_01.pdf"
  }
}
```

### Integration with Hamilton (Backend)

Verstappen indexes documents → Hamilton retrieves for RAG:

```
User Query (Hamilton)
    ↓
Retrieval (calls Qdrant + BM25)
    ↓
[Uses indices built by Verstappen]
    ↓
Generation + Citations
```

**Flow:** User asks question in ronaldo → Hamilton queries → Retrieves from Verstappen's indices

## 🔧 How It Works (Deep Dive)

### The Ingestion Pipeline

Every document flows through this 6-stage processing:

**Stage 1: Parse**
- Tool: pdfplumber
- Input: Raw PDF bytes
- Output: Extracted text + page structure
- Time: 500ms - 2s per file (PDF-dependent)

**Stage 2: Clean**
- Removes page numbers, header/footer noise
- Normalizes whitespace
- Fixes hyphenation across line breaks
- Regex patterns: 7 cleanup rules

**Stage 3: Annotate**
- Extracts: court name, case name, docket number, case date, disposition
- Uses: 5+ specialized regex patterns
- Creates: metadata dictionary for each document section

**Stage 4: Chunk**
- Splits: 1500-char chunks with 150-char overlap
- Tool: LangChain RecursiveCharacterTextSplitter
- Preserves: metadata through chunking
- Handles: edge cases (short docs, tables)

**Stage 5: Embed**
- Model: nlpaueb/legal-bert-base-uncased
- Output: 768-dimensional vectors
- Batch size: 32 (memory efficient)
- Time: 2-3ms per chunk

**Stage 6: Index**
- BM25: Full-text inverted index (local storage)
- Qdrant: Vector index (cloud)
- Blob Metadata: ingested=true flag
- Atomicity: All-or-nothing per document

### Batch Streaming (Why It's Fast)

Instead of loading all 500 files into memory:

```
Traditional:  Load 500 files (2GB) → Process → Index
              ❌ Memory spike, slow start

Verstappen:   Load batch 1 (100MB) → Process → Index
              Load batch 2 (100MB) → Process → Index
              Load batch 3 (100MB) → Process → Index
              ... (50 batches of 50 files)
              ✅ Constant 100-200MB memory, faster overall
```

**Result:** Can process unlimited-size document sets without OOM

### Metadata Extraction Patterns

Legal documents have standard structures. Verstappen uses regex to extract:

| Field | Pattern | Example |
|-------|---------|---------|
| Court | `COURT OF\|DISTRICT COURT\|SUPREME COURT` | "COURT OF APPEALS" |
| Case Name | First `v\.` separated items | "Smith v. Jones" |
| Docket | `No\. \d+-\d+` | "No. 2024-12345" |
| Date | Month/Day/Year | "March 22, 2025" |
| Disposition | `AFFIRMED\|REVERSED\|REMANDED` | "AFFIRMED" |

### Deduplication Strategy

**Problem:** Re-running ingestion processes same files again  
**Solution:** Track via blob metadata

```python
# After successful ingestion:
blob_client.set_blob_metadata({
    "ingested": "true",
    "ingested_at": "2025-03-22T10:30:00Z",
    "chunks_created": 2500,
    "embedding_model": "legal-bert"
})
```

**Next run:** Skip blobs where metadata["ingested"] == "true"

This enables:
- ✅ **Idempotent ingestion** - Run multiple times safely
- ✅ **Incremental updates** - Process only new documents
- ✅ **Failure recovery** - Resume from last good state

## 🚀 Deployment

### Local Development

```bash
# Start service
uvicorn apps.api.main:app --reload

# Logs appear in console:
# INFO: Started server process
# INFO: Waiting for application startup.
# INFO: Application startup complete [PID 12345]
```

### Docker (Recommended for Production)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t verstappen:latest .

docker run -d \
  --name verstappen \
  -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  -e AZURE_STORAGE_CONNECTION_STRING=$CONN_STR \
  -e QDRANT_URL=$QDRANT_URL \
  -e QDRANT_API_KEY=$API_KEY \
  verstappen:latest
```

### Multi-Instance with Load Balancing

For 500k+ documents, use multiple instances with Redis coordination:

```yaml
# docker-compose.yml
version: "3.8"

services:
  verstappen-1:
    image: verstappen:latest
    environment:
      - REDIS_URL=redis://redis:6379
    ports:
      - "8001:8000"
    volumes:
      - ./logs:/app/logs

  verstappen-2:
    image: verstappen:latest
    environment:
      - REDIS_URL=redis://redis:6379
    ports:
      - "8002:8000"
    volumes:
      - ./logs:/app/logs

  verstappen-3:
    image: verstappen:latest
    environment:
      - REDIS_URL=redis://redis:6379
    ports:
      - "8003:8000"
    volumes:
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: verstappen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: verstappen
  template:
    metadata:
      labels:
        app: verstappen
    spec:
      containers:
      - name: api
        image: verstappen:latest
        ports:
        - containerPort: 8000
        env:
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: connection-string
        - name: QDRANT_API_KEY
          valueFrom:
            secretKeyRef:
              name: qdrant-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: verstappen-service
spec:
  selector:
    app: verstappen
  type: LoadBalancer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

### Azure App Service

```bash
# Deploy directly
az webapp up \
  --name verstappen-prod \
  --resource-group legal-ai \
  --runtime "PYTHON:3.11" \
  --logs
```

### Production Checklist

- [ ] Change all `.env` secrets (JWT_SECRET, API keys)
- [ ] Enable Redis for distributed locking
- [ ] Set up MLflow tracking
- [ ] Configure log aggregation (e.g., Application Insights)
- [ ] Set up monitoring alerts (health check timeouts, etc.)
- [ ] Create database backups
- [ ] Enable CORS only for hamilton backend domain
- [ ] Test ingestion with sample documents
- [ ] Document rollback procedure

## 📊 Monitoring & Observability

### Health Dashboard

```bash
# Check system health
curl http://localhost:8000/health

# Monitor ingestion progress
watch -n 5 'curl -s http://localhost:8000/metrics | jq .document_metrics'

# View detailed stats
curl http://localhost:8000/stats | jq
```

### Key Metrics to Monitor

**Ingestion Health:**
- Success rate: Target >95%
- Average file size trend
- Chunks per file (should be ~50)
- Processing time per file

**Resource Health:**
- Memory usage during batches
- CPU utilization
- Network throughput to Azure
- Qdrant response latency

**Data Quality:**
- Metadata extraction accuracy
- Chunk overlap distribution
- Embedding dimensions (should be 768)

### MLflow Integration (Optional)

Track experiments and versioning:

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 --port 5050

# View experiments at http://localhost:5050
```

### Logging

All logs go to stdout via loguru:

```
2025-03-22 10:15:30.123 | INFO | Starting ingestion pipeline
2025-03-22 10:15:31.456 | INFO | ✓ Connected to Azure Blob Storage
2025-03-22 10:16:00.789 | INFO | ✓ Parsed batch 1: 50 files, 2500 chunks
2025-03-22 10:16:30.123 | INFO | ✓ Ingested to Qdrant: 2500 vectors
2025-03-22 10:17:00.456 | ERROR | ✗ Failed to parse damaged_file.pdf: [error details]
```

Change verbosity:
```bash
export LOG_LEVEL=DEBUG  # More details
export LOG_LEVEL=WARN   # Less noise
```

## 🧪 Troubleshooting

### Issue: "Azure connectivity check failed"

**Symptoms:**
```
GET /health returns: "azure_storage_connected": false
```

**Debug steps:**
```bash
# 1. Verify credentials
echo $AZURE_STORAGE_CONNECTION_STRING

# 2. Test Azure CLI
az storage blob list --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --container-name legal-docs --num-results 5

# 3. Check network
curl -I https://youraccount.blob.core.windows.net/

# 4. Verify in .env file (no accidental quotes/spaces)
cat .env | grep AZURE_STORAGE
```

### Issue: Ingestion hangs or becomes very slow

**Symptoms:**
- No logs for 5+ minutes
- CPU at 100%
- Memory usage climbing

**Debug steps:**
```bash
# 1. Check which file is being processed
docker logs verstappen | tail -20

# 2. Verify PDF integrity
file problematic_document.pdf
pdfplumber.open("problematic_document.pdf")  # Python test

# 3. Increase timeout and restart
export QDRANT_TIMEOUT=600  # 10 minutes
docker restart verstappen
```

### Issue: "No chunks created from any batch"

**Symptoms:**
- Ingestion completes
- But chunks_created = 0

**Debug steps:**
```bash
# 1. Check file contents are extracting
python -c "
import pdfplumber
with pdfplumber.open('sample.pdf') as pdf:
    print(f'Pages: {len(pdf.pages)}')
    print(f'Text length: {len(pdf.pages[0].extract_text())}')
"

# 2. Verify chunking config
grep -A 5 "chunk_size" configs/pipelines/default.yaml

# 3. Check cleaner isn't removing all text
# (Very aggressive whitespace normalization?)

# 4. Verify embedder can load
python -c "from ingestion.embeddings.models.legal_bert import SentenceTransformerEmbedder; SentenceTransformerEmbedder().get_dimension()"
```

### Issue: Qdrant and BM25 have different results

**Symptoms:**
- Hamilton returns inconsistent search results
- Different rankings between dense and sparse

**Debug steps:**
```bash
# 1. Check both indices exist
curl http://localhost:8000/metrics | jq .document_metrics

# 2. Rebuild indices (safe - appends, doesn't delete)
curl -X POST http://localhost:8000/ingest/auto

# 3. Verify embedding model hasn't changed
grep "model_name" configs/components.yaml
```

### Issue: "Out of memory" during large batch

**Symptoms:**
```
MemoryError: Unable to allocate X.XX GiB
```

**Solutions:**
```bash
# 1. Reduce batch size
curl -X POST "http://localhost:8000/ingest/batch?max_files=100"  # Default 500

# 2. Increase instance memory
docker run --memory=8g verstappen:latest

# 3. Use multiple smaller instances instead of one large
```

## 🤝 Contributing

Found a bug or have an idea? We welcome contributions!

```bash
# 1. Create feature branch
git checkout -b feature/amazing-speedup

# 2. Make changes and test
pytest tests/

# 3. Commit and push
git commit -m "Add amazing speedup"
git push origin feature/amazing-speedup

# 4. Open PR to main
```

**Development setup:**
```bash
pip install -r requirements-dev.txt  # Includes pytest, black, flake8
black .                               # Format code
flake8 .                              # Check style
pytest -v                             # Run tests
```

## 📚 Learn More

- **[Hamilton Backend](https://github.com/pranayrishith16/hamilton)** - RAG orchestration and chat API
- **[Ronaldo Frontend](https://github.com/pranayrishith16/ronaldo)** - React + Redux web interface
- **[VERITLY AI](https://www.veritlyai.com)** - Live legal research platform

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern async Python web framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search engine
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF extraction
- [LangChain](https://www.langchain.com/) - Text splitting & chunking
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Azure SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/) - Cloud storage

## 📧 Questions?

- **GitHub Issues:** [pranayrishith16/verstappen/issues](https://github.com/pranayrishith16/verstappen/issues)
- **Email:** pranayrishith@example.com
- **Website:** [veritlyai.com](https://www.veritlyai.com)

---

**⭐ If this helped you, please leave a star!** It helps others discover the project.

Made with ❤️ for attorneys and legal technologists.
