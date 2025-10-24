# Use Python 3.11 slim base image
FROM python:3.11-slim

# Disable interactive prompts and enable unbuffered output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install OS-level build dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      curl \
      git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies (ensure requirements.txt has no mlflow line)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=true

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]