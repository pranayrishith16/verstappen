# reset_ingested_to_false.py
"""
One-off script to reset ingested files from true to false.
Limited to first 2000 files.
"""

import os
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import logging
import dotenv

dotenv.load_dotenv()

# ===== SUPPRESS AZURE SDK VERBOSE LOGGING =====
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get configuration from environment
CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
FILE_EXTENSIONS = ['.pdf']
MAX_FILES = 2000  # ← LIMIT TO 2000

print("🔄 Resetting Ingested Files to False (Limited to 2000)")
print("=" * 70)

if not CONN_STR or not CONTAINER_NAME:
    print("❌ Error: Set AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME")
    exit(1)

try:
    # Connect to Azure
    client = BlobServiceClient.from_connection_string(CONN_STR)
    container_client = client.get_container_client(CONTAINER_NAME)
    
    print(f"✅ Connected to: {CONTAINER_NAME}")
    print(f"📊 Processing limit: {MAX_FILES} files\n")
    
    # Step 1: Find all ingested files (up to MAX_FILES)
    print(f"📋 Scanning for ingested files (max {MAX_FILES})...")
    ingested_files = []
    scanned_count = 0
    
    for blob in container_client.list_blobs():
        # Stop if we've scanned MAX_FILES
        if scanned_count >= MAX_FILES:
            break
        
        scanned_count += 1
        
        # Check file extension
        if not any(blob.name.endswith(ext) for ext in FILE_EXTENSIONS):
            continue
        
        blob_client = container_client.get_blob_client(blob.name)
        try:
            properties = blob_client.get_blob_properties()
            metadata = properties.metadata or {}
            
            # Check if ingested
            if metadata.get("ingested", "false").lower() == "true":
                ingested_files.append(blob.name)
        
        except Exception as e:
            logger.warning(f"Could not check {blob.name}: {e}")
    
    print(f"✅ Scanned {scanned_count} files")
    print(f"✅ Found {len(ingested_files)} ingested files\n")
    
    if len(ingested_files) == 0:
        print("✅ No ingested files found. Done!")
        exit(0)
    
    # Show files
    print("📝 Files to reset:")
    for i, file in enumerate(ingested_files[:20], 1):
        print(f"   {i}. {file}")
    if len(ingested_files) > 20:
        print(f"   ... and {len(ingested_files) - 20} more")
    print()
    
    # Confirm
    response = input(f"⚠️  Reset {len(ingested_files)} files to ingested=false? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Cancelled.")
        exit(0)
    
    print()
    print("🔄 Resetting...")
    print()
    
    # Step 2: Reset each file
    reset_count = 0
    error_count = 0
    
    for i, blob_name in enumerate(ingested_files, 1):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            
            # Get existing metadata
            properties = blob_client.get_blob_properties()
            metadata = properties.metadata or {}
            
            # Update metadata
            metadata['ingested'] = 'false'
            metadata['reset_at'] = datetime.now().isoformat()
            
            # Set updated metadata
            blob_client.set_blob_metadata(metadata)
            
            reset_count += 1
            
            if reset_count % 100 == 0 or i == len(ingested_files):
                print(f"   ✓ Reset {reset_count}/{len(ingested_files)} files")
        
        except Exception as e:
            logger.error(f"Error resetting {blob_name}: {e}")
            error_count += 1
    
    print()
    print("=" * 70)
    print(f"✅ COMPLETE!")
    print(f"   Successfully reset: {reset_count}/{len(ingested_files)} files")
    if error_count > 0:
        print(f"   Errors: {error_count} files")
    print("=" * 70)

except Exception as e:
    logger.error(f"Fatal error: {e}", exc_info=True)
    print(f"❌ Error: {e}")
    exit(1)
