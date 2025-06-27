"""
Shared storage functionality for data connectors.
Handles writing raw data to the configured storage location.

Environment variables:
    STORAGE_BUCKET: Optional bucket name for cloud storage
    STORAGE_PREFIX: Prefix for storage path
    STORAGE_TYPE: 'local', 's3', etc. (defaults to 'local')
"""
import os
import json
import pathlib
from datetime import datetime
from typing import Dict, List, Union, Any

# Storage configuration
STORAGE_TYPE = os.environ.get("STORAGE_TYPE", "local").lower()
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "")
STORAGE_PREFIX = os.environ.get("STORAGE_PREFIX", "hotel_data")

# If using S3, import boto3
if STORAGE_TYPE == "s3":
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 storage. Install it with: pip install boto3"
        )


def dump_raw_to_storage(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    data_type: str
) -> str:
    """
    Write raw data to storage.

    Args:
        payload: Data to store
        data_type: Type of data (e.g., 'availability', 'financial_txns')

    Returns:
        Path where data was stored
    """
    # Validate input
    if not payload:
        raise ValueError("Empty payload provided")
    if not data_type:
        raise ValueError("Data type must be specified")

    # Create timestamp for the file
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{data_type}_{timestamp}.json"

    # Construct storage path
    if STORAGE_TYPE == "local":
        # Local file system storage
        base_path = os.environ.get("LOCAL_STORAGE_PATH", "storage")
        # Ensure the directory exists
        storage_path = pathlib.Path(base_path) / STORAGE_PREFIX / data_type / date_str
        storage_path.mkdir(parents=True, exist_ok=True)

        # Full path to the file
        file_path = storage_path / filename

        # Write the file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        return str(file_path)

    elif STORAGE_TYPE == "s3":
        # S3 storage
        if not STORAGE_BUCKET:
            raise ValueError("STORAGE_BUCKET environment variable must be set for S3 storage")

        s3 = boto3.client("s3")
        s3_key = f"{STORAGE_PREFIX}/{data_type}/{date_str}/{filename}"

        # Write to S3
        s3.put_object(
            Bucket=STORAGE_BUCKET,
            Key=s3_key,
            Body=json.dumps(payload, ensure_ascii=False, default=str),
            ContentType="application/json"
        )

        return f"s3://{STORAGE_BUCKET}/{s3_key}"

    else:
        raise ValueError(f"Unsupported storage type: {STORAGE_TYPE}")
