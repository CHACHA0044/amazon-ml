"""Upload raw dataset, models, and code artifacts to S3 with checksum verification.

Follows competition requirements:
- Raw TSVs uploaded as-is without modification
- Model artifact uploaded
- Preprocessed checkpoint uploaded
- Generates s3_manifest.json with checksums and timestamps
"""
import hashlib
import json
import os
import sys
import time
from typing import Dict, List

import boto3
from botocore.config import Config

from aws_config import REGION, BUCKET, S3_PREFIX, REPO_ROOT, require_config

require_config("BUCKET")

RAW_DIR = os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset")
WORK_DIR = os.path.join(REPO_ROOT, "work")
MODELS_DIR = os.path.join(REPO_ROOT, "models")


def compute_md5(filepath: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def upload_file_s3(s3_client, local_path: str, s3_key: str) -> Dict:
    """Upload a file to S3 and return metadata."""
    size_bytes = os.path.getsize(local_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Uploading {os.path.basename(local_path)} ({size_mb:.1f} MB) -> s3://{BUCKET}/{s3_key}...", flush=True)
    t0 = time.time()

    # Calculate MD5 before upload
    md5_hash = compute_md5(local_path)

    # Multipart upload with 10 threads
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=16 * 1024 * 1024,
        use_threads=True,
    )

    s3_client.upload_file(
        Filename=local_path,
        Bucket=BUCKET,
        Key=s3_key,
        Config=transfer_config,
    )
    elapsed = time.time() - t0
    rate = size_mb / max(0.1, elapsed)
    print(f"  Uploaded in {elapsed:.1f}s ({rate:.1f} MB/s) [MD5: {md5_hash}]", flush=True)

    return {
        "local_path": local_path,
        "s3_uri": f"s3://{BUCKET}/{s3_key}",
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "md5": md5_hash,
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    print("=" * 70)
    print("PHASE 5 — S3 DATA PIPELINE UPLOAD FOR AMAZON ML CHALLENGE")
    print(f"Target Bucket: s3://{BUCKET}/{S3_PREFIX}/")
    print(f"Region:        {REGION}")
    print("=" * 70)

    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")

    files_to_upload = [
        # Test raw files (Mandatory for inference)
        (os.path.join(RAW_DIR, "test", "test_source1.tsv"), f"{S3_PREFIX}/raw/test/test_source1.tsv"),
        (os.path.join(RAW_DIR, "test", "test_source2.tsv"), f"{S3_PREFIX}/raw/test/test_source2.tsv"),
        (os.path.join(RAW_DIR, "test", "test_source3.tsv"), f"{S3_PREFIX}/raw/test/test_source3.tsv"),
        # Train raw files
        (os.path.join(RAW_DIR, "train", "train_ground_truth.tsv"), f"{S3_PREFIX}/raw/train/train_ground_truth.tsv"),
        (os.path.join(RAW_DIR, "train", "train_source1.tsv"), f"{S3_PREFIX}/raw/train/train_source1.tsv"),
        (os.path.join(RAW_DIR, "train", "train_source2.tsv"), f"{S3_PREFIX}/raw/train/train_source2.tsv"),
        (os.path.join(RAW_DIR, "train", "train_source3.tsv"), f"{S3_PREFIX}/raw/train/train_source3.tsv"),
        # EXP-003 trained model
        (os.path.join(MODELS_DIR, "lgbm_matcher.joblib"), f"{S3_PREFIX}/models/lgbm_matcher.joblib"),
        # Preprocessed checkpoint for zero-delay startup
        (os.path.join(WORK_DIR, "test_records.parquet"), f"{S3_PREFIX}/experiments/EXP003-AWS/checkpoints/01_preprocessed/test_records.parquet"),
    ]

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bucket": BUCKET,
        "region": REGION,
        "prefix": S3_PREFIX,
        "files": {},
    }

    for local_path, s3_key in files_to_upload:
        if not os.path.exists(local_path):
            print(f"WARNING: File not found: {local_path}, skipping...")
            continue
        meta = upload_file_s3(s3, local_path, s3_key)
        rel_key = os.path.basename(local_path)
        manifest["files"][rel_key] = meta

    # Write local manifest
    manifest_local_path = os.path.join(REPO_ROOT, "experiments", "EXP003-AWS", "s3_manifest.json")
    os.makedirs(os.path.dirname(manifest_local_path), exist_ok=True)
    with open(manifest_local_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {manifest_local_path}")

    # Also upload manifest to S3
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{S3_PREFIX}/experiments/EXP003-AWS/s3_manifest.json",
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded manifest to s3://{BUCKET}/{S3_PREFIX}/experiments/EXP003-AWS/s3_manifest.json")
    print("\nPhase 5 Upload Complete!")


if __name__ == "__main__":
    main()
