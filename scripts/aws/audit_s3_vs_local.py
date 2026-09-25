"""Audit local repository files against S3 bucket for Amazon ML Challenge.

Compares local files with S3 objects in s3://sagemaker-us-east-1-974387521486/amazon-ml-challenge/
using filenames, sizes, and MD5 checksums where practical.
"""
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import hashlib
import json
from pathlib import Path

# Configuration from aws_config.py
from aws_config import REGION, BUCKET, S3_PREFIX

# Local repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Define local data directories
RAW_DIR = REPO_ROOT / "dataset" / "student_resource" / "dataset"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
WORK_DIR = REPO_ROOT / "work"
MODELS_DIR = REPO_ROOT / "models"

# Expected S3 prefix structure
S3_PREFIX = S3_PREFIX

def compute_md5(filepath: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_local_files():
    """Collect all local data files with their metadata."""
    local_files = {}
    
    # Raw TSV files (train and test)
    raw_files = [
        (TRAIN_DIR / "train_source1.tsv", f"{S3_PREFIX}/raw/train/train_source1.tsv"),
        (TRAIN_DIR / "train_source2.tsv", f"{S3_PREFIX}/raw/train/train_source2.tsv"),
        (TRAIN_DIR / "train_source3.tsv", f"{S3_PREFIX}/raw/train/train_source3.tsv"),
        (TRAIN_DIR / "train_ground_truth.tsv", f"{S3_PREFIX}/raw/train/train_ground_truth.tsv"),
        (TEST_DIR / "test_source1.tsv", f"{S3_PREFIX}/raw/test/test_source1.tsv"),
        (TEST_DIR / "test_source2.tsv", f"{S3_PREFIX}/raw/test/test_source2.tsv"),
        (TEST_DIR / "test_source3.tsv", f"{S3_PREFIX}/raw/test/test_source3.tsv"),
    ]
    
    for local_path, s3_key in raw_files:
        if local_path.exists():
            local_files[str(local_path.relative_to(REPO_ROOT))] = {
                "path": str(local_path),
                "s3_key": s3_key,
                "size_bytes": local_path.stat().st_size,
                "md5": compute_md5(str(local_path)),
                "type": "raw_tsv"
            }
    
    # Model files
    model_files = [
        (MODELS_DIR / "lgbm_matcher.joblib", f"{S3_PREFIX}/models/lgbm_matcher.joblib"),
    ]
    
    for local_path, s3_key in model_files:
        if local_path.exists():
            local_files[str(local_path.relative_to(REPO_ROOT))] = {
                "path": str(local_path),
                "s3_key": s3_key,
                "size_bytes": local_path.stat().st_size,
                "md5": compute_md5(str(local_path)),
                "type": "model"
            }
    
    # Work/checkpoint files
    work_files = [
        (WORK_DIR / "test_records.parquet", f"{S3_PREFIX}/experiments/EXP003-AWS/checkpoints/01_preprocessed/test_records.parquet"),
    ]
    
    for local_path, s3_key in work_files:
        if local_path.exists():
            local_files[str(local_path.relative_to(REPO_ROOT))] = {
                "path": str(local_path),
                "s3_key": s3_key,
                "size_bytes": local_path.stat().st_size,
                "md5": compute_md5(str(local_path)),
                "type": "checkpoint"
            }
    
    return local_files

def get_s3_objects():
    """List all objects in the S3 bucket/prefix."""
    s3_objects = {}
    
    try:
        session = boto3.Session(region_name=REGION)
        s3_client = session.client("s3")
        
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=BUCKET,
            Prefix=S3_PREFIX
        )
        
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    # Remove the prefix from the key for comparison
                    rel_key = s3_key[len(S3_PREFIX):].lstrip("/")
                    s3_objects[rel_key] = {
                        "key": s3_key,
                        "size_bytes": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "etag": obj.get("ETag", "").strip('"')
                    }
        
        return s3_objects
    except NoCredentialsError:
        print("ERROR: AWS credentials not found")
        return {}
    except ClientError as e:
        print(f"ERROR accessing S3: {e}")
        return {}

def main():
    print("=" * 70)
    print("Amazon ML Challenge - S3 vs Local Repository Audit")
    print("=" * 70)
    print(f"S3 Bucket: s3://{BUCKET}/{S3_PREFIX}/")
    print(f"Region: {REGION}")
    print("=" * 70)
    
    # Get local files
    print("\nCollecting local files...")
    local_files = get_local_files()
    
    # Get S3 objects
    print("Collecting S3 objects...")
    s3_objects = get_s3_objects()
    
    # Perform comparison
    print("\nPerforming comparison...")
    
    # 1. Local files found
    print("\n1. Local files found:")
    for rel_path, info in sorted(local_files.items()):
        print(f"   {rel_path:60} Size: {info['size_bytes']:,} bytes MD5: {info['md5']}")
    
    # 2. Matching S3 objects:
    print("\n2. Matching S3 objects:")
    matches = []
    for rel_path, info in sorted(local_files.items()):
        s3_key = info["s3_key"]
        rel_s3_key = s3_key[len(S3_PREFIX):].lstrip("/")
        if rel_s3_key in s3_objects:
            matches.append((rel_path, info, rel_s3_key, s3_objects[rel_s3_key]))
            print(f"   [MATCH] {rel_path:60} -> S3: {rel_s3_key}")
    
    # 3. Missing from S3
    print("\n3. Missing from S3:")
    missing_from_s3 = []
    for rel_path, info in sorted(local_files.items()):
        s3_key = info["s3_key"]
        rel_s3_key = s3_key[len(S3_PREFIX):].lstrip("/")
        if rel_s3_key not in s3_objects:
            missing_from_s3.append((rel_path, info))
            print(f"   [MISSING] {rel_path:60} (S3 key: {rel_s3_key})")
    
    # 4. Present in S3 but missing locally
    print("\n4. Present in S3 but missing locally:")
    missing_locally = []
    for rel_key, info in sorted(s3_objects.items()):
        # Check if this key corresponds to any local file
        found = False
        for rel_path, local_info in local_files.items():
            if local_info["s3_key"].endswith(rel_key) or rel_key.endswith(local_info["s3_key"]):
                found = True
                break
        if not found:
            missing_locally.append((rel_key, info))
            print(f"   [S3 ONLY] S3: {rel_key:60} Size: {info['size_bytes']:,} bytes")
    
    # 5. Size/checksum mismatches
    print("\n5. Size/checksum mismatches:")
    mismatches = []
    for rel_path, info, rel_s3_key, s3_info in matches:
        if info["size_bytes"] != s3_info["size_bytes"]:
            mismatches.append({
                "type": "size",
                "file": rel_path,
                "local_size": info["size_bytes"],
                "s3_size": s3_info["size_bytes"]
            })
            print(f"   [MISMATCH] {rel_path:60} Size mismatch: Local {info['size_bytes']:,} vs S3 {s3_info['size_bytes']:,}")
        
        # Note: MD5 comparison requires downloading from S3, which we're not doing
        # For now, we'll just note that MD5 comparison would be needed
    
    # 6. Overall verdict
    print("\n6. Overall verdict:")
    total_local = len(local_files)
    total_s3 = len(s3_objects)
    total_matches = len(matches)
    
    print(f"   Total local files: {total_local}")
    print(f"   Total S3 objects: {total_s3}")
    print(f"   Matching files: {total_matches}")
    print(f"   Missing from S3: {len(missing_from_s3)}")
    print(f"   Missing locally: {len(missing_locally)}")
    print(f"   Size mismatches: {len(mismatches)}")
    
    if len(missing_from_s3) == 0 and len(mismatches) == 0:
        print("\n   [PASS] VERDICT: All local data files are present in S3 with matching sizes!")
    else:
        print("\n   [WARN] VERDICT: Some local data files are missing or have mismatches in S3.")
    
    # Generate summary report
    report = {
        "timestamp": "2026-09-25T04:55:36.614Z",  # Current time from environment
        "s3_bucket": BUCKET,
        "s3_prefix": S3_PREFIX,
        "region": REGION,
        "summary": {
            "total_local_files": total_local,
            "total_s3_objects": total_s3,
            "matching_files": total_matches,
            "missing_from_s3": len(missing_from_s3),
            "missing_locally": len(missing_locally),
            "size_mismatches": len(mismatches)
        },
        "missing_from_s3": [
            {"file": rel_path, "size_bytes": info["size_bytes"], "md5": info["md5"]}
            for rel_path, info in missing_from_s3
        ],
        "missing_locally": [
            {"s3_key": rel_key, "size_bytes": info["size_bytes"]}
            for rel_key, info in missing_locally
        ],
        "mismatches": mismatches
    }
    
    # Write report to file
    report_path = REPO_ROOT / "scripts" / "aws" / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport written to: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()