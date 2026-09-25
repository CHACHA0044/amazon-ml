"""Resume prep: build test records + GT long only (train records already built)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import (
    WORK_DIR, TEST_S1, TEST_S2, TEST_S3, TRAIN_GT,
    TEST_RECORDS, GT_LONG,
)
from prep import build_records, build_gt


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK_DIR, "chunks"), exist_ok=True)
    print("Building test records ...", flush=True)
    build_records(TEST_S1, os.path.join(WORK_DIR, "test_s1.parquet"), "te_s1")
    build_records(TEST_S2, os.path.join(WORK_DIR, "test_s2.parquet"), "te_s2")
    build_records(TEST_S3, os.path.join(WORK_DIR, "test_s3.parquet"), "te_s3")

    s1 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s1.parquet"))
    s2 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s2.parquet"))
    s3 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s3.parquet"))
    pl.concat([s1, s2, s3]).sink_parquet(TEST_RECORDS, compression="zstd")
    os.remove(os.path.join(WORK_DIR, "test_s1.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s2.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s3.parquet"))

    print("Building GT long ...", flush=True)
    build_gt(TRAIN_GT, GT_LONG)
    print("PREP RESUME DONE", flush=True)