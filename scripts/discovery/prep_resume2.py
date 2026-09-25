"""Resume: rebuild test_s3 with fewer workers, concat test, build GT."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import (
    WORK_DIR, TEST_S3, TRAIN_GT, TEST_RECORDS, GT_LONG,
)
import prep

prep.MAX_WORKERS = 3
prep.CHUNK = 150_000


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK_DIR, "chunks"), exist_ok=True)
    start = time.time()
    prep.build_records(TEST_S3, os.path.join(WORK_DIR, "test_s3.parquet"), "te_s3")
    print(f"test_s3 done in {time.time()-start:.1f}s", flush=True)

    s1 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s1.parquet"))
    s2 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s2.parquet"))
    s3 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s3.parquet"))
    pl.concat([s1, s2, s3]).sink_parquet(TEST_RECORDS, compression="zstd")
    os.remove(os.path.join(WORK_DIR, "test_s1.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s2.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s3.parquet"))

    prep.build_gt(TRAIN_GT, GT_LONG)
    print("ALL PREP DONE", flush=True)