"""Generate 100% valid Unstop submission in under 15 seconds using pure Polars C/Rust engine.

Vectorized pipeline:
1. Load test_records.parquet
2. Exact name_norm + suffix name_norm_suf inner joins across S1 and (S2+S3)
3. Vectorized Disjoint Star Clustering (unique cand_id by confidence, cum_count <= 11)
4. Left join to all 1,732,544 test S1 IDs (nulls filled with empty string)
5. Direct write_csv(separator='\t')
6. Official validation + ZIP creation
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def generate_baseline():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("VECTORIZED POLARS BASELINE GENERATOR (< 15 SECONDS)", flush=True)
    print("=" * 70, flush=True)

    parquet_path = os.path.join(REPO_ROOT, "work", "test_records.parquet")
    print(f"Loading {parquet_path}...", flush=True)
    df = pl.read_parquet(parquet_path, columns=["src", "id", "country", "name_norm", "name_norm_suf"])
    print(f"Loaded {df.height:,} records in {time.time()-t0:.1f}s.", flush=True)

    # Separate S1 (queries) and S2+S3 (candidates)
    s1_df = df.filter(pl.col("src") == 1).select(["id", "country", "name_norm", "name_norm_suf"])
    cand_df = df.filter(pl.col("src") != 1).select(["id", "country", "name_norm", "name_norm_suf"])

    print(f"Total S1 entities: {s1_df.height:,}, Candidate pool: {cand_df.height:,}", flush=True)

    # 1. Exact name match join
    t_join = time.time()
    print("\n1. Finding exact name matches...", flush=True)
    s1_valid_name = s1_df.filter(pl.col("name_norm").str.len_bytes() >= 3)
    cand_valid_name = cand_df.filter(pl.col("name_norm").str.len_bytes() >= 3)

    exact_matches = s1_valid_name.join(
        cand_valid_name,
        on=["country", "name_norm"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("id").alias("s1_id"),
        pl.col("id_cand").alias("cand_id"),
        pl.lit(0.95).alias("confidence")
    ])
    print(f"   Found {exact_matches.height:,} exact matches in {time.time()-t_join:.1f}s.", flush=True)

    # 2. Suffix match join
    t_suf = time.time()
    print("\n2. Finding suffix-normalized matches...", flush=True)
    s1_valid_suf = s1_df.filter((pl.col("name_norm_suf").str.len_bytes() >= 3) & (pl.col("name_norm_suf") != pl.col("name_norm")))
    cand_valid_suf = cand_df.filter((pl.col("name_norm_suf").str.len_bytes() >= 3) & (pl.col("name_norm_suf") != pl.col("name_norm")))

    suf_matches = s1_valid_suf.join(
        cand_valid_suf,
        on=["country", "name_norm_suf"],
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("id").alias("s1_id"),
        pl.col("id_cand").alias("cand_id"),
        pl.lit(0.88).alias("confidence")
    ])
    print(f"   Found {suf_matches.height:,} suffix matches in {time.time()-t_suf:.1f}s.", flush=True)

    # Combine
    all_pairs = pl.concat([exact_matches, suf_matches]).unique(subset=["s1_id", "cand_id"])
    print(f"\nTotal candidate pairs: {all_pairs.height:,}", flush=True)

    # 3. Vectorized Disjoint Star Cluster
    t_clust = time.time()
    print("\n3. Vectorized Disjoint Star Clustering...", flush=True)
    # Disjoint 1-to-1: each cand_id assigned to top-confidence S1
    disjoint_pairs = (
        all_pairs
        .sort(["confidence", "s1_id"], descending=[True, False])
        .unique(subset=["cand_id"], keep="first")
        .with_columns(
            pl.int_range(0, pl.len()).over("s1_id").alias("rank")
        )
        .filter(pl.col("rank") < 11)
    )
    print(f"   Disjoint matched pairs: {disjoint_pairs.height:,} in {time.time()-t_clust:.1f}s.", flush=True)

    # 4. Group matches per S1 and right-join to all S1 IDs
    t_out = time.time()
    print("\n4. Formatting matching_results.tsv and candidate_pairs.tsv...", flush=True)
    
    grouped_matches = (
        disjoint_pairs
        .group_by("s1_id")
        .agg(pl.col("cand_id").str.join(","))
        .rename({"s1_id": "source1_entity_id", "cand_id": "matched_entity_ids"})
    )

    # Candidate pairs must contain all disjoint matched pairs + top candidate pairs
    all_cand_pairs = (
        pl.concat([
            disjoint_pairs.select(["s1_id", "cand_id"]),
            all_pairs.select(["s1_id", "cand_id"])
        ])
        .unique(subset=["s1_id", "cand_id"])
        .with_columns(pl.int_range(0, pl.len()).over("s1_id").alias("c_rank"))
        .filter(pl.col("c_rank") < 50)
        .group_by("s1_id")
        .agg(pl.col("cand_id").str.join(","))
        .rename({"s1_id": "source1_entity_id", "cand_id": "candidate_entity_ids"})
    )

    all_s1_base = s1_df.select(pl.col("id").alias("source1_entity_id"))

    matching_df = (
        all_s1_base
        .join(grouped_matches, on="source1_entity_id", how="left")
        .select(["source1_entity_id", "matched_entity_ids"])
    )

    candidate_df = (
        all_s1_base
        .join(all_cand_pairs, on="source1_entity_id", how="left")
        .select(["source1_entity_id", "candidate_entity_ids"])
    )

    matching_tsv = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    matching_df.write_csv(matching_tsv, separator="\t", null_value="")
    candidate_df.write_csv(candidate_tsv, separator="\t", null_value="")
    print(f"   TSVs written in {time.time()-t_out:.1f}s. Row count: {matching_df.height:,}", flush=True)

    # 5. Validate
    print("\n5. Running official submission validator...", flush=True)
    test_dir = os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset", "test")
    val_cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "validate_submission.py"),
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--test-dir", test_dir,
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, flush=True)

    if res.returncode != 0:
        raise RuntimeError("Validation FAILED!")

    # 6. Package ZIP
    print("\n6. Packaging final submission ZIP...", flush=True)
    zip_path = os.path.join(SUBMISSIONS_DIR, "EXP001_Fast_Baseline_submission.zip")
    standalone_matching = os.path.join(SUBMISSIONS_DIR, "EXP001_Fast_Baseline_matching_results.tsv")
    shutil.copy2(matching_tsv, standalone_matching)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_tsv, "output/matching_results.tsv")
        zf.write(candidate_tsv, "output/candidate_pairs.tsv")

        src_dir = os.path.join(REPO_ROOT, "src")
        for root, dirs, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, REPO_ROOT)
                    zf.write(full_p, os.path.join("code", "business_entity_resolution", rel_p))

        readme = """# Business Entity Resolution Solution (Fast Exact Baseline)
Amazon ML Challenge 2026

## Method
Vectorized exact and suffix normalized business entity matching with 1-to-1 Disjoint Star Clustering.

## Instructions
```bash
pip install polars pyarrow
python scripts/generate_fast_baseline.py
```
"""
        zf.writestr("code/business_entity_resolution/README.md", readme)
        zf.writestr("code/business_entity_resolution/requirements.txt", "polars>=1.0.0\npyarrow>=14.0.0\n")

        doc = os.path.join(REPO_ROOT, "docs", "problem_understanding.md")
        if os.path.exists(doc):
            zf.write(doc, "Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n" + "=" * 70, flush=True)
    print(f"SUBMISSION READY IN {time.time()-t0:.1f} SECONDS TOTAL!", flush=True)
    print(f"1. Standalone TSV for Leaderboard Portal: {standalone_matching}", flush=True)
    print(f"2. Complete Submission ZIP:             {zip_path} ({zip_size_mb:.1f} MB)", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    generate_baseline()
