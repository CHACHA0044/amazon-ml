"""Entity-level Stratified Validation Split Module.

Creates deterministic, cluster-isolated train/validation splits by partitioning
Source 1 (query) entities. Ensures:
1. Zero leakage: S1 entities and their true ground-truth links are isolated.
2. Real-world candidate pool: Validation queries are searched against the FULL
   candidate pool (including all distractors / noise candidates) to match test conditions.
3. Stratification: Preserves country distribution (US vs India) and match-cardinality
   distribution (singletons, 1, 2-3, 4-5, 6+ matches).
"""

import os
import random
from typing import Dict, List, Set, Tuple
import polars as pl

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")
SPLITS_DIR = os.path.join(WORK_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)


def create_stratified_validation_split(
    val_size: int = 100_000,
    mini_val_size: int = 10_000,
    seed: int = 42
) -> Dict[str, str]:
    """Generate deterministic validation and mini-validation S1 ID splits."""
    train_records_path = os.path.join(WORK_DIR, "train_records.parquet")
    gt_long_path = os.path.join(WORK_DIR, "gt_long.parquet")

    # Load S1 records
    s1_df = (
        pl.scan_parquet(train_records_path)
        .filter(pl.col("src") == 1)
        .select(["id", "country"])
        .collect()
    )

    # Count true matches per S1 from ground truth
    gt_counts = (
        pl.scan_parquet(gt_long_path)
        .group_by("s1_id")
        .agg(pl.len().alias("match_count"))
        .collect()
    )

    # Join match count to S1 records (null -> 0 matches / singleton)
    s1_with_gt = s1_df.join(
        gt_counts, left_on="id", right_on="s1_id", how="left"
    ).with_columns(
        pl.col("match_count").fill_null(0).alias("match_count")
    )

    # Define cardinality strata: 0 (singleton), 1, 2-3, 4-5, 6+
    def bucket_cardinality(c: int) -> str:
        if c == 0:
            return "k0"
        elif c == 1:
            return "k1"
        elif c in (2, 3):
            return "k2_3"
        elif c in (4, 5):
            return "k4_5"
        else:
            return "k6_plus"

    s1_with_strata = s1_with_gt.with_columns(
        pl.struct(["country", "match_count"]).map_elements(
            lambda x: f"{x['country']}_{bucket_cardinality(x['match_count'])}",
            return_dtype=pl.String
        ).alias("stratum")
    )

    # Stratified sampling
    total_s1 = s1_with_strata.height
    val_fraction = val_size / total_s1
    mini_fraction = mini_val_size / val_size

    rng = random.Random(seed)

    # Group by stratum and sample
    val_ids: List[str] = []
    mini_val_ids: List[str] = []

    strata_counts = s1_with_strata.group_by("stratum").agg(pl.col("id")).to_dicts()

    for row in strata_counts:
        stratum = row["stratum"]
        ids = row["id"]
        rng.shuffle(ids)

        k_val = max(1, int(round(len(ids) * val_fraction)))
        sampled_val = ids[:k_val]
        val_ids.extend(sampled_val)

        k_mini = max(1, int(round(len(sampled_val) * mini_fraction)))
        mini_val_ids.extend(sampled_val[:k_mini])

    # Save to parquet
    val_df = pl.DataFrame({"s1_id": val_ids})
    mini_val_df = pl.DataFrame({"s1_id": mini_val_ids})

    val_path = os.path.join(SPLITS_DIR, "val_s1_ids.parquet")
    mini_val_path = os.path.join(SPLITS_DIR, "val_mini_s1_ids.parquet")

    val_df.write_parquet(val_path)
    mini_val_df.write_parquet(mini_val_path)

    print(f"Validation split created: {len(val_ids)} S1 IDs -> {val_path}")
    print(f"Mini-validation split created: {len(mini_val_ids)} S1 IDs -> {mini_val_path}")

    return {
        "val_path": val_path,
        "mini_val_path": mini_val_path,
        "val_count": len(val_ids),
        "mini_val_count": len(mini_val_ids),
    }


def load_validation_gt(s1_ids_set: Set[str]) -> Dict[str, Set[str]]:
    """Load ground truth for a given set of S1 IDs as {s1_id: set(matched_ids)}."""
    gt_long_path = os.path.join(WORK_DIR, "gt_long.parquet")
    
    # Pre-populate all S1 entities with empty sets (singletons included!)
    gt_dict = {s1_id: set() for s1_id in s1_ids_set}

    gt_df = (
        pl.scan_parquet(gt_long_path)
        .filter(pl.col("s1_id").is_in(list(s1_ids_set)))
        .select(["s1_id", "matched_id"])
        .collect()
    )

    for row in gt_df.iter_rows():
        s1_id, matched_id = row
        if s1_id in gt_dict:
            gt_dict[s1_id].add(matched_id)

    return gt_dict


if __name__ == "__main__":
    create_stratified_validation_split(val_size=100_000, mini_val_size=10_000, seed=42)
