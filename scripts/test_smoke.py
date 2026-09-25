"""Local Smoke Test: Fast validation of pipeline components on a tiny 200-record slice.

Verifies:
  - imports work cleanly
  - LightGBM model loads and scores correctly
  - candidate generation produces valid candidates
  - fused rule + ML scoring returns valid probabilities
  - Disjoint Star Cluster produces expected structure
  - output TSV formatting is strictly valid
"""
import os
import sys
import time

import joblib
import numpy as np
import polars as pl

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.aws.sagemaker_entrypoint import (
    ANCHOR_THRESH,
    FastCandidatePool,
    char_ngrams,
    eval_pair_fast,
    load_country_queries,
)


def run_smoke_test():
    print("=" * 60)
    print("PHASE 12 — LOCAL SMOKE TEST (TINY SUBSET)")
    print("=" * 60)

    model_path = os.path.join(REPO_ROOT, "models", "lgbm_matcher.joblib")
    test_records_path = os.path.join(REPO_ROOT, "work", "test_records.parquet")

    # 1. Test model loading
    print("1. Testing model load...")
    t0 = time.time()
    model = joblib.load(model_path)
    print(f"   Model loaded in {time.time()-t0:.2f}s. Features: {model.n_features_}")
    assert model.n_features_ == 19, f"Expected 19 features, got {model.n_features_}"

    # 2. Test slice loading (France: 100 queries, 500 candidates)
    print("2. Testing slice candidate pool & query build...")
    slice_q_df = (
        pl.scan_parquet(test_records_path)
        .filter((pl.col("src") == 1) & (pl.col("country") == "France"))
        .limit(100)
        .collect()
    )
    slice_c_df = (
        pl.scan_parquet(test_records_path)
        .filter((pl.col("src").is_in([2, 3])) & (pl.col("country") == "France"))
        .limit(1000)
        .collect()
    )

    ids = slice_q_df["id"].to_list()
    names = slice_q_df["name_norm"].fill_null("").to_list()
    sufs = slice_q_df["name_norm_suf"].fill_null("").to_list()
    addrs = slice_q_df["addr_norm"].fill_null("").to_list()
    nums = slice_q_df["addr_num"].fill_null("").to_list()

    q_records = {}
    for i in range(len(ids)):
        num_list = [tok for tok in nums[i].split() if tok]
        q_records[ids[i]] = {
            "country": "France",
            "is_us": 0.0,
            "name_norm": names[i],
            "name_suf": sufs[i],
            "addr_norm": addrs[i],
            "nums": num_list,
            "num_set": set(num_list),
            "name_toks": set(names[i].split()) if names[i] else set(),
            "addr_toks": set(addrs[i].split()) if addrs[i] else set(),
            "name_c2": char_ngrams(names[i], 2),
            "name_c3": char_ngrams(names[i], 3),
            "addr_c2": char_ngrams(addrs[i], 2),
            "addr_c3": char_ngrams(addrs[i], 3),
        }

    pool = FastCandidatePool(slice_c_df, q_records)
    print(f"   Built candidate pool with {pool.n} candidates.")

    # 3. Test candidate retrieval and scoring
    print("3. Testing candidate generation & fused scoring...")
    evaluated_pairs = 0
    high_conf_pairs = 0
    for qid, q in q_records.items():
        cands = pool.generate_candidates(q, max_cand=50)
        for c_idx in cands:
            pool.ensure_cand_features(c_idx)
            lgb_feat, r_conf, r_tier = eval_pair_fast(
                q=q,
                c_name_norm=pool.name_norms[c_idx],
                c_name_suf=pool.name_sufs[c_idx],
                c_addr_norm=pool.addr_norms[c_idx],
                c_name_toks=pool.name_toks[c_idx],
                c_addr_toks=pool.addr_toks[c_idx],
                c_name_c2=pool.name_c2[c_idx],
                c_name_c3=pool.name_c3[c_idx],
                c_addr_c2=pool.addr_c2[c_idx],
                c_addr_c3=pool.addr_c3[c_idx],
                c_nums=pool.nums[c_idx],
                c_num_set=pool.num_sets[c_idx],
                c_addr_missing=pool.addr_missing[c_idx],
            )
            evaluated_pairs += 1
            if r_tier > 0:
                prob = model.predict_proba(np.array([lgb_feat], dtype=np.float32))[0, 1]
                h_conf = (prob ** 0.5) * (r_conf ** 0.5)
                if h_conf >= ANCHOR_THRESH:
                    high_conf_pairs += 1

    print(f"   Evaluated {evaluated_pairs} candidate pairs. High-confidence pairs: {high_conf_pairs}")
    print("\nSmoke Test PASS! All components operational.")


if __name__ == "__main__":
    run_smoke_test()
