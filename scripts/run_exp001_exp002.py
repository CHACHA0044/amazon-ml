"""Complete Benchmark Runner for EXP-001 and EXP-002.

Evaluates:
  1. Candidate Generation Coverage (Blocking Recall@K)
  2. Baseline 0 (Predict Nothing)
  3. Baseline 1 (Exact Name Match)
  4. Baseline 2 (Exact Name + Numeric Agreement)
  5. Baseline 3 (Name + Address Deterministic Rule)
  6. Baseline 4 / EXP-001 (Multi-Channel Blocking + Deterministic Scorer)
  7. EXP-002 Selection Policies:
     - Policy A: Global Pair Threshold
     - Policy B: Tiered Confidence Thresholds
     - Policy C: Per-S1 Ranked Top-K Selection
     - Policy D: Anchor-Based Expansion (Strict Singleton Protection)

Outputs:
  - Macro F0.5 per S1 (official metric)
  - Micro F0.5, Precision, Recall, TP, FP, FN
  - Country breakdown (US vs India)
  - Cardinality breakdown (k=0, 1, 2-3, 4-5, 6+)
  - Structured Error Analysis
"""

import collections
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import polars as pl
from src.candidate_generation.blocking import (
    MultiChannelIndex,
    CH_NAME_EXACT,
    CH_NAME_SUF_EXACT,
    CH_RARE_NAME_TOK,
    CH_NUMERIC,
    CH_ADDR_EXACT,
)
from src.features.pairwise import (
    char_ngrams,
    compute_pairwise_features,
)
from src.models.rules import evaluate_pair_rules
from src.models.selection import (
    select_policy_a,
    select_policy_b,
    select_policy_c,
    select_policy_d,
)
from src.evaluation.metrics import compute_f05_score
from src.data.split import load_validation_gt

WORK_DIR = os.path.join(REPO_ROOT, "work")
EXP_DIR_1 = os.path.join(REPO_ROOT, "experiments", "EXP-001")
EXP_DIR_2 = os.path.join(REPO_ROOT, "experiments", "EXP-002")
os.makedirs(EXP_DIR_1, exist_ok=True)
os.makedirs(EXP_DIR_2, exist_ok=True)


def run_benchmark(use_mini: bool = False, max_queries: int = None):
    start_time = time.time()
    print("=" * 70, flush=True)
    print(f"RUNNING EXP-001 + EXP-002 BENCHMARK (use_mini={use_mini})", flush=True)
    print("=" * 70, flush=True)

    # 1. Load validation S1 IDs
    split_filename = "val_mini_s1_ids.parquet" if use_mini else "val_s1_ids.parquet"
    split_path = os.path.join(WORK_DIR, "splits", split_filename)
    val_s1_df = pl.read_parquet(split_path)
    val_s1_ids = val_s1_df["s1_id"].to_list()
    if max_queries:
        val_s1_ids = val_s1_ids[:max_queries]
    val_s1_set = set(val_s1_ids)
    print(f"Loaded {len(val_s1_ids):,} validation query IDs.", flush=True)

    # 2. Load Ground Truth for validation queries
    gt_dict = load_validation_gt(val_s1_set)
    total_gt_pairs = sum(len(v) for v in gt_dict.values())
    gt_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"Ground Truth loaded: {total_gt_pairs:,} true links, {gt_singletons:,} singletons ({gt_singletons/len(val_s1_set)*100:.2f}%).", flush=True)

    # 3. Load S1 query records first (to build query vocabulary)
    train_records_path = os.path.join(WORK_DIR, "train_records.parquet")
    print("Loading S1 query records...", flush=True)
    q_df = (
        pl.scan_parquet(train_records_path)
        .filter(pl.col("id").is_in(val_s1_ids))
        .collect()
    )

    q_records = {}
    for row in q_df.iter_rows(named=True):
        qid = row["id"]
        country = row["country"]
        n_norm = row["name_norm"] or ""
        n_suf = row["name_norm_suf"] or ""
        a_norm = row["addr_norm"] or ""
        nums = [tok.strip() for tok in (row["addr_num"] or "").split() if tok.strip()]

        q_records[qid] = {
            "country": country,
            "name_norm": n_norm,
            "name_suf": n_suf,
            "addr_norm": a_norm,
            "nums": nums,
            "name_toks": set(n_norm.split()),
            "name_c2": char_ngrams(n_norm, 2),
            "name_c3": char_ngrams(n_norm, 3),
            "addr_toks": set(a_norm.split()),
            "addr_c2": char_ngrams(a_norm, 2),
            "addr_c3": char_ngrams(a_norm, 3),
        }

    # 4. Load candidate records (S2 + S3) and build query-filtered index
    print("Loading candidate records (S2 + S3)...", flush=True)
    cand_df = (
        pl.scan_parquet(train_records_path)
        .filter(pl.col("src").is_in([2, 3]))
        .collect()
    )
    print(f"Candidate records loaded: {cand_df.height:,} rows.", flush=True)

    index = MultiChannelIndex()
    index.build_index_for_queries(cand_df, q_records)

    # Containers for prediction sets across baselines and policies
    preds_b0: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_b1_name_eq: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_b2_name_num: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_b3_name_addr: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_exp1_tiered: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_exp2_pol_a: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_exp2_pol_b: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_exp2_pol_c: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}
    preds_exp2_pol_d: Dict[str, Set[str]] = {qid: set() for qid in val_s1_ids}

    # Tracking candidate generation recall
    retrieved_true_pairs = 0
    total_candidates_generated = 0
    q_cand_counts = []

    # Error analysis containers
    sample_fps = []
    sample_fns = []
    sample_singleton_fps = []

    print(f"Processing candidate retrieval and scoring for {len(val_s1_ids):,} queries...", flush=True)
    eval_start = time.time()
    processed_count = 0

    for qid in val_s1_ids:
        q = q_records[qid]
        true_set = gt_dict.get(qid, set())

        # Candidate Generation
        cand_map = index.generate_candidates_for_query(
            country=q["country"],
            name_norm=q["name_norm"],
            name_suf=q["name_suf"],
            addr_norm=q["addr_norm"],
            nums=q["nums"],
            max_candidates_per_query=150,
        )

        n_cands = len(cand_map)
        total_candidates_generated += n_cands
        q_cand_counts.append(n_cands)

        # Measure blocking recall
        cand_id_set = {index.cand_ids[c_idx] for c_idx in cand_map}
        retrieved_true_pairs += len(cand_id_set & true_set)

        scored_candidates: List[Tuple[str, float, int]] = []

        # Feature computation & baseline rule evaluation on-the-fly for retrieved candidates
        for c_idx, ch_mask in cand_map.items():
            cid = index.cand_ids[c_idx]
            c_name_norm = index.cand_name_norms[c_idx]
            c_name_suf = index.cand_name_sufs[c_idx]
            c_addr_norm = index.cand_addr_norms[c_idx]
            c_nums = index.cand_nums[c_idx]

            c_name_toks = set(c_name_norm.split())
            c_name_c2 = char_ngrams(c_name_norm, 2)
            c_name_c3 = char_ngrams(c_name_norm, 3)

            c_addr_toks = set(c_addr_norm.split())
            c_addr_c2 = char_ngrams(c_addr_norm, 2)
            c_addr_c3 = char_ngrams(c_addr_norm, 3)

            feat = compute_pairwise_features(
                q_name_norm=q["name_norm"],
                q_name_suf=q["name_suf"],
                q_name_toks=q["name_toks"],
                q_name_c2=q["name_c2"],
                q_name_c3=q["name_c3"],
                q_addr_norm=q["addr_norm"],
                q_addr_toks=q["addr_toks"],
                q_addr_c2=q["addr_c2"],
                q_addr_c3=q["addr_c3"],
                q_nums=q["nums"],
                c_name_norm=c_name_norm,
                c_name_suf=c_name_suf,
                c_name_toks=c_name_toks,
                c_name_c2=c_name_c2,
                c_name_c3=c_name_c3,
                c_addr_norm=c_addr_norm,
                c_addr_toks=c_addr_toks,
                c_addr_c2=c_addr_c2,
                c_addr_c3=c_addr_c3,
                c_nums=c_nums,
                channel_mask=ch_mask,
            )

            # Baseline 1: Exact Name
            if feat["name_eq"] > 0.5:
                preds_b1_name_eq[qid].add(cid)

            # Baseline 2: Exact Name + Numeric
            if feat["name_eq"] > 0.5 and feat["num_overlap"] >= 1.0:
                preds_b2_name_num[qid].add(cid)

            # Baseline 3: Strong Name + Address
            if (feat["name_tok_jac"] >= 0.6 or feat["name_eq"] > 0.5) and feat["addr_tok_jac"] >= 0.5:
                preds_b3_name_addr[qid].add(cid)

            # Scorer / Tier evaluation
            conf, tier = evaluate_pair_rules(feat, country=q["country"])
            if tier > 0:
                scored_candidates.append((cid, conf, tier))

        # Apply Selection Policies
        # EXP-001 (Tiered acceptance: Tier 1 + Tier 2)
        preds_exp1_tiered[qid] = {cid for cid, conf, tier in scored_candidates if tier in (1, 2)}

        # EXP-002 Policies
        preds_exp2_pol_a[qid] = select_policy_a(scored_candidates, threshold=0.75)
        preds_exp2_pol_b[qid] = select_policy_b(scored_candidates, tier1_thresh=0.88, tier2_thresh=0.75)
        preds_exp2_pol_c[qid] = select_policy_c(scored_candidates, anchor_min_conf=0.85, expansion_delta=0.15)
        preds_exp2_pol_d[qid] = select_policy_d(scored_candidates, tier1_min_conf=0.88, tier2_anchor_conf=0.82, tier2_expansion_conf=0.75)

        # Sample error cases for qualitative diagnosis
        pred_d = preds_exp2_pol_d[qid]
        fps = pred_d - true_set
        fns = true_set - pred_d

        if fps and len(true_set) == 0 and len(sample_singleton_fps) < 10:
            sample_singleton_fps.append({
                "s1_id": qid,
                "country": q["country"],
                "s1_name": q["name_norm"],
                "s1_addr": q["addr_norm"],
                "fp_cids": list(fps),
            })
        if fps and len(sample_fps) < 10:
            sample_fps.append({
                "s1_id": qid,
                "country": q["country"],
                "s1_name": q["name_norm"],
                "s1_addr": q["addr_norm"],
                "fp_cids": list(fps),
            })
        if fns and len(sample_fns) < 10:
            sample_fns.append({
                "s1_id": qid,
                "country": q["country"],
                "s1_name": q["name_norm"],
                "s1_addr": q["addr_norm"],
                "fn_cids": list(fns),
                "retrieved_in_blocking": list(true_set & cand_id_set),
                "missed_in_blocking": list(true_set - cand_id_set),
            })

        processed_count += 1
        if processed_count % 2000 == 0:
            elapsed = time.time() - eval_start
            print(f"Processed {processed_count:,}/{len(val_s1_ids):,} queries ({processed_count/elapsed:.1f} q/s)...", flush=True)

    eval_time = time.time() - eval_start
    blocking_recall = retrieved_true_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0
    avg_cands_per_q = total_candidates_generated / len(val_s1_ids)

    print("-" * 70, flush=True)
    print("CANDIDATE GENERATION RECALL & EFFICIENCY:", flush=True)
    print(f"  Blocking Recall@150: {blocking_recall*100:.2f}% ({retrieved_true_pairs:,}/{total_gt_pairs:,} true links)", flush=True)
    print(f"  Avg candidates per query: {avg_cands_per_q:.1f}", flush=True)
    print(f"  Evaluation throughput: {len(val_s1_ids)/eval_time:.1f} queries/sec ({eval_time:.1f}s)", flush=True)
    print("-" * 70, flush=True)

    # 6. Evaluate all Baselines & Policies
    models_to_eval = [
        ("Baseline 0 (Predict Nothing)", preds_b0),
        ("Baseline 1 (Exact Name)", preds_b1_name_eq),
        ("Baseline 2 (Exact Name + Numeric)", preds_b2_name_num),
        ("Baseline 3 (Name + Address Rule)", preds_b3_name_addr),
        ("EXP-001 (Deterministic Tiered)", preds_exp1_tiered),
        ("EXP-002 Policy A (Global Thresh 0.75)", preds_exp2_pol_a),
        ("EXP-002 Policy B (Tiered 0.88/0.75)", preds_exp2_pol_b),
        ("EXP-002 Policy C (Per-S1 Ranked)", preds_exp2_pol_c),
        ("EXP-002 Policy D (Anchor Expansion)", preds_exp2_pol_d),
    ]

    results_table = []
    detailed_metrics = {}

    for name, pred_dict in models_to_eval:
        res = compute_f05_score(pred_dict, gt_dict)
        n_predicted_s1 = sum(1 for p in pred_dict.values() if len(p) > 0)
        total_pred_pairs = sum(len(p) for p in pred_dict.values())
        avg_preds = total_pred_pairs / max(1, n_predicted_s1)

        # Country breakdown
        res_us = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "US"},
            {qid: gt_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "US"}
        )
        res_in = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "India"},
            {qid: gt_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "India"}
        )

        # Cardinality breakdown
        res_k0 = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 0},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 0}
        )
        res_k1 = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 1},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 1}
        )
        res_k2_3 = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) in (2, 3)},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) in (2, 3)}
        )
        res_k4_5 = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) in (4, 5)},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) in (4, 5)}
        )
        res_k6_plus = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) >= 6},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) >= 6}
        )

        detailed_metrics[name] = {
            "overall": res,
            "us": res_us,
            "india": res_in,
            "cardinality": {
                "k0_singleton": res_k0,
                "k1": res_k1,
                "k2_3": res_k2_3,
                "k4_5": res_k4_5,
                "k6_plus": res_k6_plus,
            },
            "stats": {
                "predicted_s1_count": n_predicted_s1,
                "predicted_pairs_total": total_pred_pairs,
                "avg_preds_per_predicted_s1": round(avg_preds, 2),
            }
        }

        results_table.append({
            "Model / Policy": name,
            "Macro F0.5": res["f05"],
            "Micro F0.5": res["micro_f05"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "TP": res["tp"],
            "FP": res["fp"],
            "FN": res["fn"],
            "US F0.5": res_us["f05"],
            "India F0.5": res_in["f05"],
            "k=0 F0.5": res_k0["f05"],
        })

    # Print summary table
    print("\n" + "=" * 110, flush=True)
    print(f"{'Model / Policy':<36} | {'Macro F0.5':<10} | {'Prec':<7} | {'Recall':<7} | {'US F0.5':<8} | {'IN F0.5':<8} | {'k0 F0.5':<7}", flush=True)
    print("-" * 110, flush=True)
    for row in results_table:
        print(f"{row['Model / Policy']:<36} | {row['Macro F0.5']:<10.4f} | {row['Precision']:<7.4f} | {row['Recall']:<7.4f} | {row['US F0.5']:<8.4f} | {row['India F0.5']:<8.4f} | {row['k=0 F0.5']:<7.4f}", flush=True)
    print("=" * 110, flush=True)

    # Save metrics JSONs
    with open(os.path.join(EXP_DIR_1, "metrics.json"), "w") as f:
        json.dump(detailed_metrics["EXP-001 (Deterministic Tiered)"], f, indent=2)
    with open(os.path.join(EXP_DIR_2, "metrics.json"), "w") as f:
        json.dump(detailed_metrics["EXP-002 Policy D (Anchor Expansion)"], f, indent=2)

    # Save error analysis
    error_analysis_data = {
        "blocking_recall": blocking_recall,
        "sample_fps": sample_fps,
        "sample_fns": sample_fns,
        "sample_singleton_fps": sample_singleton_fps,
        "summary_table": results_table,
    }
    with open(os.path.join(WORK_DIR, "analysis_exp001_exp002.json"), "w") as f:
        json.dump(error_analysis_data, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.1f}s.", flush=True)
    return detailed_metrics, error_analysis_data


if __name__ == "__main__":
    run_benchmark(use_mini=True)
