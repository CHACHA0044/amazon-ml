"""Benchmark Runner for EXP-003: Calibrated & Hybrid LightGBM Matcher.

Implements:
  1. Prior-Shift Calibrated LightGBM (correcting 42% sample prior -> 5% test prior)
  2. Hybrid Conjunctive Rule-Gated LightGBM Scorer (combining ML ranking + Rule FP rejection)
  3. Ultra-High Precision Selection Policies
"""

import collections
import gc
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import polars as pl
from src.candidate_generation.blocking import MultiChannelIndex
from src.features.pairwise import char_ngrams, dice_sim, jaccard_sim, containment_sim, compute_pairwise_features
from src.models.rules import evaluate_pair_rules
from src.models.selection import select_policy_c
from src.evaluation.metrics import compute_f05_score
from src.data.split import load_validation_gt

WORK_DIR = os.path.join(REPO_ROOT, "work")
EXP_DIR_3 = os.path.join(REPO_ROOT, "experiments", "EXP-003")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(EXP_DIR_3, exist_ok=True)


def extract_lgbm_features(
    q: Dict,
    c_name_norm: str,
    c_name_suf: str,
    c_addr_norm: str,
    c_nums: List[str],
    c_addr_missing: int,
    is_us: float,
) -> List[float]:
    """Extract feature vector matching LightGBM training schema."""
    q_name_toks = q["name_toks"]
    q_addr_toks = q["addr_toks"]
    c_name_toks = set(c_name_norm.split()) if c_name_norm else set()
    c_addr_toks = set(c_addr_norm.split()) if c_addr_norm else set()

    # Name features
    name_tok_jac = jaccard_sim(q_name_toks, c_name_toks)
    name_contain_q = len(q_name_toks & c_name_toks) / max(1, len(q_name_toks))
    name_contain_c = len(q_name_toks & c_name_toks) / max(1, len(c_name_toks))
    name_c2 = dice_sim(q["name_c2"], char_ngrams(c_name_norm, 2))
    name_c3 = dice_sim(q["name_c3"], char_ngrams(c_name_norm, 3))

    # Addr features
    if not c_addr_norm or not q["addr_norm"]:
        addr_tok_jac = 0.0
        addr_contain_q = 0.0
        addr_contain_c = 0.0
        addr_c2 = 0.0
        addr_c3 = 0.0
    else:
        addr_tok_jac = jaccard_sim(q_addr_toks, c_addr_toks)
        addr_contain_q = len(q_addr_toks & c_addr_toks) / max(1, len(q_addr_toks))
        addr_contain_c = len(q_addr_toks & c_addr_toks) / max(1, len(c_addr_toks))
        addr_c2 = dice_sim(q["addr_c2"], char_ngrams(c_addr_norm, 2))
        addr_c3 = dice_sim(q["addr_c3"], char_ngrams(c_addr_norm, 3))

    # Numeric & Rare
    num_overlap = float(len(set(q["nums"]) & set(c_nums)))
    rare_shared = float(len(q["name_toks"] & c_name_toks))

    # Boolean equality checks
    pool_name_eq = 1.0 if (q["name_norm"] and q["name_norm"] == c_name_norm) else 0.0
    pool_addr_eq = 1.0 if (q["addr_norm"] and q["addr_norm"] == c_addr_norm) else 0.0
    pool_both_eq = 1.0 if (pool_name_eq > 0.5 and pool_addr_eq > 0.5) else 0.0
    pool_ntok_eq = 1.0 if (q_name_toks and q_name_toks == c_name_toks) else 0.0
    pool_atok_eq = 1.0 if (q_addr_toks and q_addr_toks == c_addr_toks) else 0.0
    pool_addr_any_missing = 1.0 if c_addr_missing == 1 or not q["addr_norm"] else 0.0

    return [
        name_tok_jac,
        name_contain_q,
        name_contain_c,
        name_c2,
        name_c3,
        addr_tok_jac,
        addr_contain_q,
        addr_contain_c,
        addr_c2,
        addr_c3,
        num_overlap,
        rare_shared,
        pool_name_eq,
        pool_addr_eq,
        pool_both_eq,
        pool_ntok_eq,
        pool_atok_eq,
        pool_addr_any_missing,
        is_us,
    ]


def run_exp003_benchmark(use_mini: bool = True):
    t0 = time.time()
    print("=" * 70, flush=True)
    print(f"RUNNING EXP-003: CALIBRATED & HYBRID LIGHTGBM BENCHMARK (use_mini={use_mini})", flush=True)
    print("=" * 70, flush=True)

    # 1. Load LightGBM model
    model_path = os.path.join(MODELS_DIR, "lgbm_matcher.joblib")
    print(f"Loading LightGBM model from {model_path}...", flush=True)
    model = joblib.load(model_path)

    # 2. Load validation queries
    split_filename = "val_mini_s1_ids.parquet" if use_mini else "val_s1_ids.parquet"
    split_path = os.path.join(WORK_DIR, "splits", split_filename)
    val_s1_df = pl.read_parquet(split_path)
    val_s1_ids = val_s1_df["s1_id"].to_list()
    val_s1_set = set(val_s1_ids)
    print(f"Loaded {len(val_s1_ids):,} validation queries.", flush=True)

    # 3. Load Ground Truth
    gt_dict = load_validation_gt(val_s1_set)
    total_gt_pairs = sum(len(v) for v in gt_dict.values())
    gt_singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
    print(f"Ground truth loaded: {total_gt_pairs:,} links ({gt_singletons:,} singletons).", flush=True)

    # 4. Load S1 query records
    train_records_path = os.path.join(WORK_DIR, "train_records.parquet")
    q_df = pl.scan_parquet(train_records_path).filter(pl.col("id").is_in(val_s1_ids)).collect()
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
            "is_us": 1.0 if country == "US" else 0.0,
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

    # 5. Load candidate pool (S2 + S3) and build query index
    cand_df = pl.scan_parquet(train_records_path).filter(pl.col("src").is_in([2, 3])).collect()
    index = MultiChannelIndex()
    index.build_index_for_queries(cand_df, q_records)

    # 6. Candidate retrieval & scoring
    print(f"Generating candidate pairs and computing hybrid scores for {len(val_s1_ids):,} queries...", flush=True)
    t_score_start = time.time()

    all_rule_scored: Dict[str, List[Tuple[str, float, int]]] = {qid: [] for qid in val_s1_ids}
    all_hybrid_scored: Dict[str, List[Tuple[str, float, int]]] = {qid: [] for qid in val_s1_ids}

    batch_feat_rows = []
    batch_meta = []  # (qid, cid, rule_conf, rule_tier)

    for qid in val_s1_ids:
        q = q_records[qid]

        cand_map = index.generate_candidates_for_query(
            country=q["country"],
            name_norm=q["name_norm"],
            name_suf=q["name_suf"],
            addr_norm=q["addr_norm"],
            nums=q["nums"],
            max_candidates_per_query=150,
        )

        for c_idx, ch_mask in cand_map.items():
            cid = index.cand_ids[c_idx]
            c_name_norm = index.cand_name_norms[c_idx]
            c_name_suf = index.cand_name_sufs[c_idx]
            c_addr_norm = index.cand_addr_norms[c_idx]
            c_nums = index.cand_nums[c_idx]
            c_addr_missing = index.cand_addr_missing[c_idx]

            # 1. Rule features & score
            feat_dict = compute_pairwise_features(
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
                c_name_toks=set(c_name_norm.split()),
                c_name_c2=char_ngrams(c_name_norm, 2),
                c_name_c3=char_ngrams(c_name_norm, 3),
                c_addr_norm=c_addr_norm,
                c_addr_toks=set(c_addr_norm.split()),
                c_addr_c2=char_ngrams(c_addr_norm, 2),
                c_addr_c3=char_ngrams(c_addr_norm, 3),
                c_nums=c_nums,
                channel_mask=ch_mask,
            )
            rule_conf, rule_tier = evaluate_pair_rules(feat_dict, country=q["country"])
            if rule_tier > 0:
                all_rule_scored[qid].append((cid, rule_conf, rule_tier))

            # 2. LightGBM features
            lgb_feats = extract_lgbm_features(
                q=q,
                c_name_norm=c_name_norm,
                c_name_suf=c_name_suf,
                c_addr_norm=c_addr_norm,
                c_nums=c_nums,
                c_addr_missing=c_addr_missing,
                is_us=q["is_us"],
            )
            batch_feat_rows.append(lgb_feats)
            batch_meta.append((qid, cid, rule_conf, rule_tier))

    print(f"Running LightGBM inference on {len(batch_feat_rows):,} candidate pairs...", flush=True)
    X_mat = np.array(batch_feat_rows, dtype=np.float32)
    lgb_probs = model.predict_proba(X_mat)[:, 1]

    # Prior shift calibration formula (train prior p_tr=0.422 -> test prior p_te=0.06)
    p_tr = 0.422144
    p_te = 0.060
    prior_ratio = (p_te / (1.0 - p_te)) / (p_tr / (1.0 - p_tr))

    for (qid, cid, r_conf, r_tier), p in zip(batch_meta, lgb_probs):
        # Calibrated probability
        odds = (p / max(1e-7, 1.0 - p)) * prior_ratio
        calib_p = odds / (1.0 + odds)

        # Hybrid confidence: geometric mean of LightGBM probability and rule confidence
        if r_tier > 0:
            hybrid_conf = (p ** 0.5) * (r_conf ** 0.5)
            all_hybrid_scored[qid].append((cid, hybrid_conf, r_tier))

    # 7. Evaluate Hybrid Policies
    print("\nEvaluating Hybrid LightGBM + Rule Policies...", flush=True)

    policies = {
        "EXP-002 Policy C (Rule-Only Baseline)": {
            qid: select_policy_c(pairs, anchor_min_conf=0.85, expansion_delta=0.15)
            for qid, pairs in all_rule_scored.items()
        },
    }

    # Hybrid Policy sweeps
    for anchor_thresh in [0.82, 0.85, 0.88, 0.90]:
        for delta in [0.10, 0.12, 0.15]:
            name = f"EXP-003 Hybrid (Anchor={anchor_thresh:.2f}, delta={delta:.2f})"
            preds = {
                qid: select_policy_c(pairs, anchor_min_conf=anchor_thresh, expansion_delta=delta)
                for qid, pairs in all_hybrid_scored.items()
            }
            policies[name] = preds

    # Hybrid Disjoint Cluster Mutually Exclusive Assignment
    for anchor_thresh in [0.85, 0.88]:
        name = f"EXP-003 Hybrid + Disjoint Star Cluster ({anchor_thresh:.2f})"
        all_triples = []
        for qid, pairs in all_hybrid_scored.items():
            for cid, conf, tier in pairs:
                if conf >= anchor_thresh:
                    all_triples.append((conf, qid, cid))
        all_triples.sort(key=lambda x: x[0], reverse=True)
        assigned_cids = set()
        disjoint_preds = {qid: set() for qid in val_s1_ids}
        for conf, qid, cid in all_triples:
            if cid not in assigned_cids:
                if len(disjoint_preds[qid]) < 11:
                    disjoint_preds[qid].add(cid)
                    assigned_cids.add(cid)
        policies[name] = disjoint_preds

    results_table = []
    best_policy_name = None
    best_macro_f05 = -1.0
    best_metrics = None

    for name, pred_dict in policies.items():
        res = compute_f05_score(pred_dict, gt_dict)
        res_us = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "US"},
            {qid: gt_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "US"}
        )
        res_in = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "India"},
            {qid: gt_dict[qid] for qid in val_s1_ids if q_records[qid]["country"] == "India"}
        )
        res_k0 = compute_f05_score(
            {qid: pred_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 0},
            {qid: gt_dict[qid] for qid in val_s1_ids if len(gt_dict[qid]) == 0}
        )

        row = {
            "Policy": name,
            "Macro F0.5": res["f05"],
            "Micro F0.5": res["micro_f05"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "US F0.5": res_us["f05"],
            "India F0.5": res_in["f05"],
            "k=0 F0.5": res_k0["f05"],
        }
        results_table.append(row)

        if res["f05"] > best_macro_f05:
            best_macro_f05 = res["f05"]
            best_policy_name = name
            best_metrics = {
                "overall": res,
                "us": res_us,
                "india": res_in,
                "k0_singleton": res_k0,
                "policy_name": name,
            }

    # Print results table
    print("\n" + "=" * 120, flush=True)
    print(f"{'Hybrid Model / Selection Policy':<56} | {'Macro F0.5':<10} | {'Prec':<7} | {'Recall':<7} | {'US F0.5':<8} | {'IN F0.5':<8} | {'k0 F0.5':<7}", flush=True)
    print("-" * 120, flush=True)
    for row in results_table:
        print(f"{row['Policy']:<56} | {row['Macro F0.5']:<10.4f} | {row['Precision']:<7.4f} | {row['Recall']:<7.4f} | {row['US F0.5']:<8.4f} | {row['India F0.5']:<8.4f} | {row['k=0 F0.5']:<7.4f}", flush=True)
    print("=" * 120, flush=True)

    print(f"\nNEW BEST MACRO F0.5: {best_policy_name} -> {best_macro_f05:.4f}", flush=True)

    # Save metrics JSON
    with open(os.path.join(EXP_DIR_3, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)

    total_time = time.time() - t0
    print(f"EXP-003 complete in {total_time:.1f}s total.", flush=True)
    return best_metrics, results_table


if __name__ == "__main__":
    run_exp003_benchmark(use_mini=True)
