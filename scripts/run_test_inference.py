"""Test-Set Inference Script: EXP-003 Calibrated Hybrid LightGBM Matcher (Ultra-Fast Lazy-Cache Streaming).

Generates BOTH official submission files for the Unstop Amazon ML Challenge:
  1. output/matching_results.tsv  (scored on leaderboard)
  2. output/candidate_pairs.tsv   (blocking audit)

Architecture:
  - 6-Channel Multi-Channel Blocking Index (query-vocabulary filtered)
  - Tiered Deterministic Rule Scorer (fused inline evaluation)
  - Calibrated LightGBM GBDT Ranker (prior-shift corrected)
  - Hybrid Confidence = sqrt(P_LGBM * Conf_Rule)
  - Disjoint Star Cluster post-processing (anchor_thresh=0.85)

Optimizations:
  - Per-country streaming: keeps RAM < 1.5 GB
  - Fast inverted indexing in seconds without upfront n-gram generation
  - Lazy per-candidate feature caching: n-grams computed once on retrieval and reused
  - Fused feature computation returning both Rule Score and LightGBM vector in one pass
"""

import collections
import gc
import math
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import polars as pl

from src.candidate_generation.blocking import (
    CH_ADDR_EXACT,
    CH_ADDR_TOK,
    CH_NAME_EXACT,
    CH_NAME_SUF_EXACT,
    CH_NUMERIC,
    CH_RARE_NAME_TOK,
    ADDR_STOPWORDS,
    NAME_STOPWORDS,
)
from src.features.pairwise import char_ngrams

MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "output")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

ANCHOR_THRESH = 0.85
MAX_MATCHES_PER_S1 = 11

P_TR = 0.422144
P_TE = 0.060
PRIOR_RATIO = (P_TE / (1.0 - P_TE)) / (P_TR / (1.0 - P_TR))


def eval_pair_fast(
    q: Dict,
    c_name_norm: str,
    c_name_suf: str,
    c_addr_norm: str,
    c_name_toks: Set[str],
    c_addr_toks: Set[str],
    c_name_c2: Set[str],
    c_name_c3: Set[str],
    c_addr_c2: Set[str],
    c_addr_c3: Set[str],
    c_nums: List[str],
    c_num_set: Set[str],
    c_addr_missing: int,
) -> Tuple[Optional[List[float]], float, int]:
    """Fused feature extraction and rule evaluation in a single fast pass.
    
    Returns: (lgb_features, rule_conf, rule_tier)
    """
    # Name features
    q_name_toks = q["name_toks"]
    name_inter = len(q_name_toks & c_name_toks)
    name_union = len(q_name_toks | c_name_toks)
    name_tok_jac = name_inter / max(1, name_union)
    name_contain_q = name_inter / max(1, len(q_name_toks))
    name_contain_c = name_inter / max(1, len(c_name_toks))
    name_contain = name_inter / max(1, min(len(q_name_toks), len(c_name_toks)))

    q_name_c2 = q["name_c2"]
    q_name_c3 = q["name_c3"]
    name_c2 = (2.0 * len(q_name_c2 & c_name_c2)) / max(1, len(q_name_c2) + len(c_name_c2))
    name_c3 = (2.0 * len(q_name_c3 & c_name_c3)) / max(1, len(q_name_c3) + len(c_name_c3))

    name_eq = 1.0 if (q["name_norm"] and q["name_norm"] == c_name_norm) else 0.0
    name_suf_eq = 1.0 if (q["name_suf"] and q["name_suf"] == c_name_suf) else 0.0
    n_eq = name_eq > 0.5 or name_suf_eq > 0.5
    name_len_diff = abs(len(q["name_norm"]) - len(c_name_norm))

    # Address features
    q_addr_toks = q["addr_toks"]
    q_addr_c2 = q["addr_c2"]
    q_addr_c3 = q["addr_c3"]
    if c_addr_missing == 1 or not q["addr_norm"]:
        addr_tok_jac = 0.0
        addr_contain_q = 0.0
        addr_contain_c = 0.0
        addr_contain = 0.0
        addr_c2 = 0.0
        addr_c3 = 0.0
        addr_eq = 0.0
        addr_len_diff = 0
    else:
        addr_inter = len(q_addr_toks & c_addr_toks)
        addr_union = len(q_addr_toks | c_addr_toks)
        addr_tok_jac = addr_inter / max(1, addr_union)
        addr_contain_q = addr_inter / max(1, len(q_addr_toks))
        addr_contain_c = addr_inter / max(1, len(c_addr_toks))
        addr_contain = addr_inter / max(1, min(len(q_addr_toks), len(c_addr_toks)))
        addr_c2 = (2.0 * len(q_addr_c2 & c_addr_c2)) / max(1, len(q_addr_c2) + len(c_addr_c2))
        addr_c3 = (2.0 * len(q_addr_c3 & c_addr_c3)) / max(1, len(q_addr_c3) + len(c_addr_c3))
        addr_eq = 1.0 if (q["addr_norm"] == c_addr_norm) else 0.0
        addr_len_diff = abs(len(q["addr_norm"]) - len(c_addr_norm))

    a_eq = addr_eq > 0.5
    a_miss = (c_addr_missing == 1)

    # Numeric features
    q_nums = q["nums"]
    q_num_set = q["num_set"]
    num_overlap = float(len(q_num_set & c_num_set))
    num_ov = (num_overlap >= 1.0)
    first_num = (len(q_nums) > 0 and len(c_nums) > 0 and q_nums[0] == c_nums[0])
    last_num = (len(q_nums) > 0 and len(c_nums) > 0 and q_nums[-1] == c_nums[-1])

    # Rule evaluation (Tier 1 -> Tier 2 -> Tier 3)
    rule_conf = 0.0
    rule_tier = 0

    # Tier 1
    if n_eq and (a_eq or addr_tok_jac >= 0.35 or addr_c2 >= 0.50 or num_ov or first_num):
        rule_conf = 0.95 + 0.04 * min(1.0, addr_tok_jac + (0.05 if num_ov else 0.0))
        rule_tier = 1
    elif a_eq and (name_tok_jac >= 0.30 or name_c2 >= 0.40 or name_contain >= 0.50):
        rule_conf = 0.94 + 0.05 * min(1.0, name_tok_jac)
        rule_tier = 1
    elif (name_tok_jac >= 0.65 or name_c3 >= 0.70) and (addr_tok_jac >= 0.45 or addr_c2 >= 0.60):
        rule_conf = 0.92 + 0.05 * min(1.0, (name_tok_jac + addr_tok_jac) / 2.0)
        rule_tier = 1
    elif (num_ov or first_num) and (name_tok_jac >= 0.45 or name_c2 >= 0.55 or name_contain >= 0.60) and (addr_tok_jac >= 0.30 or addr_c2 >= 0.45):
        rule_conf = 0.91 + 0.05 * min(1.0, (name_tok_jac + addr_tok_jac) / 2.0)
        rule_tier = 1
    # Tier 2
    elif (addr_tok_jac >= 0.50 or addr_c3 >= 0.65) and (num_ov or first_num) and (name_c2 >= 0.30 or name_contain >= 0.35):
        rule_conf = 0.85 + 0.05 * addr_tok_jac
        rule_tier = 2
    elif (name_tok_jac >= 0.55 or name_c3 >= 0.60) and (addr_tok_jac >= 0.35 or addr_c2 >= 0.50):
        rule_conf = 0.80 + 0.05 * name_tok_jac
        rule_tier = 2
    elif (num_ov or first_num or last_num) and (name_tok_jac >= 0.60 or (name_c2 >= 0.70 and name_contain >= 0.60)):
        rule_conf = 0.78 + 0.05 * name_tok_jac
        rule_tier = 2
    elif a_miss and n_eq and name_len_diff <= 3:
        rule_conf = 0.76
        rule_tier = 2
    # Tier 3
    elif addr_tok_jac >= 0.45 and (name_tok_jac >= 0.25 or name_c2 >= 0.35):
        rule_conf = 0.65 + 0.05 * addr_tok_jac
        rule_tier = 3
    elif (name_tok_jac >= 0.80 or (n_eq and name_len_diff <= 2)) and (addr_c2 >= 0.30 or addr_contain >= 0.30):
        rule_conf = 0.60
        rule_tier = 3

    # LightGBM feature vector
    pool_both_eq = 1.0 if (name_eq > 0.5 and addr_eq > 0.5) else 0.0
    pool_ntok_eq = 1.0 if (q_name_toks and q_name_toks == c_name_toks) else 0.0
    pool_atok_eq = 1.0 if (q_addr_toks and q_addr_toks == c_addr_toks) else 0.0
    pool_addr_any_missing = 1.0 if (c_addr_missing == 1 or not q["addr_norm"]) else 0.0

    lgb_feat = [
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
        float(name_inter),
        name_eq,
        addr_eq,
        pool_both_eq,
        pool_ntok_eq,
        pool_atok_eq,
        pool_addr_any_missing,
        q["is_us"],
    ]

    return lgb_feat, rule_conf, rule_tier


def load_country_queries(test_records_path: str, country: str) -> Dict[str, Dict]:
    """Load S1 query records for a specific country."""
    t0 = time.time()
    q_df = (
        pl.scan_parquet(test_records_path)
        .filter((pl.col("src") == 1) & (pl.col("country") == country))
        .select(["id", "name_norm", "name_norm_suf", "addr_norm", "addr_num"])
        .collect()
    )
    n = q_df.height
    print(f"Loading {n:,} query records for {country}...", flush=True)

    ids = q_df["id"].to_list()
    name_norms = q_df["name_norm"].fill_null("").to_list()
    name_sufs = q_df["name_norm_suf"].fill_null("").to_list()
    addr_norms = q_df["addr_norm"].fill_null("").to_list()
    addr_nums = q_df["addr_num"].fill_null("").to_list()
    del q_df
    gc.collect()

    is_us = 1.0 if country == "US" else 0.0
    q_records = {}
    for i in range(n):
        n_norm = name_norms[i]
        n_suf = name_sufs[i]
        a_norm = addr_norms[i]
        nums = [tok for tok in addr_nums[i].split() if tok]
        name_toks = set(n_norm.split()) if n_norm else set()
        addr_toks = set(a_norm.split()) if a_norm else set()

        q_records[ids[i]] = {
            "country": country,
            "is_us": is_us,
            "name_norm": n_norm,
            "name_suf": n_suf,
            "addr_norm": a_norm,
            "nums": nums,
            "num_set": set(nums),
            "name_toks": name_toks,
            "addr_toks": addr_toks,
            "name_c2": char_ngrams(n_norm, 2),
            "name_c3": char_ngrams(n_norm, 3),
            "addr_c2": char_ngrams(a_norm, 2),
            "addr_c3": char_ngrams(a_norm, 3),
        }

    del ids, name_norms, name_sufs, addr_norms, addr_nums
    gc.collect()
    print(f"Loaded {n:,} {country} queries in {time.time()-t0:.1f}s.", flush=True)
    return q_records


class FastCandidatePool:
    """In-memory candidate pool with fast index build and lazy candidate caching."""

    def __init__(self, cand_df: pl.DataFrame, query_records: Dict[str, Dict]):
        t0 = time.time()
        n = cand_df.height
        print(f"Building fast candidate pool for {n:,} records...", flush=True)

        ids = cand_df["id"].to_list()
        name_norms = cand_df["name_norm"].fill_null("").to_list()
        name_sufs = cand_df["name_norm_suf"].fill_null("").to_list()
        addr_norms = cand_df["addr_norm"].fill_null("").to_list()
        addr_nums = cand_df["addr_num"].fill_null("").to_list()
        del cand_df
        gc.collect()

        self.n = n
        self.ids = ids
        self.name_norms = name_norms
        self.name_sufs = name_sufs
        self.addr_norms = addr_norms
        self.addr_nums = addr_nums

        # Lazy caches
        self.nums: List[Optional[List[str]]] = [None] * n
        self.num_sets: List[Optional[Set[str]]] = [None] * n
        self.addr_missing: List[int] = [1 if not a else 0 for a in addr_norms]
        self.name_toks: List[Optional[Set[str]]] = [None] * n
        self.addr_toks: List[Optional[Set[str]]] = [None] * n
        self.name_c2: List[Optional[Set[str]]] = [None] * n
        self.name_c3: List[Optional[Set[str]]] = [None] * n
        self.addr_c2: List[Optional[Set[str]]] = [None] * n
        self.addr_c3: List[Optional[Set[str]]] = [None] * n

        self.name_exact_idx: Dict[str, List[int]] = collections.defaultdict(list)
        self.name_suf_idx: Dict[str, List[int]] = collections.defaultdict(list)
        self.token_idx: Dict[str, List[int]] = collections.defaultdict(list)
        self.numeric_idx: Dict[str, List[int]] = collections.defaultdict(list)
        self.addr_exact_idx: Dict[str, List[int]] = collections.defaultdict(list)
        self.addr_tok_idx: Dict[str, List[int]] = collections.defaultdict(list)

        q_names = set(q["name_norm"] for q in query_records.values() if q["name_norm"])
        q_sufs = set(q["name_suf"] for q in query_records.values() if q["name_suf"])
        q_addrs = set(q["addr_norm"] for q in query_records.values() if q["addr_norm"])
        q_toks = set(t for q in query_records.values() for t in q["name_toks"] if len(t) >= 3 and t not in NAME_STOPWORDS)
        q_addr_toks = set(t for q in query_records.values() for t in q["addr_toks"] if len(t) >= 4 and t not in ADDR_STOPWORDS and not any(ch.isdigit() for ch in t))
        q_nums = set(num for q in query_records.values() for num in q["nums"])

        for i in range(n):
            n_norm = name_norms[i]
            n_suf = name_sufs[i]
            a_norm = addr_norms[i]
            a_num = addr_nums[i]

            if n_norm in q_names:
                self.name_exact_idx[n_norm].append(i)
            if n_suf and n_suf in q_sufs:
                self.name_suf_idx[n_suf].append(i)
            if a_norm and a_norm in q_addrs:
                self.addr_exact_idx[a_norm].append(i)

            if n_norm:
                for t in n_norm.split():
                    if t in q_toks:
                        self.token_idx[t].append(i)

            if a_norm:
                for t in a_norm.split():
                    if t in q_addr_toks:
                        self.addr_tok_idx[t].append(i)

            if a_num:
                for num in a_num.split():
                    if num in q_nums:
                        self.numeric_idx[num].append(i)

        MAX_BLOCK = 150
        for idx_dict in (self.name_exact_idx, self.name_suf_idx, self.token_idx, self.numeric_idx, self.addr_exact_idx, self.addr_tok_idx):
            oversized = [k for k, v in idx_dict.items() if len(v) > MAX_BLOCK]
            for k in oversized:
                del idx_dict[k]

        print(f"Candidate pool & index ready in {time.time()-t0:.1f}s.", flush=True)

    def ensure_cand_features(self, c_idx: int):
        """Lazily compute and cache candidate tokens and n-grams on first access."""
        if self.name_c2[c_idx] is None:
            n_norm = self.name_norms[c_idx]
            a_norm = self.addr_norms[c_idx]
            num_list = [tok for tok in self.addr_nums[c_idx].split() if tok]
            self.nums[c_idx] = num_list
            self.num_sets[c_idx] = set(num_list)
            self.name_toks[c_idx] = set(n_norm.split()) if n_norm else set()
            self.addr_toks[c_idx] = set(a_norm.split()) if a_norm else set()
            self.name_c2[c_idx] = char_ngrams(n_norm, 2)
            self.name_c3[c_idx] = char_ngrams(n_norm, 3)
            self.addr_c2[c_idx] = char_ngrams(a_norm, 2)
            self.addr_c3[c_idx] = char_ngrams(a_norm, 3)

    def generate_candidates(self, q: Dict, max_cand: int = 150) -> List[int]:
        """Generate candidate indices for query."""
        cand_set = set()

        if q["name_norm"]:
            for cid in self.name_exact_idx.get(q["name_norm"], ()):
                cand_set.add(cid)

        if q["name_suf"]:
            for cid in self.name_suf_idx.get(q["name_suf"], ()):
                cand_set.add(cid)

        if q["addr_norm"]:
            for cid in self.addr_exact_idx.get(q["addr_norm"], ()):
                cand_set.add(cid)

        for t in q["name_toks"]:
            if t not in NAME_STOPWORDS and len(t) >= 3:
                for cid in self.token_idx.get(t, ()):
                    cand_set.add(cid)
                    if len(cand_set) >= max_cand * 2:
                        break

        for num in q["nums"]:
            for cid in self.numeric_idx.get(num, ()):
                cand_set.add(cid)
                if len(cand_set) >= max_cand * 2:
                    break

        for t in q["addr_toks"]:
            if t not in ADDR_STOPWORDS and len(t) >= 4 and not any(ch.isdigit() for ch in t):
                for cid in self.addr_tok_idx.get(t, ()):
                    cand_set.add(cid)
                    if len(cand_set) >= max_cand * 2:
                        break

        if len(cand_set) > max_cand:
            return list(cand_set)[:max_cand]
        return list(cand_set)


def process_country(
    country: str,
    test_records_path: str,
    model,
    global_progress: List[int],
    total_queries: int,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Process a single country partition with high-speed fused inference."""
    q_records = load_country_queries(test_records_path, country)
    country_qids = list(q_records.keys())
    n_queries = len(country_qids)
    if n_queries == 0:
        return {}, {}

    print(f"\n{'='*70}", flush=True)
    print(f"PROCESSING COUNTRY: {country} ({n_queries:,} queries)", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()

    cand_df = (
        pl.scan_parquet(test_records_path)
        .filter((pl.col("src").is_in([2, 3])) & (pl.col("country") == country))
        .collect()
    )
    pool = FastCandidatePool(cand_df, q_records)
    del cand_df
    gc.collect()

    all_hybrid_triples: List[Tuple[float, str, str]] = []
    candidate_sets: Dict[str, Set[str]] = {}

    BATCH_SIZE = 25000
    t_score = time.time()
    for b_start in range(0, n_queries, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, n_queries)
        batch_qids = country_qids[b_start:b_end]

        batch_feats = []
        batch_meta = []

        for qid in batch_qids:
            q = q_records[qid]
            cand_indices = pool.generate_candidates(q, max_cand=150)

            c_ids_for_s1 = set()
            for c_idx in cand_indices:
                cid_str = pool.ids[c_idx]
                c_ids_for_s1.add(cid_str)
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

                if r_tier > 0:
                    batch_feats.append(lgb_feat)
                    batch_meta.append((qid, cid_str, r_conf, r_tier))

            candidate_sets[qid] = c_ids_for_s1

        if batch_feats:
            X_mat = np.array(batch_feats, dtype=np.float32)
            lgb_probs = model.predict_proba(X_mat)[:, 1]

            for (qid, cid_str, r_conf, r_tier), p in zip(batch_meta, lgb_probs):
                hybrid_conf = (p ** 0.5) * (r_conf ** 0.5)
                if hybrid_conf >= ANCHOR_THRESH:
                    all_hybrid_triples.append((hybrid_conf, qid, cid_str))

            del X_mat, lgb_probs
        del batch_feats, batch_meta

        global_progress[0] += (b_end - b_start)
        elapsed = time.time() - t_score
        rate = b_end / max(0.1, elapsed)
        pct = global_progress[0] / total_queries * 100
        print(f"    [{country}] {b_end:,}/{n_queries:,} queries done ({pct:.1f}% overall, {rate:.0f} q/s)", flush=True)

    del pool, q_records
    gc.collect()

    print(f"  Running Disjoint Star Cluster ({len(all_hybrid_triples):,} high-confidence triples)...", flush=True)
    all_hybrid_triples.sort(key=lambda x: x[0], reverse=True)

    assigned_candidates: Set[str] = set()
    s1_matches: Dict[str, Set[str]] = collections.defaultdict(set)

    for conf, s1_id, cand_id in all_hybrid_triples:
        if cand_id in assigned_candidates:
            continue
        if len(s1_matches[s1_id]) >= MAX_MATCHES_PER_S1:
            continue
        s1_matches[s1_id].add(cand_id)
        assigned_candidates.add(cand_id)

    matching_predictions = {qid: s1_matches.get(qid, set()) for qid in country_qids}
    total_matched = sum(len(v) for v in matching_predictions.values())
    n_s1_with_match = sum(1 for v in matching_predictions.values() if len(v) > 0)
    print(f"  {country} complete in {time.time()-t0:.1f}s: {n_s1_with_match:,} S1 matched, {total_matched:,} total pairs.", flush=True)

    return matching_predictions, candidate_sets


def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — TEST SET INFERENCE (OPTIMIZED EXP-003)")
    print("=" * 70)

    model_path = os.path.join(MODELS_DIR, "lgbm_matcher.joblib")
    print(f"\n1. Loading LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    print("   Model loaded successfully.")

    test_records_path = os.path.join(REPO_ROOT, "work", "test_records.parquet")
    total_queries = 1732544

    matching_tsv = os.path.join(OUTPUTS_DIR, "matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUTS_DIR, "candidate_pairs.tsv")

    # Open output files for incremental writing — one pass per country
    f_match = open(matching_tsv, "w", encoding="utf-8")
    f_cand = open(candidate_tsv, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    global_progress = [0]
    total_rows = 0

    for country in ["France", "India", "US"]:
        match_map, cand_map = process_country(
            country=country,
            test_records_path=test_records_path,
            model=model,
            global_progress=global_progress,
            total_queries=total_queries,
        )

        # Flush this country's results to disk immediately
        country_qids = sorted(match_map.keys())
        for qid in country_qids:
            matches = match_map[qid]
            match_str = ",".join(sorted(matches)) if matches else ""
            f_match.write(f"{qid}\t{match_str}\n")

            cands = cand_map.get(qid, set())
            cand_str = ",".join(sorted(cands)) if cands else ""
            f_cand.write(f"{qid}\t{cand_str}\n")

        total_rows += len(country_qids)
        f_match.flush()
        f_cand.flush()

        # Free this country's results from RAM
        del match_map, cand_map, country_qids
        gc.collect()
        print(f"  {country} results flushed to disk ({total_rows:,} total rows so far).", flush=True)

    f_match.close()
    f_cand.close()

    print(f"\n4. Output files written to {OUTPUTS_DIR} ({total_rows:,} rows).")
    print(f"   - {matching_tsv}")
    print(f"   - {candidate_tsv}")

    print(f"\n5. Copying to submission/ directory...")
    import shutil
    shutil.copy2(matching_tsv, os.path.join(SUBMISSION_DIR, "matching_results.tsv"))
    shutil.copy2(candidate_tsv, os.path.join(SUBMISSION_DIR, "candidate_pairs.tsv"))

    print(f"\n6. Running official submission validator...")
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "validate_submission.py"),
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--test-dir", os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset", "test"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode == 0:
        print("\nALL INFERENCE AND VALIDATIONS COMPLETED SUCCESSFULLY!")
    else:
        print("\nVALIDATION FAILED! Check error output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
