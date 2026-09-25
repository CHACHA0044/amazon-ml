"""SageMaker Entrypoint: Full Test-Set Inference for Amazon ML Challenge 2026.

Executes EXP-003 Calibrated Hybrid LightGBM Matcher + Disjoint Star Cluster.
Runs inside SageMaker Processing Job on a high-memory CPU instance.

Outputs:
  - matching_results.tsv  (Official leaderboard submission file)
  - candidate_pairs.tsv   (Candidate blocking audit file)
  - run_summary.json      (Execution metrics, counts, timing)
"""
import collections
import gc
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Step 1: Install required packages inside SageMaker container if missing
def ensure_packages():
    required = ["polars", "lightgbm", "pyarrow", "joblib", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("Packages installed successfully.", flush=True)

ensure_packages()

import joblib
import numpy as np
import polars as pl

# Problem & Model Constants
ANCHOR_THRESH = 0.85
MAX_MATCHES_PER_S1 = 11

NAME_STOPWORDS = {
    "ltd", "pvt", "inc", "llc", "corp", "co", "limited", "private", "corporation",
    "company", "services", "solutions", "technologies", "technology", "enterprises",
    "associates", "group", "partners", "international", "global", "trading", "brothers",
    "ventures", "industries", "consultancy", "consultants", "care", "and", "the", "of",
    "in", "for", "center", "centre", "hub", "studio", "sarl", "sas", "llp", "pllc"
}

ADDR_STOPWORDS = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "apartment", "suite",
    "highway", "nagar", "marg", "allee", "rue", "place", "st", "rd", "ave", "blvd",
    "dr", "ln", "apt", "ste", "hwy", "ngr", "mrg", "all", "r", "pl", "floor", "unit",
    "block", "near", "opp", "opposite", "behind", "dist", "district", "city", "state",
    "india", "usa", "france", "north", "south", "east", "west"
}


def char_ngrams(s: str, n: int) -> Set[str]:
    """Generate set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


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
    """Fused feature extraction and rule evaluation in a single fast pass."""
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

    # LightGBM 19-feature vector
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


def process_country_streaming(
    country: str,
    test_records_path: str,
    model,
    global_progress: List[int],
    total_queries: int,
    f_cand,
) -> Dict[str, Set[str]]:
    """Process a single country partition with streaming candidate export and in-memory matching."""
    q_records = load_country_queries(test_records_path, country)
    country_qids = list(q_records.keys())
    n_queries = len(country_qids)
    if n_queries == 0:
        return {}

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

            c_ids_for_s1 = []
            for c_idx in cand_indices:
                cid_str = pool.ids[c_idx]
                c_ids_for_s1.append(cid_str)
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

            # Write candidates immediately to avoid multi-gigabyte memory accumulation
            cand_str = ",".join(sorted(set(c_ids_for_s1))) if c_ids_for_s1 else ""
            f_cand.write(f"{qid}\t{cand_str}\n")

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

    f_cand.flush()
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

    del all_hybrid_triples, assigned_candidates
    gc.collect()

    matching_predictions = {qid: s1_matches.get(qid, set()) for qid in country_qids}
    total_matched = sum(len(v) for v in matching_predictions.values())
    n_s1_with_match = sum(1 for v in matching_predictions.values() if len(v) > 0)
    print(f"  {country} complete in {time.time()-t0:.1f}s: {n_s1_with_match:,} S1 matched, {total_matched:,} total pairs.", flush=True)

    return matching_predictions


def main():
    t_global_start = time.time()
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — SAGEMAKER TEST INFERENCE (EXP-003)")
    print("=" * 70)

    # Determine paths (SageMaker Processing standard paths vs local/fallback)
    base_input_dir = "/opt/ml/processing/input"
    base_output_dir = "/opt/ml/processing/output"

    if os.path.exists(base_input_dir):
        # Running in SageMaker container
        model_path = os.path.join(base_input_dir, "models", "lgbm_matcher.joblib")
        test_records_path = os.path.join(base_input_dir, "preprocessed", "test_records.parquet")
        output_dir = base_output_dir
    else:
        # Fallback / standalone mode
        model_path = os.path.join("models", "lgbm_matcher.joblib")
        test_records_path = os.path.join("work", "test_records.parquet")
        output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Model path:        {model_path}")
    print(f"Test records path: {test_records_path}")
    print(f"Output directory:  {output_dir}")

    print(f"\n1. Loading LightGBM model from {model_path}...")
    model = joblib.load(model_path)
    print("   Model loaded successfully.")

    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    # Open candidate TSV for streaming
    f_cand = open(candidate_tsv, "w", encoding="utf-8")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_queries = 1732544
    global_progress = [0]
    all_matching: Dict[str, Set[str]] = {}

    for country in ["France", "India", "US"]:
        match_map = process_country_streaming(
            country=country,
            test_records_path=test_records_path,
            model=model,
            global_progress=global_progress,
            total_queries=total_queries,
            f_cand=f_cand,
        )
        all_matching.update(match_map)
        del match_map
        gc.collect()

    f_cand.close()
    print(f"\n2. candidate_pairs.tsv written ({candidate_tsv}).", flush=True)

    print(f"\n3. Writing matching_results.tsv ({matching_tsv})...", flush=True)
    with open(matching_tsv, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in sorted(all_matching.keys()):
            matches = all_matching[qid]
            match_str = ",".join(sorted(matches)) if matches else ""
            f_match.write(f"{qid}\t{match_str}\n")

    total_matched_pairs = sum(len(v) for v in all_matching.values())
    n_s1_with_match = sum(1 for v in all_matching.values() if len(v) > 0)
    total_s1 = len(all_matching)

    elapsed_total = time.time() - t_global_start
    print(f"\n4. Inference completed in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min).")
    print(f"   Total S1 queries processed: {total_s1:,}")
    print(f"   S1 with predicted matches:  {n_s1_with_match:,} ({n_s1_with_match/total_s1*100:.1f}%)")
    print(f"   Total matched pairs:        {total_matched_pairs:,}")

    # Write summary metadata
    summary = {
        "experiment_id": "EXP003-AWS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed_total, 1),
        "total_s1_queries": total_s1,
        "s1_with_matches": n_s1_with_match,
        "total_matched_pairs": total_matched_pairs,
        "anchor_threshold": ANCHOR_THRESH,
        "max_matches_per_s1": MAX_MATCHES_PER_S1,
        "status": "COMPLETED",
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   Summary written to {summary_path}")

    print("\nALL INFERENCE PHASES COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
