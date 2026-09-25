"""Recompute 01 summary from saved pair_pool.parquet with fixed AUC + real rare_shared."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS
from pairfeat import auc_score, name_tokens

POOL_PATH = os.path.join(WORK_DIR, "pair_pool.parquet")
OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def recompute_rare_shared(pool, rare_vocab):
    vocab = set(rare_vocab)
    q = pool["q_name_norm_suf"].to_list()
    c = pool["c_name_norm_suf"].to_list()
    vals = []
    for a, b in zip(q, c):
        ta = set(name_tokens(a)) & vocab
        if not ta:
            vals.append(0)
        else:
            tb = set(name_tokens(b)) & vocab
            vals.append(len(ta & tb))
    return vals


def main():
    pool = pl.read_parquet(POOL_PATH)
    # recompute rare vocab consistently
    tr = pl.scan_parquet(TRAIN_RECORDS)
    df = (tr.select(pl.col("name_norm_suf").str.split(" "))
          .explode("name_norm_suf")
          .filter(pl.col("name_norm_suf") != "")
          .group_by("name_norm_suf").agg(pl.len().alias("df"))
          .filter((pl.col("df") >= 3) & (pl.col("df") <= 4000)).collect())
    rare_vocab = df["name_norm_suf"].to_list()
    print(f"rare vocab recomputed: {len(rare_vocab):,}")

    pool = pool.with_columns(pl.Series(recompute_rare_shared(pool, rare_vocab)).alias("rare_shared"))

    features = [
        "pool_name_eq", "pool_addr_eq", "pool_both_eq", "pool_ntok_eq", "pool_atok_eq",
        "name_tok_jac", "name_contain_q", "name_contain_c",
        "addr_tok_jac", "addr_contain_q", "addr_contain_c",
        "name_char2", "name_char3", "addr_char2", "addr_char3",
        "num_overlap", "rare_shared",
    ]
    pos = pool.filter(pl.col("label") == 1)
    neg = pool.filter(pl.col("label") == 0)
    summary = {
        "n_pos": int(pos.height),
        "n_neg": int(neg.height),
        "channels": {str(r["channel"]): int(r["n"]) for r in
                     pool.group_by("channel").agg(pl.len().alias("n")).sort("channel").iter_rows(named=True)},
    }
    for f in features:
        pv = pos[f].to_numpy().astype(np.float64)
        nv = neg[f].to_numpy().astype(np.float64)
        summary[f] = {
            "pos_mean": round(float(pv.mean()), 5),
            "neg_mean": round(float(nv.mean()), 5),
            "pos_q": {q: round(float(np.quantile(pv, q)), 5) for q in (0.5, 0.9, 0.99)},
            "neg_q": {q: round(float(np.quantile(nv, q)), 5) for q in (0.5, 0.9, 0.99)},
            "auc": round(float(auc_score(pv, nv)), 5),
        }
    probes = {
        "pool_name_eq": pool["pool_name_eq"],
        "pool_name_eq & pool_addr_eq": (pool["pool_name_eq"] & pool["pool_addr_eq"]),
        "name_tok_jac>=0.5": pool["name_tok_jac"] >= 0.5,
        "name_char2>=0.8": pool["name_char2"] >= 0.8,
        "addr_char2>=0.8": pool["addr_char2"] >= 0.8,
        "num_overlap>=1": pool["num_overlap"] >= 1,
        "pool_name_eq & addr_char2>=0.5": (pool["pool_name_eq"] & (pool["addr_char2"] >= 0.5)),
        "pool_both_eq": pool["pool_both_eq"],
        "rare_shared>=1": pool["rare_shared"] >= 1,
        "addr_char3>=0.8 & num_overlap>=1": (pool["addr_char3"] >= 0.8) & (pool["num_overlap"] >= 1),
    }
    summary["probes"] = {}
    for name, m in probes.items():
        sub = pool.filter(m)
        summary["probes"][name] = {
            "n": int(sub.height),
            "pos_frac": round(float(sub["label"].mean()), 5),
            "neg_frac": round(1.0 - float(sub["label"].mean()), 5),
        } if sub.height else {"n": 0}

    with open(os.path.join(OUT, "01_pair_sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()