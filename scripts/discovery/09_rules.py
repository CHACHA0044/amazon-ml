"""09 - Deterministic-rule coverage sanity vs official macro F0.5 on the pool.

For each rule, run it per S1 on the pool pairs (pos=known matches, neg=sampled
candidates), predict, and compute the macro (per-S1) F0.5 exactly like the
official metric over the covered S1.  Also report raw recall on true matches
and FP rate among negative candidates.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)
POOL = os.path.join(WORK_DIR, "pair_pool.parquet")


def rule_f05(pool, mask):
    """Per-S1 aggregate F0.5 from pool pairs (pred vs true labels)."""
    df = pool.select(["s1_id", "label", mask.alias("pred")]).rename({"label": "true"})
    agg = (df.with_columns([
        ((pl.col("pred") & pl.col("true")).cast(pl.Int32)).alias("tp"),
        ((pl.col("pred") & ~pl.col("true")).cast(pl.Int32)).alias("fp"),
        ((~pl.col("pred") & pl.col("true")).cast(pl.Int32)).alias("fn"),
    ]).group_by("s1_id").agg([
        pl.col("tp").sum().alias("tp"),
        pl.col("fp").sum().alias("fp"),
        pl.col("fn").sum().alias("fn"),
        pl.col("true").any().alias("has_true"),
    ]).with_columns([
        (pl.col("tp") + pl.col("fp")).alias("npred"),
    ]).filter((pl.col("npred") > 0) | pl.col("has_true")))
    # singletons (no pairs in pool): skip; covered-only F0.5
    f05s = []
    for r in agg.rows():
        _, tp, fp, fn, has_true, npred = r
        if not has_true and npred == 0:
            continue
        if tp + fp == 0 or tp + fn == 0:
            f05s.append(1.0 if has_true and tp == fn == fp == 0 else 0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f05s.append(1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0)
    return {
        "f05_macro_covered": round(sum(f05s) / len(f05s), 5) if f05s else None,
        "n_s1_covered": len(f05s),
        "n_s1_with_pred": int(agg.filter(pl.col("npred") > 0).height),
        "n_s1_with_true": int(agg.filter(pl.col("has_true")).height),
    }


def main():
    pool = pl.read_parquet(POOL)
    lbl = pl.col("label")

    rules = {
        "R1_name_norm_eq": pool["pool_name_eq"],
        "R2_name_eq_and_addr_norm_eq": pool["pool_both_eq"],
        "R3_name_eq_and_num_overlap": pool["pool_name_eq"] & (pool["num_overlap"] >= 1),
        "R4_name_tokjac_0.8": pool["name_tok_jac"] >= 0.8,
        "R5_name_tokjac_0.8_or_char2_0.9": (pool["name_tok_jac"] >= 0.8) | (pool["name_char2"] >= 0.9),
        "R6_name_0.6_and_addr_0.5": (pool["name_tok_jac"] >= 0.6) & (pool["addr_tok_jac"] >= 0.5),
        "R7_addr_0.4_and_num_overlap": (pool["addr_tok_jac"] >= 0.4) & (pool["num_overlap"] >= 1),
        "R8_addr_char2_0.8": pool["addr_char2"] >= 0.8,
        "R9_num_share_and_name_0.5": (pool["num_overlap"] >= 1) & (pool["name_tok_jac"] >= 0.5),
        "R10_any_strong": (pool["name_tok_jac"] >= 0.9) | (pool["addr_tok_jac"] >= 0.6)
                          | (pool["addr_char2"] >= 0.9) | (pool["num_overlap"] >= 1),
        "R11_rare_and_addr_0.3": (pool["rare_shared"] >= 1) & (pool["addr_tok_jac"] >= 0.3),
    }

    out = {}
    for name, mask in rules.items():
        sub = pool.select(["s1_id", "label", mask.alias("m")])
        recall = float(sub.filter(pl.col("label") == 1)["m"].mean())
        fp_count = int(sub.filter((pl.col("label") == 0) & pl.col("m")).height)
        neg_labelled = int(sub.filter(pl.col("label") == 0).height)
        can_overlap = sub.filter(sub["m"] & (sub["label"] == 0)).select("s1_id").n_unique()
        out[name] = {
            "pos_rate": round(recall, 5),
            "neg_candidate_rate": round(fp_count / neg_labelled, 5) if neg_labelled else None,
            "neg_hits": fp_count,
            "s1_with_neg_hit": int(can_overlap),
            "f05": rule_f05(sub, mask),
            "n": int(sub["m"].sum()),
        }
    with open(os.path.join(OUT, "09_rules.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()