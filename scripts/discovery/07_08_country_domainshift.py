"""07/08 - Country-specific signals on the pool + train/test domain shift incl. France."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS, TEST_RECORDS
from pairfeat import auc_score

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)
POOL = os.path.join(WORK_DIR, "pair_pool.parquet")

FEATS = ["name_tok_jac", "name_char2", "name_char3", "addr_tok_jac",
         "addr_char2", "addr_char3", "num_overlap", "rare_shared"]


def main():
    pool = pl.read_parquet(POOL)
    # ---- 07 country-specific pooling behaviour ----
    country = {}
    for ctry in ["India", "US"]:
        p = pool.filter(pl.col("country") == ctry)
        pos, neg = p.filter(pl.col("label") == 1), p.filter(pl.col("label") == 0)
        if pos.height == 0 or neg.height == 0:
            continue
        entry = {"n_pos": int(pos.height), "n_neg": int(neg.height)}
        for f in FEATS:
            entry[f] = {
                "pos_mean": round(float(pos[f].mean()), 5),
                "neg_mean": round(float(neg[f].mean()), 5),
                "auc": round(float(auc_score(pos[f].to_numpy(), neg[f].to_numpy())), 5),
            }
        country[ctry] = entry
    result = {"07_pool_by_country": country}

    # block-growth rate per country: how many candidates share a name block
    rec = pl.scan_parquet(TRAIN_RECORDS).select(["src", "country", "name_norm_suf"]).collect()
    block_growth = {}
    for ctry in ["India", "US"]:
        sub = rec.filter((pl.col("country") == ctry) & (pl.col("src") == 1))
        cand = rec.filter((pl.col("country") == ctry) & (pl.col("src") > 1))
        cnt = (sub.group_by("name_norm_suf").agg(pl.len().alias("n")))
        cand_cnt = (cand.group_by("name_norm_suf").agg(pl.len().alias("m")))
        j = cnt.join(cand_cnt, on="name_norm_suf", how="left").filter(pl.col("m").is_not_null())
        sizes = j["m"].to_list()
        if sizes:
            block_growth[ctry] = {
                "n_s1_with_block_partners": int(j.height),
                "partner_median": int(np.median(sizes)),
                "partner_mean": round(float(np.mean(sizes)), 3),
                "partner_p90": int(np.quantile(sizes, 0.9)),
                "partner_p99": int(np.quantile(sizes, 0.99)),
                "partner_max": int(max(sizes)),
            }
        else:
            block_growth[ctry] = {"n_s1_with_block_partners": 0}
    result["07_block_growth_by_country"] = block_growth

    # ---- 08 train vs test domain shift ----
    def vocab(df_path):
        return (pl.scan_parquet(df_path)
                .select(pl.col("name_norm_suf").str.split(" ").alias("toks"))
                .explode("toks").filter(pl.col("toks") != "")
                .select("toks").unique().collect()["toks"].to_list())

    tr_toks = set(vocab(TRAIN_RECORDS))
    te_toks = vocab(TEST_RECORDS)
    te_set = set(te_toks)
    overlap = len(te_set & tr_toks)
    result["08_name_token_overlap"] = {
        "train_unique": len(tr_toks),
        "test_unique": len(te_set),
        "test_seen_in_train": overlap,
        "test_seen_frac": round(overlap / len(te_set), 5) if te_set else None,
    }

    def blocks(df_path):
        return pl.scan_parquet(df_path).select("name_norm_suf").unique().collect()["name_norm_suf"].to_list()

    tr_blk = set(blocks(TRAIN_RECORDS))
    te_blk = blocks(TEST_RECORDS)
    ov_blk = len(set(te_blk) & tr_blk)
    result["08_name_block_overlap"] = {
        "train_unique": len(tr_blk), "test_unique": len(set(te_blk)),
        "test_seen_in_train": ov_blk,
        "test_seen_frac": round(ov_blk / len(set(te_blk)), 5) if te_blk else None,
    }

    # country distribution in test
    test_country = (pl.scan_parquet(TEST_RECORDS)
                    .group_by(["src", "country"]).agg(pl.len().alias("n")).collect())
    result["08_test_country_counts"] = [dict(r) for r in test_country.iter_rows(named=True)]

    # France accent signal: share of non-ASCII chars in name raw? use name_norm
    fr = (pl.scan_parquet(TEST_RECORDS)
          .filter(pl.col("country") == "France")
          .group_by("src").agg(pl.len().alias("n")).collect())
    result["08_france_by_src"] = [dict(r) for r in fr.iter_rows(named=True)]

    with open(os.path.join(OUT, "07_08_country_domainshift.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()