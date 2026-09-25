"""Backfill pair_pool.parquet with correctly computed rare_shared (workers lacked vocab).
Overwrites the rare_shared column in place; also recompute num_overlap is already fine."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS
from pairfeat import name_tokens


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
    tr = pl.scan_parquet(TRAIN_RECORDS)
    df = (tr.select(pl.col("name_norm_suf").str.split(" "))
          .explode("name_norm_suf")
          .filter(pl.col("name_norm_suf") != "")
          .group_by("name_norm_suf").agg(pl.len().alias("df"))
          .filter((pl.col("df") >= 3) & (pl.col("df") <= 4000)).collect())
    rare_vocab = df["name_norm_suf"].to_list()
    print(f"rare vocab: {len(rare_vocab):,}")
    pool = pl.read_parquet(os.path.join(WORK_DIR, "pair_pool.parquet"))
    vals = recompute_rare_shared(pool, rare_vocab)
    pool = pool.with_columns(pl.Series(vals).alias("rare_shared"))
    pool.write_parquet(os.path.join(WORK_DIR, "pair_pool.parquet"))
    print("pool rare_shared backfilled")
    print(f"pos mean rare_shared: {pool.filter(pl.col('label')==1)['rare_shared'].mean():.4f}")
    print(f"neg mean rare_shared: {pool.filter(pl.col('label')==0)['rare_shared'].mean():.4f}")


if __name__ == "__main__":
    main()