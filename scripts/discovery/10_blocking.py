"""10 - Blocking-signal analysis: candidate collision sizes for candidate generators."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS, TEST_RECORDS

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def block_stats(df, key, country_restrict=True):
    inner = df
    g = (inner.group_by([key, "country"] if country_restrict else [key])
         .agg(pl.len().alias("n")))
    sizes = g["n"].to_numpy()
    out_raw = {
        "n_groups": int(g.height),
        "singleton_groups": int((g["n"] == 1).sum()),
        "group_mean": round(float(sizes.mean()), 3),
        "group_p50": int(np.median(sizes)),
        "group_p90": int(np.quantile(sizes, 0.90)),
        "group_p99": int(np.quantile(sizes, 0.99)),
        "group_max": int(sizes.max()),
        "n_groups_gt_10": int((g["n"] > 10).sum()),
        "n_groups_gt_100": int((g["n"] > 100).sum()),
        "n_groups_gt_1000": int((g["n"] > 1000).sum()),
        "max_block": str(g.sort("n", descending=True).select(key).row(0)[0]) if g.height else None,
    }
    return out_raw


def main():
    rec = pl.scan_parquet(TRAIN_RECORDS).select(
        ["src", "country", "name_norm_suf", "addr_norm", "addr_num"])
    recdf = rec.collect()

    out = {"train": {}, "test": {}}

    def add_blocks(prefix, df):
        out[prefix] = {
            "by_name": block_stats(df, "name_norm_suf"),
            "by_addr": block_stats(df, "addr_norm"),
        }

    add_blocks("train", recdf)

    # by country
    for ctry in ["India", "US"]:
        sub = recdf.filter(pl.col("country") == ctry)
        out["train"][f"by_name_{ctry}"] = block_stats(sub, "name_norm_suf", country_restrict=False)
        out["train"][f"by_addr_{ctry}"] = block_stats(sub, "addr_norm", country_restrict=False)

    # numeric-token block: records sharing any digit token (sample risk of digits
    # being too common, e.g. India "110001" style)
    numgrp = (recdf.filter(pl.col("addr_num") != "")
              .with_columns(pl.col("addr_num").str.split(" ").alias("dtoks"))
              .explode("dtoks").group_by(["dtoks", "country"])
              .agg(pl.len().alias("n")))
    sizes = numgrp["n"].to_numpy()
    out["train"]["by_any_digit_token"] = {
        "n_groups": int(numgrp.height),
        "singleton_groups": int((numgrp["n"] == 1).sum()),
        "group_mean": round(float(sizes.mean()), 3),
        "group_p90": int(np.quantile(sizes, 0.90)),
        "group_p99": int(np.quantile(sizes, 0.99)),
        "group_max": int(sizes.max()),
        "max_block": str(numgrp.sort("n", descending=True).select("dtoks").row(0)[0]),
    }

    # test
    tedf = pl.scan_parquet(TEST_RECORDS).select(
        ["src", "country", "name_norm_suf", "addr_norm"]).collect()
    add_blocks("test", tedf)

    with open(os.path.join(OUT, "10_blocking.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()