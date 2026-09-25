"""02 - Corruption process: how S2/S3 records differ from their matched S1 record.

Vectorized polars-based taxonomy per positive (S1, S2|S3) pair:

  name_type / addr_type:
    exact          byte-identical copy
    case           differs only by letter case
    punct          differs only by punctuation/whitespace
    suffix         differs only by corporate suffix variants  (name only)
    abbr           differs only by address-abbreviation/punct map (addr only)
    order          same token set, different order
    subtoken       one side's tokens are a strict subset of the other
    heavy          all other cases (combination of edits)
    missing        empty address on one side (addr only)
    both_missing   empty address on both sides (addr only)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)

POS_PAIRS_FULL = os.path.join(WORK_DIR, "pos_pairs_full.parquet")
ABBR = "abbr"  # label for address norm-map-only differences


def split_set(s):
    return s.str.split(" ").list.unique()


def main():
    rec = pl.scan_parquet(TRAIN_RECORDS).select(
        ["src", "id", "name_norm_suf", "addr_norm", "addr_num"]
    )
    src = rec.filter(pl.col("src") == 1).select(
        pl.col("id").alias("s1_id"),
        pl.col("name_norm_suf").alias("q_name_norm_suf"),
        pl.col("addr_norm").alias("q_addr_norm"),
        pl.col("addr_num").alias("q_addr_num"),
    )
    cand = rec.filter(pl.col("src") > 1).select(
        pl.col("id"),
        pl.col("name_norm_suf").alias("c_name_norm_suf"),
        pl.col("addr_norm").alias("c_addr_norm"),
        pl.col("addr_num").alias("c_addr_num"),
    )

    df = (pl.scan_parquet(POS_PAIRS_FULL)
          .filter((pl.col("s1_id").hash(0) % 100) < 13)
          .join(src, on="s1_id", how="left")
          .join(cand, left_on="matched_id", right_on="id", how="left")
          .with_columns([
              # ---- token-set metrics (names) ----
              split_set(pl.col("q_name_norm_suf")).alias("q_ntoks"),
              split_set(pl.col("c_name_norm_suf")).alias("c_ntoks"),
              split_set(pl.col("q_addr_norm")).alias("q_atoks"),
              split_set(pl.col("c_addr_norm")).alias("c_atoks"),
              split_set(pl.col("q_addr_num")).alias("q_ntok"),
              split_set(pl.col("c_addr_num")).alias("c_ntok"),
          ])
          .with_columns([
              pl.col("q_ntoks").list.set_intersection("c_ntoks").list.len().alias("name_inter"),
              pl.col("q_ntoks").list.set_union("c_ntoks").list.len().alias("name_union"),
              pl.col("q_ntoks").list.len().alias("q_ntok_n"),
              pl.col("c_ntoks").list.len().alias("c_ntok_n"),
              pl.col("q_atoks").list.set_intersection("c_atoks").list.len().alias("addr_inter"),
              pl.col("q_atoks").list.set_union("c_atoks").list.len().alias("addr_union"),
              pl.col("q_atoks").list.len().alias("q_atok_n"),
              pl.col("c_atoks").list.len().alias("c_atok_n"),
              pl.col("q_ntok").list.set_intersection("c_ntok").list.len().alias("num_overlap"),
          ])
          .with_columns([
              pl.when(pl.col("name_union") > 0)
                .then(pl.col("name_inter") / pl.col("name_union")).otherwise(1.0).alias("name_tok_jac"),
              pl.when((pl.col("q_ntok_n") > 0) & (pl.col("c_ntok_n") > 0))
                .then(pl.col("name_inter") / pl.min_horizontal("q_ntok_n", "c_ntok_n")).otherwise(0.0).alias("name_contain"),
              pl.when(pl.col("addr_union") > 0)
                .then(pl.col("addr_inter") / pl.col("addr_union")).otherwise(1.0).alias("addr_tok_jac"),
              pl.when((pl.col("q_atok_n") > 0) & (pl.col("c_atok_n") > 0))
                .then(pl.col("addr_inter") / pl.min_horizontal("q_atok_n", "c_atok_n")).otherwise(0.0).alias("addr_contain"),
          ])
          .with_columns([
              # ---- name type ----
              pl.when(pl.col("name_raw_eq")).then(pl.lit("exact"))
                .when(pl.col("name_lower_eq")).then(pl.lit("case"))
                .when(pl.col("name_norm_eq")).then(pl.lit("punct"))
                .when(pl.col("name_norm_suf_eq")).then(pl.lit("suffix"))
                .when(pl.col("name_tok_jac") >= 1.0).then(pl.lit("order"))
                .when((pl.col("name_tok_jac") < 1.0)
                      & (pl.col("name_inter") == pl.min_horizontal("q_ntok_n", "c_ntok_n"))
                      & (pl.min_horizontal("q_ntok_n", "c_ntok_n") > 0)).then(pl.lit("subtoken"))
                .otherwise(pl.lit("heavy")).alias("name_type"),
              # ---- addr type ----
              pl.when(pl.col("addr_missing_both")).then(pl.lit("both_missing"))
                .when(pl.col("addr_missing_any")).then(pl.lit("missing"))
                .when(pl.col("addr_raw_eq")).then(pl.lit("exact"))
                .when(pl.col("addr_lower_eq")).then(pl.lit("case"))
                .when(pl.col("addr_norm_eq")).then(pl.lit(ABBR))
                .when(pl.col("addr_tok_jac") >= 1.0).then(pl.lit("order"))
                .when((pl.col("addr_tok_jac") < 1.0)
                      & (pl.col("addr_inter") == pl.min_horizontal("q_atok_n", "c_atok_n"))
                      & (pl.min_horizontal("q_atok_n", "c_atok_n") > 0)).then(pl.lit("subtoken"))
                .otherwise(pl.lit("heavy")).alias("addr_type"),
          ])
          .collect())

    print(f"pos pairs with strings: {df.height:,}", flush=True)

    def table(col, extra=None):
        t = (df.group_by([col] + (extra or []))
             .agg(pl.len().alias("n"),
                  pl.col("name_tok_jac").mean().alias("name_tok_jac_mean"),
                  pl.col("addr_tok_jac").mean().alias("addr_tok_jac_mean"),
                  pl.col("num_overlap").mean().alias("num_overlap_mean"))
             .sort("n", descending=True))
        rows = [dict(r) for r in t.iter_rows(named=True)]
        if not extra:
            tot = sum(r["n"] for r in rows)
            for r in rows:
                r["frac"] = round(r["n"] / tot, 5)
        return rows

    result = {
        "n_pairs": int(df.height),
        "name_type": table("name_type"),
        "addr_type": table("addr_type"),
        "name_type_by_country": table("name_type", ["q_country"]),
        "addr_type_by_country": table("addr_type", ["q_country"]),
        "name_type_by_msrc": table("name_type", ["msrc"]),
        "addr_type_by_msrc": table("addr_type", ["msrc"]),
        "name_x_addr": table("name_type", ["addr_type"]),
        "both_exact": int(((df["name_type"] == "exact") & (df["addr_type"] == "exact")).sum()),
        "name_exact_addr_diff": int(((df["name_type"] == "exact")
                                     & (df["addr_type"].is_in(["order", "subtoken", "heavy"]))).sum()),
        "addr_exact_name_diff": int(((df["addr_type"] == "exact")
                                     & (df["name_type"].is_in(["order", "subtoken", "heavy"]))).sum()),
    }
    with open(os.path.join(OUT, "02_corruption.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print("\nDONE")


if __name__ == "__main__":
    main()