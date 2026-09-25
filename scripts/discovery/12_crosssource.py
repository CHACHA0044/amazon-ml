"""12 - Cross-source corroboration: when an S1 matches both S2 and S3, how often
do the S2 and S3 records agree with each other (name / address)?"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, GT_LONG, TRAIN_RECORDS

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def main():
    gt = (pl.scan_parquet(GT_LONG)
          .filter(pl.col("s1_id").hash(0) % 100 < 25).collect())
    rec = pl.scan_parquet(TRAIN_RECORDS).select(
        ["id", "name_norm_suf", "addr_norm", "country"]).collect()

    s2gt = gt.filter(pl.col("msrc") == "S2").select(["s1_id", "matched_id"])
    s3gt = gt.filter(pl.col("msrc") == "S3").select(["s1_id", "matched_id"])
    both = s2gt.join(s3gt, on="s1_id", how="inner")
    print(f"S1 with both S2 and S3 matches: {both.height:,}")

    a = both.join(rec, left_on="matched_id", right_on="id", how="left") \
        .rename({"name_norm_suf": "name2", "addr_norm": "addr2", "country": "country2"})
    b = a.join(rec.rename({
        "name_norm_suf": "name3", "addr_norm": "addr3"}),
        left_on="matched_id_right", right_on="id", how="left")
    b = b.rename({"matched_id_right": "s3_id"})
    # recompute pair-wise: cross products per s1 (S2 vs S3)
    x = a.select(["s1_id", "matched_id", "name2", "addr2"]).rename({"matched_id": "s2_id"})
    y = s3gt.select(["s1_id", "matched_id"]).rename({"matched_id": "s3_id"})
    cross = x.join(y, on="s1_id", how="inner").join(
        rec.rename({"name_norm_suf": "name3", "addr_norm": "addr3"}),
        left_on="s3_id", right_on="id", how="left")
    cross = cross.with_columns([
        (pl.col("name2") == pl.col("name3")).alias("name_agree"),
        ((pl.col("name2") == pl.col("name3")) & (pl.col("addr2") == pl.col("addr3"))).alias("full_agree"),
    ])
    out = {
        "n_s1_both": int(both.height),
        "n_cross_pairs": int(cross.height),
        "cross_pairs_per_s1_mean": round(float(cross.height / max(both.height, 1)), 3),
        "name_agree_rate": round(float(cross["name_agree"].mean()), 5),
        "name_agree_per_s1": round(float(cross.group_by("s1_id").agg(pl.col("name_agree").any())
                                          ["name_agree"].mean()), 5),
        "full_agree_rate": round(float(cross["full_agree"].mean()), 5),
    }
    with open(os.path.join(OUT, "12_crosssource.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()