"""11 - Match cardinality from ground truth (gt_long.parquet)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, GT_LONG

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def main():
    gt = pl.scan_parquet(GT_LONG)

    per_s1 = (gt.group_by("s1_id").agg([
        pl.len().alias("n_matches"),
        (pl.col("msrc") == "S2").sum().alias("n_s2"),
        (pl.col("msrc") == "S3").sum().alias("n_s3"),
    ]).collect())

    n_s1_total = per_s1.height
    out = {
        "n_s1_matched": int(n_s1_total),
        "n_matches_per_s1": {
            "mean": round(float(per_s1["n_matches"].mean()), 3),
            "median": int(per_s1["n_matches"].median()),
            "p90": int(per_s1["n_matches"].quantile(0.9)),
            "p99": int(per_s1["n_matches"].quantile(0.99)),
            "max": int(per_s1["n_matches"].max()),
            "n_1": int((per_s1["n_matches"] == 1).sum()),
            "n_2": int((per_s1["n_matches"] == 2).sum()),
            "n_3": int((per_s1["n_matches"] == 3).sum()),
            "n_4p": int((per_s1["n_matches"] >= 4).sum()),
        },
        "match_count_hist": {str(r["n_matches"]): int(r["c"]) for r in
                             per_s1.group_by("n_matches").agg(pl.len().alias("c"))
                             .sort("n_matches").iter_rows(named=True)},
        "s2_only": int(((per_s1["n_s2"] > 0) & (per_s1["n_s3"] == 0)).sum()),
        "s3_only": int(((per_s1["n_s3"] > 0) & (per_s1["n_s2"] == 0)).sum()),
        "both": int(((per_s1["n_s2"] > 0) & (per_s1["n_s3"] > 0)).sum()),
        "by_msrc": [dict(r) for r in gt.group_by("msrc").agg(pl.len().alias("n")).collect().iter_rows(named=True)],
    }

    # coverage: how many S2 / S3 records appear in GT (matched at least once)
    rec = pl.scan_parquet(os.path.join(WORK_DIR, "train_records.parquet"))
    src_counts = rec.group_by("src").agg(pl.len().alias("total")).collect()
    gt_src = (gt.group_by("msrc").agg(pl.col("matched_id").n_unique().alias("matched_unique")).collect())
    cov = {}
    agg_gt = gt_src.to_dict(as_series=False)
    gt_lookup = dict(zip(agg_gt["msrc"], agg_gt["matched_unique"]))
    for r in src_counts.iter_rows(named=True):
        s = r["src"]
        msrc = f"S{s}"
        m = int(gt_lookup.get(msrc, 0))
        cov[msrc] = {"total": int(r["total"]), "matched_unique": m,
                     "unmatched": int(r["total"] - m),
                     "unmatched_frac": round(float(r["total"] - m) / float(r["total"]), 5)}
    out["candidate_coverage"] = cov

    # by-country cardinality
    s1c = pl.scan_parquet(os.path.join(WORK_DIR, "train_records.parquet")).select(
        ["id", "country"]).collect().rename({"id": "s1_id"})
    j = per_s1.join(s1c, on="s1_id", how="left")
    out["by_country"] = {
        ctry: {
            "n_s1": int((j["country"] == ctry).sum()),
            "mean": round(float(j.filter(pl.col("country") == ctry)["n_matches"].mean()), 3),
            "median": int(j.filter(pl.col("country") == ctry)["n_matches"].median()),
        } for ctry in ["India", "US"]
    }

    with open(os.path.join(OUT, "11_cardinality.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()