"""05 - Address decomposition over raw-ish normalized fields.

Quantify per-component match evidence using numeric-token positions:
  first numeric token  -> house / plot number
  last numeric token   -> zip (US, 5-digit) / pincode (India, 6-digit) / postal code
Also report record-level address structure per country: whether an address
starts with a house number, contains a trailing postal token of expected length,
and how often matched pairs share house number or postal code.
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

ABBR = "abbr"


def _clean(lst):
    vals = lst.to_list() if isinstance(lst, pl.Series) else list(lst)
    return [] if vals == [""] else vals


def main():
    rec = pl.scan_parquet(TRAIN_RECORDS).select(
        ["src", "id", "country", "addr_norm", "addr_num"])

    df = (rec.select(
        ["src", "id", "country", "addr_norm",
         pl.col("addr_num").str.split(" ").map_elements(_clean, return_dtype=pl.List(pl.String)).alias("ntoks")]
    )
    .with_columns([
        pl.col("ntoks").list.first().alias("first_num"),
        pl.col("ntoks").list.last().alias("last_num"),
        pl.col("ntoks").list.len().alias("n_num_toks"),
        pl.col("addr_norm").str.len_chars().alias("alen"),
    ]).collect())

    def _len(s):
        return len(s) if s else 0

    df = df.with_columns([
        pl.col("last_num").map_elements(_len, return_dtype=pl.Int32).alias("last_num_len"),
        pl.col("first_num").is_not_null().alias("starts_house_no"),
    ])

    recstats = {}
    for country in ["India", "US"]:
        c = df.filter(pl.col("country") == country)
        recstats[country] = {
            "n_records": int(c.height),
            "addr_missing": round(float((c["addr_norm"] == "").mean()), 5),
            "starts_house_no": round(float(c["starts_house_no"].mean()), 5),
            "zip5_present": round(float((c["last_num_len"] == 5).mean()), 5),
            "pin6_present": round(float((c["last_num_len"] == 6).mean()), 5),
            "addr_with_digits": round(float((c["n_num_toks"] > 0).mean()), 5),
            "addr_median_tokens": int(c["addr_norm"].str.split(" ").list.len().median()),
        }
    result = {"record_stats": recstats}

    # pair-level component agreement (sample positives via hash)
    rec2 = rec.select(["id", "src", "addr_num"]).with_columns([
        pl.col("addr_num").str.split(" ").map_elements(_clean, return_dtype=pl.List(pl.String)).alias("ntoks"),
    ])
    src = rec2.filter(pl.col("src") == 1).select(
        pl.col("id").alias("s1_id"), pl.col("ntoks").alias("q_ntoks"))
    cand = rec2.filter(pl.col("src") > 1).select(
        pl.col("id"), pl.col("ntoks").alias("c_ntoks"))

    pair = (pl.scan_parquet(POS_PAIRS_FULL)
            .filter((pl.col("s1_id").hash(0) % 100) < 13)
            .join(src, on="s1_id", how="left")
            .join(cand, left_on="matched_id", right_on="id", how="left")
            .with_columns([
                pl.col("q_ntoks").list.first().alias("q_first"),
                pl.col("q_ntoks").list.last().alias("q_last"),
                pl.col("c_ntoks").list.first().alias("c_first"),
                pl.col("c_ntoks").list.last().alias("c_last"),
                pl.col("q_ntoks").list.len().alias("q_n"),
                pl.col("c_ntoks").list.len().alias("c_n"),
            ])
            .with_columns([
                (pl.col("q_first") == pl.col("c_first")).fill_null(False).alias("house_no_eq"),
                (pl.col("q_last") == pl.col("c_last")).fill_null(False).alias("postal_eq"),
                ((pl.col("q_n") > 0) & (pl.col("c_n") > 0)).alias("both_have_nums"),
            ])
            .collect())

    pairstats = {
        "n_pairs": int(pair.height),
        "house_no_present_both": round(float(pair["both_have_nums"].mean()), 5),
        "house_no_eq_given_both": round(float(pair["house_no_eq"].mean()), 5),
        "postal_eq_given_both": round(float(pair["postal_eq"].mean()), 5),
        "house_no_eq_given_present": round(float(pair["house_no_eq"].mean()), 5),
    }
    for country in ["India", "US"]:
        c = pair.filter(pl.col("q_country") == country)
        pairstats[country] = {
            "n_pairs": int(c.height),
            "both_have_nums": round(float(c["both_have_nums"].mean()), 5),
            "house_no_eq": round(float(c["house_no_eq"].mean()), 5),
            "postal_eq": round(float(c["postal_eq"].mean()), 5),
            "addr_missing_any": round(float(c["addr_missing_any"].mean()), 5),
        }
    result["pair_component_agreement"] = pairstats
    with open(os.path.join(OUT, "05_address.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()