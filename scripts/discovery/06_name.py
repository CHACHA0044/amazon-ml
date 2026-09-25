"""06 - Name structure: token/suffix patterns per country (S1 records only)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def main():
    s1 = (pl.scan_parquet(TRAIN_RECORDS)
          .filter(pl.col("src") == 1)
          .select(["country", "name_norm_suf"]))

    name_rows = (s1.with_columns([
        pl.col("name_norm_suf").str.split(" ").alias("toks"),
    ]).collect())

    result = {"ntok_by_country": {}}
    for country in ["India", "US"]:
        c = name_rows.filter(pl.col("country") == country)
        result["ntok_by_country"][country] = {
            "n": int(c.height),
            "ntok_mean": round(float(c["toks"].list.len().mean()), 3),
            "ntok_median": int(c["toks"].list.len().median()),
            "ntok_p90": int(c["toks"].list.len().quantile(0.9)),
            "empty_name": round(float((c["name_norm_suf"] == "").mean()), 5),
        }

    toks = (s1.with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
            .explode("toks")
            .filter(pl.col("toks") != "")
            .with_columns([
                (pl.col("toks").str.len_chars() > 2).alias("disc"),
            ]))
    firsttok = (s1.with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
                .select(["country", pl.col("toks").list.first().alias("ftok")])
                .collect())
    lasttok = (s1.with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
               .select(["country", pl.col("toks").list.last().alias("ltok")])
               .collect())

    def top_vals(df, col, n=20):
        return (df.group_by([col, "country"]).agg(pl.len().alias("n"))
                .sort(["country", "n"], descending=[False, True])
                .group_by("country", maintain_order=True)
                .agg(pl.col(col).alias("top"), pl.col("n").alias("cnt"))
                .to_dict(as_series=False))

    # global token frequency (per country)
    tokfreq = (s1.with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
               .explode("toks").filter(pl.col("toks") != "")
               .group_by(["toks", "country"]).agg(pl.len().alias("df")))
    tokcols = tokfreq.collect()
    per_country = {}
    for country in ["India", "US"]:
        tt = tokcols.filter(pl.col("country") == country)
        table = tt.sort("df", descending=True).head(25).to_dict(as_series=False)
        per_country[country] = {"top_tokens": table}
    result["top_tokens"] = per_country

    for label, col in [("first_tokens", "ftok"), ("last_tokens", "ltok")]:
        tv = {}
        for country in ["India", "US"]:
            tt = (firsttok if label == "first_tokens" else lasttok)
            rows = (tt.filter(pl.col("country") == country)
                    .group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).head(15)
                    .to_dict(as_series=False))
            tv[country] = rows
        result[label] = tv

    with open(os.path.join(OUT, "06_name.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()