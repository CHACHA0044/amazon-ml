"""01: Positive vs negative pair similarity analysis (memory-bounded).

Constructs:
  - full positive-pair frame (all GT pairs) with CHEAP exact-match flags
    -> work/pos_pairs_full.parquet
  - sampled pair pool (positives + negatives) with CHEAP + HEAVY features
    -> work/pair_pool.parquet
  - feature-separation summary -> work/analysis/01_pair_sim_summary.json

Negatives channels (all bounded by analytic block-size caps):
  n_random : random same-country candidate (easy)
  n_nameblk: same normalized name, different entity (hard)
  n_addrblk: same normalized address, different entity (hard)
  n_raretok: shares a rare name token (df in [3,4000]) (hard)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import polars as pl

from common import WORK_DIR, TRAIN_RECORDS, GT_LONG
from pairfeat import heavy_features_batch, auc_score, set_rare_vocab

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_POS_SAMPLE = 600_000
N_NEG_PER_CHANNEL = 250_000
SIZE_CAP = 20_000
PER_S1 = 6
HEAVY_WORKERS = 8


def _hf_worker(task):
    return heavy_features_batch(*task)


def load_records():
    tr = pl.scan_parquet(TRAIN_RECORDS)
    s1 = tr.filter(pl.col("src") == 1).select([
        "id", "country", "name", "name_norm", "name_norm_suf", "addr",
        "addr_norm", "addr_num", "nlen", "alen", "ntok", "atok",
        "non_a_name", "non_a_addr", "addr_missing",
    ])
    cand = tr.filter(pl.col("src") > 1).select([
        "src", "id", "country", "name", "name_norm", "name_norm_suf", "addr",
        "addr_norm", "addr_num", "nlen", "alen", "ntok", "atok",
        "non_a_name", "non_a_addr", "addr_missing",
    ])
    return s1, cand


def build_pos_full(s1, cand):
    gt = pl.scan_parquet(GT_LONG)
    q = s1.select([
        "id", "country", "name", "name_norm", "name_norm_suf", "addr",
        "addr_norm", "addr_num", "nlen", "alen", "ntok", "atok",
        "non_a_name", "non_a_addr", "addr_missing",
    ]).rename({
        "id": "s1_id",
        "country": "q_country", "name": "q_name", "name_norm": "q_name_norm",
        "name_norm_suf": "q_name_norm_suf", "addr": "q_addr",
        "addr_norm": "q_addr_norm", "addr_num": "q_addr_num",
        "nlen": "q_nlen", "alen": "q_alen", "ntok": "q_ntok", "atok": "q_atok",
        "non_a_name": "q_non_a_name", "non_a_addr": "q_non_a_addr",
        "addr_missing": "q_addr_missing",
    })
    c = cand.rename({
        "src": "c_src", "id": "cand_id", "country": "c_country",
        "name": "c_name", "name_norm": "c_name_norm",
        "name_norm_suf": "c_name_norm_suf", "addr": "c_addr",
        "addr_norm": "c_addr_norm", "addr_num": "c_addr_num",
        "nlen": "c_nlen", "alen": "c_alen", "ntok": "c_ntok", "atok": "c_atok",
        "non_a_name": "c_non_a_name", "non_a_addr": "c_non_a_addr",
        "addr_missing": "c_addr_missing",
    })
    pos = (
        gt.lazy()
        .join(q, on="s1_id", how="inner")
        .join(c, left_on="matched_id", right_on="cand_id", how="inner")
        .with_columns([
            (pl.col("q_name") == pl.col("c_name")).alias("name_raw_eq"),
            (pl.col("q_name").str.to_lowercase() == pl.col("c_name").str.to_lowercase()).alias("name_lower_eq"),
            (pl.col("q_name_norm") == pl.col("c_name_norm")).alias("name_norm_eq"),
            (pl.col("q_name_norm_suf") == pl.col("c_name_norm_suf")).alias("name_norm_suf_eq"),
            (pl.col("q_addr") == pl.col("c_addr")).alias("addr_raw_eq"),
            (pl.col("q_addr").str.to_lowercase() == pl.col("c_addr").str.to_lowercase()).alias("addr_lower_eq"),
            (pl.col("q_addr_norm") == pl.col("c_addr_norm")).alias("addr_norm_eq"),
            ((pl.col("q_name_norm_suf") == pl.col("c_name_norm_suf"))
             & (pl.col("q_addr_norm") == pl.col("c_addr_norm"))).alias("both_norm_eq"),
            (pl.col("q_ntok") == pl.col("c_ntok")).alias("ntok_eq"),
            (pl.col("q_atok") == pl.col("c_atok")).alias("atok_eq"),
            (pl.col("q_nlen") - pl.col("c_nlen")).abs().alias("nlen_diff"),
            (pl.col("q_alen") - pl.col("c_alen")).abs().alias("alen_diff"),
            ((pl.col("q_addr_missing") == 1) | (pl.col("c_addr_missing") == 1)).alias("addr_missing_any"),
            ((pl.col("q_addr_missing") == 1) & (pl.col("c_addr_missing") == 1)).alias("addr_missing_both"),
            (pl.col("q_country") == pl.col("c_country")).alias("country_eq"),
        ])
        .select([
            "s1_id", "matched_id", "msrc", "q_country", "c_country",
            "name_raw_eq", "name_lower_eq", "name_norm_eq", "name_norm_suf_eq",
            "addr_raw_eq", "addr_lower_eq", "addr_norm_eq", "both_norm_eq",
            "ntok_eq", "atok_eq", "nlen_diff", "alen_diff",
            "addr_missing_any", "addr_missing_both", "country_eq",
        ])
    )
    out_path = os.path.join(WORK_DIR, "pos_pairs_full.parquet")
    pos.sink_parquet(out_path, compression="zstd")
    print(f"positive full frame -> {out_path}", flush=True)
    return pl.scan_parquet(out_path)


def sample_pos(pool_lazy, n):
    return (
        pool_lazy
        .with_columns(pl.col("s1_id").hash(SEED).alias("rnd"))
        .sort("rnd")
        .head(n)
    )


def bounded_join_negs(q, c, key_cols, channel, cap=SIZE_CAP, per_s1=PER_S1):
    """Pairs from exact-(key_cols) blocks with bounded per-block size."""
    q = q.select(["id", *key_cols])
    c = c.select(["id", *key_cols])
    qcnt = q.group_by(key_cols).agg(pl.len().alias("qcnt"))
    ccnt = c.group_by(key_cols).agg(pl.len().alias("ccnt"))
    keys = (
        qcnt.join(ccnt, on=key_cols, how="inner")
        .with_columns((pl.col("qcnt") * pl.col("ccnt")).alias("prod"))
        .filter(pl.col("prod") <= cap)
        .select(key_cols)
    )
    q = q.join(keys, on=key_cols, how="inner").rename({"id": "s1_id"})
    c = c.join(keys, on=key_cols, how="inner").rename({"id": "cand_id"})
    joined = (
        q.join(c, on=key_cols, how="inner")
        .with_columns(pl.col("s1_id").hash(SEED).alias("rnd"))
    )
    out = (
        joined.sort(["s1_id", "rnd"])
        .group_by("s1_id", maintain_order=True)
        .head(per_s1)
        .filter((pl.col("s1_id").hash(SEED + 7) % 100) < 55)
        .select(["s1_id", "cand_id", "country"])
        .with_columns(
            msrc=pl.when(pl.col("cand_id").str.starts_with("S2-")).then(pl.lit("S2")).otherwise(pl.lit("S3")),
            channel=pl.lit(channel),
        )
    )
    return out


def raretok_negs(s1, cand, rare_tok_set):
    q = (s1.select(["id", "country", "name_norm_suf"])
         .with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
         .explode("toks")
         .filter(pl.col("toks").is_in(list(rare_tok_set)) & (pl.col("toks") != "")))
    c = (cand.select(["id", "country", "name_norm_suf"])
         .with_columns(pl.col("name_norm_suf").str.split(" ").alias("toks"))
         .explode("toks")
         .filter(pl.col("toks").is_in(list(rare_tok_set)) & (pl.col("toks") != "")))
    return bounded_join_negs(q, c, ["country", "toks"], "n_raretok")


def build_random_negs(s1, cand):
    rng = np.random.default_rng(SEED)
    s1f = s1.select(["id", "country"]).collect()
    candf = cand.select(["id", "country"]).collect()
    rows = []
    for ctry in s1f["country"].unique().to_list():
        s1c = s1f.filter(pl.col("country") == ctry)["id"].to_list()
        c2 = candf.filter((pl.col("country") == ctry) & pl.col("id").str.starts_with("S2-"))["id"].to_list()
        c3 = candf.filter((pl.col("country") == ctry) & pl.col("id").str.starts_with("S3-"))["id"].to_list()
        n = min(len(s1c), (N_NEG_PER_CHANNEL // 4))
        s1sel = rng.choice(s1c, size=n, replace=False)
        if c2:
            rows.extend((s1sel[i], c2[rng.integers(len(c2))], "S2", ctry, "n_random") for i in range(n))
        if c3:
            rows.extend((s1sel[i], c3[rng.integers(len(c3))], "S3", ctry, "n_random") for i in range(n))
    return pl.DataFrame(rows, schema=["s1_id", "cand_id", "msrc", "country", "channel"],
                        orient="row")


def drop_matches(neg_frame_lazy):
    gt = pl.scan_parquet(GT_LONG).select(["s1_id", "matched_id"]).rename({"matched_id": "cand_id"})
    return neg_frame_lazy.join(gt, on=["s1_id", "cand_id"], how="anti")


def compute_rare_vocab(tr, lo=3, hi=4000):
    tokens = (
        tr.select(pl.col("name_norm_suf").str.split(" "))
        .explode("name_norm_suf")
        .filter(pl.col("name_norm_suf") != "")
        .group_by("name_norm_suf")
        .agg(pl.len().alias("df"))
        .filter((pl.col("df") >= lo) & (pl.col("df") <= hi))
    )
    df = tokens.collect()
    vocab = df["name_norm_suf"].to_list()
    print(f"rare vocab size (df in [{lo},{hi}]): {len(vocab):,}", flush=True)
    return vocab


def heavytopool(pool, nproc=HEAVY_WORKERS):
    names_q = pool["q_name_norm_suf"].to_list()
    names_c = pool["c_name_norm_suf"].to_list()
    addr_q = pool["q_addr_norm"].to_list()
    addr_c = pool["c_addr_norm"].to_list()
    num_q = pool["q_addr_num"].to_list()
    num_c = pool["c_addr_num"].to_list()
    n = len(names_q)
    chunk = (n + nproc - 1) // nproc
    tasks = []
    for i in range(0, n, chunk):
        tasks.append((names_q[i:i + chunk], names_c[i:i + chunk],
                      addr_q[i:i + chunk], addr_c[i:i + chunk],
                      num_q[i:i + chunk], num_c[i:i + chunk]))
    results = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for part in ex.map(_hf_worker, tasks):
            results.append(part)
    merged = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    return pool.with_columns([pl.Series(k, v).alias(k) for k, v in merged.items()])


def cheap_stats(pos_full):
    n = pos_full.height
    st = {"n_pairs": int(n)}
    for col in ["name_raw_eq", "name_lower_eq", "name_norm_eq", "name_norm_suf_eq",
                "addr_raw_eq", "addr_lower_eq", "addr_norm_eq", "both_norm_eq",
                "ntok_eq", "atok_eq", "country_eq", "addr_missing_any", "addr_missing_both"]:
        st[col] = round(float(pos_full[col].mean()), 6)
    st["by_msrc"] = {str(r["msrc"]): int(r["n"]) for r in
                     pos_full.group_by("msrc").agg(pl.len().alias("n")).sort("msrc").iter_rows(named=True)}
    st["by_country"] = {str(r["q_country"]): int(r["n"]) for r in
                        pos_full.group_by("q_country").agg(pl.len().alias("n")).sort("q_country").iter_rows(named=True)}
    st["nlen_diff_mean"] = float(pos_full["nlen_diff"].mean())
    st["alen_diff_mean"] = float(pos_full["alen_diff"].mean())
    st["nlen_diff_p90"] = float(pos_full["nlen_diff"].quantile(0.9))
    st["alen_diff_p90"] = float(pos_full["alen_diff"].quantile(0.9))
    return st


def summarize_pool(pool):
    features = [
        "pool_name_eq", "pool_addr_eq", "pool_both_eq", "pool_ntok_eq", "pool_atok_eq",
        "name_tok_jac", "name_contain_q", "name_contain_c",
        "addr_tok_jac", "addr_contain_q", "addr_contain_c",
        "name_char2", "name_char3", "addr_char2", "addr_char3",
        "num_overlap", "rare_shared",
    ]
    summary = {}
    pos = pool.filter(pl.col("label") == 1)
    neg = pool.filter(pl.col("label") == 0)
    summary["n_pos"] = int(pos.height)
    summary["n_neg"] = int(neg.height)
    summary["channels"] = {str(r["channel"]): int(r["n"]) for r in
                           pool.group_by("channel").agg(pl.len().alias("n")).sort("channel").iter_rows(named=True)}
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
        "pool_name_eq & pool_addr_eq": ((pool["pool_name_eq"]) & (pool["pool_addr_eq"])),
        "name_tok_jac>=0.5": (pool["name_tok_jac"] >= 0.5),
        "name_char2>=0.8": pool["name_char2"] >= 0.8,
        "addr_char2>=0.8": pool["addr_char2"] >= 0.8,
        "num_overlap>=1": pool["num_overlap"] >= 1,
        "pool_name_eq & addr_char2>=0.5": ((pool["pool_name_eq"]) & (pool["addr_char2"] >= 0.5)),
        "pool_both_eq": pool["pool_both_eq"],
    }
    summary["probes"] = {}
    for name, m in probes.items():
        sub = pool.filter(m)
        summary["probes"][name] = {
            "n": int(sub.height),
            "pos_frac": round(float(sub["label"].mean()), 5),
            "neg_frac": round(1.0 - float(sub["label"].mean()), 5),
        } if sub.height else {"n": 0}
    return summary


def main():
    t0 = time.time()
    print("loading records ...", flush=True)
    s1, cand = load_records()
    tr = pl.scan_parquet(TRAIN_RECORDS)

    print("building full positive frame ...", flush=True)
    pos_full = build_pos_full(s1, cand)
    st = cheap_stats(pos_full.collect())
    with open(os.path.join(OUT, "01_pos_cheap.json"), "w") as f:
        json.dump(st, f, indent=2)
    print("cheap positive stats:", json.dumps(st), flush=True)

    print("computing rare vocab ...", flush=True)
    rare_tok_set = set(compute_rare_vocab(tr))
    set_rare_vocab(rare_tok_set)

    print("building negatives ...", flush=True)
    channel_frames = {}
    neg_random = build_random_negs(s1, cand)
    channel_frames["n_nameblk"] = drop_matches(
        bounded_join_negs(s1, cand, ["country", "name_norm_suf"], "n_nameblk"))
    channel_frames["n_addrblk"] = drop_matches(
        bounded_join_negs(s1, cand, ["country", "addr_norm"], "n_addrblk"))
    channel_frames["n_raretok"] = drop_matches(raretok_negs(s1, cand, rare_tok_set))

    slot = os.path.join(WORK_DIR, "chunks", "neg_channel.parquet")
    capped = []
    for ch in ["n_random", "n_nameblk", "n_addrblk", "n_raretok"]:
        if ch == "n_random":
            sub = neg_random.sample(n=min(N_NEG_PER_CHANNEL, neg_random.height), seed=SEED)
        else:
            channel_frames[ch].sink_parquet(slot, compression="zstd")
            sub = pl.read_parquet(slot)
            if sub.height > N_NEG_PER_CHANNEL:
                sub = sub.sample(n=N_NEG_PER_CHANNEL, seed=SEED)
        sub = sub.select(["s1_id", "cand_id", "country", "msrc", "channel"])
        capped.append(sub)
    negs = pl.concat(capped)
    print(f"negatives total: {negs.height:,}", flush=True)

    print("sampling positives for heavy features ...", flush=True)
    pos_pool = (sample_pos(pos_full, N_POS_SAMPLE)
                .collect()
                .select([
                    "s1_id", "matched_id", "msrc", "q_country", "c_country",
                    "name_raw_eq", "name_lower_eq", "name_norm_eq", "name_norm_suf_eq",
                    "addr_raw_eq", "addr_lower_eq", "addr_norm_eq", "both_norm_eq",
                    "ntok_eq", "atok_eq", "nlen_diff", "alen_diff", "addr_missing_any",
                    "addr_missing_both", "country_eq",
                ])
                .with_columns([pl.lit("pos").alias("channel"), pl.lit(1).alias("label")])
                .rename({"q_country": "country"}))

    neg_pool = negs.with_columns(pl.lit(0).alias("label")).rename({"cand_id": "matched_id"})

    rec = tr.select(["src", "id", "name_norm_suf", "addr_norm", "addr_num"]).collect()
    src_df = rec.filter(pl.col("src") == 1).select(
        ["id", pl.col("name_norm_suf").alias("q_name_norm_suf"),
         pl.col("addr_norm").alias("q_addr_norm"),
         pl.col("addr_num").alias("q_addr_num")]).rename({"id": "s1_id"})
    cand_df = rec.filter(pl.col("src") > 1).select(
        ["id", pl.col("name_norm_suf").alias("c_name_norm_suf"),
         pl.col("addr_norm").alias("c_addr_norm"),
         pl.col("addr_num").alias("c_addr_num")])

    pos_pool = (pos_pool.join(src_df, on="s1_id", how="left")
                .join(cand_df, left_on="matched_id", right_on="id", how="left"))
    neg_pool = (neg_pool.join(src_df, on="s1_id", how="left")
                .join(cand_df, left_on="matched_id", right_on="id", how="left"))

    def add_pool_flags(df):
        return df.with_columns([
            (pl.col("q_name_norm_suf") == pl.col("c_name_norm_suf")).alias("pool_name_eq"),
            (pl.col("q_addr_norm") == pl.col("c_addr_norm")).alias("pool_addr_eq"),
            ((pl.col("q_name_norm_suf") == pl.col("c_name_norm_suf"))
             & (pl.col("q_addr_norm") == pl.col("c_addr_norm"))).alias("pool_both_eq"),
            ((pl.col("q_name_norm_suf").str.split(" ").list.len())
             == (pl.col("c_name_norm_suf").str.split(" ").list.len())).alias("pool_ntok_eq"),
            ((pl.col("q_addr_norm").str.split(" ").list.len())
             == (pl.col("c_addr_norm").str.split(" ").list.len())).alias("pool_atok_eq"),
            ((pl.col("q_addr_norm") == "") | (pl.col("c_addr_norm") == "")).alias("pool_addr_any_missing"),
        ])

    pos_pool = add_pool_flags(pos_pool)
    neg_pool = add_pool_flags(neg_pool)

    cols = ["s1_id", "matched_id", "msrc", "country", "channel", "label",
            "pool_name_eq", "pool_addr_eq", "pool_both_eq", "pool_ntok_eq",
            "pool_atok_eq", "pool_addr_any_missing",
            "q_name_norm_suf", "c_name_norm_suf", "q_addr_norm", "c_addr_norm",
            "q_addr_num", "c_addr_num"]
    pool = pl.concat([pos_pool.select(cols), neg_pool.select(cols)])
    print(f"pool size: {pool.height:,}", flush=True)

    print("computing heavy features ...", flush=True)
    pool = heavytopool(pool)
    pool.write_parquet(os.path.join(WORK_DIR, "pair_pool.parquet"), compression="zstd")

    summary = summarize_pool(pool)
    with open(os.path.join(OUT, "01_pair_sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()