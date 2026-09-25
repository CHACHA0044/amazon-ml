"""One-time preprocessing: build compact normalized record artifacts.

Reads raw TSVs (read-only), computes normalization/token features, writes:
  work/train_records.parquet   (all train S1/S2/S3 records)
  work/test_records.parquet    (all test S1/S2/S3 records)
  work/gt_long.parquet         (s1_id, matched_id, matched_source for train GT)

Cannot be re-run harmlessly; the raw files are never modified.
"""
import concurrent.futures as cf
import glob
import os
import re
import sys
import time
import unicodedata

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import (
    WORK_DIR, TRAIN_S1, TRAIN_S2, TRAIN_S3, TEST_S1, TEST_S2, TEST_S3, TRAIN_GT,
    TRAIN_RECORDS, TEST_RECORDS, GT_LONG,
    normalize_name, normalize_address, numeric_tokens, fold_accent,
)

CHUNK = 250_000
MAX_WORKERS = 8


def _build_accents():
    m = {}
    start = time.time()
    for cp in range(0x00C0, 0x0370):  # Latin-1 Supplement, Ext-A, Ext-B, some IPA/Greek
        ch = chr(cp)
        folded = unicodedata.normalize("NFKD", ch)
        ascii_f = "".join(c for c in folded if not unicodedata.combining(c))
        if ascii_f and ascii_f.isascii():
            m[cp] = ascii_f
    # Handle ligatures / non-Latin precomposed that NFKD expands to ASCII+marks
    for ch in ["\uFB01", "\uFB02", "\u0153", "\u0152", "\u00df", "\u00d0", "\u00de"]:
        folded = unicodedata.normalize("NFKD", ch)
        ascii_f = "".join(c for c in folded if not unicodedata.combining(c))
        if ascii_f:
            m[ord(ch)] = ascii_f
    print(f"  accent map size={len(m)} built in {time.time()-start:.1f}s", flush=True)
    return m


ACCENT_MAP = _build_accents()


def fold_fast(s):
    if not s:
        return s
    if s.isascii():
        return s
    out = s.translate(ACCENT_MAP)
    if out == s and any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", s)):
        out = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return out


PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")


def core(s):
    t = fold_fast(s).lower()
    t = PUNCT_RE.sub(" ", t)
    return SPACE_RE.sub(" ", t).strip()


def clean_name(name):
    return core(name)


def clean_name_suf(name):
    return re.sub(r"\s+", " ", re.sub(r"\blimited\b", "ltd",
            re.sub(r"\bincorporated\b", "inc",
            re.sub(r"\bcorporation\b", "corp",
            re.sub(r"\bcompany\b", "co",
            re.sub(r"\bprivate\b", "pvt", core(name))))))).strip()


def clean_addr(addr):
    if not addr:
        return ""
    return core(addr)


def process_chunk(df):
    """Worker: pandas chunk -> polars records frame."""
    ids = df["entity_id"].astype(str).tolist()
    src_vals = [1 if v.startswith("S1-") else (2 if v.startswith("S2-") else 3)
                for v in ids]
    country = df["country"].astype(str).tolist()
    name = df["business_name"].astype(str).tolist()
    addr_all = df["business_address"].astype(str).tolist()

    nn = [clean_name(x) for x in name]
    ns = [clean_name_suf(x) for x in name]
    an = [clean_addr(x) for x in addr_all]
    anum = [" ".join(numeric_tokens(x)) for x in addr_all]
    nlen = [len(x) for x in name]
    alen = [len(x) for x in addr_all]
    ntok = [len(x.split()) if x else 0 for x in nn]
    atok = [len(x.split()) if x else 0 for x in an]
    na_name = [1 if not x.isascii() else 0 for x in name]
    na_addr = [1 if not x.isascii() else 0 for x in addr_all]
    missing = [1 if x == "" else 0 for x in addr_all]

    return pl.DataFrame({
        "src": pl.Series(src_vals, dtype=pl.Int8),
        "id": pl.Series(ids),
        "country": pl.Series(country),
        "name": pl.Series(name),
        "addr": pl.Series(addr_all),
        "name_norm": pl.Series(nn),
        "name_norm_suf": pl.Series(ns),
        "addr_norm": pl.Series(an),
        "addr_num": pl.Series(anum),
        "nlen": pl.Series(nlen, dtype=pl.Int32),
        "alen": pl.Series(alen, dtype=pl.Int32),
        "ntok": pl.Series(ntok, dtype=pl.Int32),
        "atok": pl.Series(atok, dtype=pl.Int32),
        "non_a_name": pl.Series(na_name, dtype=pl.Int8),
        "non_a_addr": pl.Series(na_addr, dtype=pl.Int8),
        "addr_missing": pl.Series(missing, dtype=pl.Int8),
    })


def build_records(inpath, outpath, label):
    t0 = time.time()
    chunks = []
    for df in pd.read_csv(inpath, sep="\t", dtype=str, keep_default_na=False,
                          encoding="utf-8", encoding_errors="replace",
                          chunksize=CHUNK):
        chunks.append(df)
    print(f"[{label}] read {len(chunks)} chunks", flush=True)

    parts = []
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, out in enumerate(ex.map(process_chunk, chunks)):
            cpath = os.path.join(WORK_DIR, "chunks", f"{label}_{i}.parquet")
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            out.write_parquet(cpath)
            parts.append(cpath)
            if (i + 1) % 10 == 0:
                print(f"[{label}] {i+1}/{len(chunks)} chunks", flush=True)

    pl.scan_parquet(sorted(parts)).sink_parquet(outpath, compression="zstd")
    for p in parts:
        os.remove(p)
    print(f"[{label}] wrote {outpath} in {time.time()-t0:.1f}s", flush=True)


def build_gt(inpath, outpath):
    t0 = time.time()
    rows = []
    with open(inpath, "r", encoding="utf-8", errors="replace", newline="") as f:
        import csv
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            s1 = row["source1_entity_id"].strip()
            mids = (row.get("matched_entity_ids") or "").strip()
            if not mids:
                continue
            for m in mids.split(","):
                m = m.strip()
                if not m:
                    continue
                rows.append((s1, m, "S2" if m.startswith("S2-") else "S3"))
    df = pl.DataFrame({
        "s1_id": [r[0] for r in rows],
        "matched_id": [r[1] for r in rows],
        "msrc": [r[2] for r in rows],
    })
    df.write_parquet(outpath, compression="zstd")
    print(f"[gt] {len(rows):,} pairs -> {outpath} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK_DIR, "chunks"), exist_ok=True)
    print("Building train records ...", flush=True)
    build_records(TRAIN_S1, os.path.join(WORK_DIR, "train_s1.parquet"), "tr_s1")
    build_records(TRAIN_S2, os.path.join(WORK_DIR, "train_s2.parquet"), "tr_s2")
    build_records(TRAIN_S3, os.path.join(WORK_DIR, "train_s3.parquet"), "tr_s3")

    s1 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s1.parquet"))
    s2 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s2.parquet"))
    s3 = pl.scan_parquet(os.path.join(WORK_DIR, "train_s3.parquet"))
    pl.concat([s1, s2, s3]).sink_parquet(TRAIN_RECORDS, compression="zstd")
    os.remove(os.path.join(WORK_DIR, "train_s1.parquet"))
    os.remove(os.path.join(WORK_DIR, "train_s2.parquet"))
    os.remove(os.path.join(WORK_DIR, "train_s3.parquet"))

    print("Building test records ...", flush=True)
    build_records(TEST_S1, os.path.join(WORK_DIR, "test_s1.parquet"), "te_s1")
    build_records(TEST_S2, os.path.join(WORK_DIR, "test_s2.parquet"), "te_s2")
    build_records(TEST_S3, os.path.join(WORK_DIR, "test_s3.parquet"), "te_s3")

    s1 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s1.parquet"))
    s2 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s2.parquet"))
    s3 = pl.scan_parquet(os.path.join(WORK_DIR, "test_s3.parquet"))
    pl.concat([s1, s2, s3]).sink_parquet(TEST_RECORDS, compression="zstd")
    os.remove(os.path.join(WORK_DIR, "test_s1.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s2.parquet"))
    os.remove(os.path.join(WORK_DIR, "test_s3.parquet"))

    print("Building GT long ...", flush=True)
    build_gt(TRAIN_GT, GT_LONG)
    print("PREP DONE", flush=True)