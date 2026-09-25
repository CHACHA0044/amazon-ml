"""13 - Validation-split sanity check: is there a pre-built train/valid/test split,
and are S1 ids disjoint across splits? (leakage check)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import polars as pl

from common import WORK_DIR, REPO_ROOT

OUT = os.path.join(WORK_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)


def main():
    # 1) find any split artifacts or split references in the repo
    split_files = []
    for pat in ["**/split*", "**/*valid*", "**/*val*", "**/*fold*"]:
        split_files += glob.glob(os.path.join(REPO_ROOT, pat), recursive=True)
    split_files = [p for p in split_files if "node_modules" not in p and ".venv" not in p
                   and "dataset" not in p.lower()]

    refs = []
    for root, _, files in os.walk(REPO_ROOT):
        if ".venv" in root or "node_modules" in root or "work" in root or "dataset" in root:
            continue
        for f in files:
            if f.endswith((".py", ".md", ".json", ".ipynb")):
                p = os.path.join(root, f)
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                except Exception:
                    continue
                for kw in ["valid", "validation_split", "train_test_split", "fold", "create_splits"]:
                    if kw in txt:
                        refs.append(p)
                        break

    # 2) ids in train vs test are inherently disjoint (separate files); check no S1 id string
    #    appears in both, and that GT only references train ids.
    tr_s1 = set(pl.scan_parquet(os.path.join(WORK_DIR, "train_records.parquet"))
                .filter(pl.col("src") == 1).select("id").unique().collect()["id"].to_list())
    te_s1 = set(pl.scan_parquet(os.path.join(WORK_DIR, "test_records.parquet"))
                .filter(pl.col("src") == 1).select("id").unique().collect()["id"].to_list())
    overlap = tr_s1 & te_s1

    # GT vs test overlap (GT should only cover train candidates)
    s2_in_test = set(pl.scan_parquet(os.path.join(WORK_DIR, "test_records.parquet"))
                     .filter(pl.col("src") == 2).select("id").unique().collect()["id"].to_list())
    gt_matched = set()
    # sample GT ids for speed
    gt_s2 = set(pl.scan_parquet(os.path.join(WORK_DIR, "gt_long.parquet"))
                .filter(pl.col("msrc") == "S2").select("matched_id").unique()
                .collect()["matched_id"].to_list())
    gt_in_test = gt_s2 & s2_in_test

    split_refs = [r for r in refs for kw in ["split", "fold", "validation"] if kw in r]
    out = {
        "split_artifacts_found": split_files,
        "code_references_to_splits": split_refs,
        "train_test_s1_id_overlap": len(overlap),
        "gt_s2_ids_present_in_test_s2": len(gt_in_test),
        "conclusion": "no_prebuilt_train_valid_split" if not split_files and not split_refs else "split_exists",
    }
    with open(os.path.join(OUT, "13_splitcheck.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()