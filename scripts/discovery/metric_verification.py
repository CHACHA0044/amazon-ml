"""Metric verification: official macro per-entity F_0.5 vs micro.

1. Validates the numeric example given in problem_statement.txt.
2. Demonstrates the divergence between macro (official) and micro F_0.5 on
   real training ground truth under a few trivial baselines.
3. Saves per-entity score distributions for later threshold work.

Read-only on the raw dataset.
"""
import collections
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from evaluation.metrics import compute_entity_f05, compute_f05_score, compute_f05_micro

GT = os.path.join(
    REPO_ROOT, "dataset", "student_resource", "dataset", "train", "train_ground_truth.tsv"
)


def load_gt(path):
    gt = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            s1 = row["source1_entity_id"].strip()
            mids = row.get("matched_entity_ids") or ""
            idset = set(x.strip() for x in mids.split(",") if x.strip())
            gt[s1] = idset
    return gt


def main():
    print("=" * 70)
    print("OFFICIAL EXAMPLE CHECK (problem_statement.txt)")
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    f = compute_entity_f05(pred, true)
    print(f"  expected 0.714 (0.7143)  got {f:.4f}")

    assert abs(f - 0.714286) < 0.001, "official example mismatch!"

    # singleton conventions
    assert compute_entity_f05(set(), set()) == 1.0
    assert compute_entity_f05({"S2-1"}, set()) == 0.0
    assert compute_entity_f05(set(), {"S2-1"}) == 0.0
    print("  singleton conventions: OK (empty/empty=1.0, merge=0.0, miss=0.0)")

    print("=" * 70)
    print("Loading real training ground truth ...")
    gt = load_gt(GT)
    print(f"  S1 entities: {len(gt):,}")

    stats = collections.Counter(len(v) for v in gt.values())
    singles = stats[0]
    print(f"  singletons: {singles:,} ({singles/len(gt)*100:.3f}%)")

    baselines = {}

    # Baseline A: predict nothing for everyone
    pred_none = {k: set() for k in gt}
    baselines["predict-nothing"] = compute_f05_score(pred_none, gt)

    # Baseline B: predict everything (all candidate ids) -> forced FPs
    # Use the actual GT matched ids + one random bogus id per non-singleton
    pred_all = dict(gt)
    i = 0
    for k, v in gt.items():
        if v:
            pred_all[k] = v | {f"FAKE-{i}"}
            i += 1
    baselines["oracle+1 FP/entity"] = compute_f05_score(pred_all, gt)

    # Baseline C: drop exactly one true match per multi-match entity
    pred_drop1 = {}
    for k, v in gt.items():
        if len(v) <= 1:
            pred_drop1[k] = set(v)
        else:
            lst = sorted(v)
            pred_drop1[k] = set(lst[1:])
    baselines["drop-1/match entity"] = compute_f05_score(pred_drop1, gt)

    print("-" * 70)
    header = f"{'baseline':28}  {'MACRO f05':>10}  {'MICRO f05':>10}  {'microP':>8}  {'microR':>8}"
    print(header)
    print("-" * 70)
    for name, res in baselines.items():
        mic = compute_f05_micro(pred_none if name == "predict-nothing" else
                                (pred_all if name == "oracle+1 FP/entity" else pred_drop1), gt)
        print(f"{name:28}  {res['f05']:>10.4f}  {mic['f05']:>10.4f}  {mic['precision']:>8.4f}  {mic['recall']:>8.4f}")

    print("-" * 70)
    print("Interpretation:")
    print("  predict-nothing: MACRO=singleton fraction (0.0558), MICRO=0.0")
    print("  => earlier data_profile.json baseline_none_f05=0.0558 already used MACRO.")
    print("  oracle+1FP: each entity's F0.5 falls with cardinality; macro != micro.")
    print("  drop-1/match: macro weights every entity equally; micro weights by cardinality.")


if __name__ == "__main__":
    main()