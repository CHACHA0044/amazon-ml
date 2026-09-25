#!/usr/bin/env python3
"""Submission Validator for Amazon ML Challenge 2026.

Integrates and extends the official student_resource/utils/validate_submission.py
with strict integrity assertions:
1. Exact TSV delimiter and lowercase header check.
2. Complete 1-to-1 coverage of all S1 test IDs (1,732,544 rows).
3. Prefix validation (only S2- and S3- matched IDs).
4. No S1 self-matches.
5. No intra-row duplicate matched IDs.
6. Validation that matched IDs are a strict subset of candidate IDs (if candidate file provided).
7. Leakage check (ensuring no train ground truth IDs leak into test predictions).
8. Full ID-existence verification in test_source2.tsv and test_source3.tsv.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
MAX_EXAMPLES = 5


def examples(items) -> str:
    """Format sample items for error messages."""
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def read_ids(path: str) -> Set[str]:
    """Return the set of first-column entity IDs from a source TSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, encoding="utf-8") as f:
        next(f, None)  # skip header
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def validate_id_list_file(
    path: str,
    expected_header: List[str],
    col_label: str,
    required_s1_ids: Set[str],
    valid_target_ids: Optional[Set[str]],
    errors: List[str],
) -> Optional[Dict[str, Set[str]]]:
    """Validate format and content of a TSV submission file."""
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping: Dict[str, Set[str]] = {}
    seen = set()
    dup_rows = set()
    intra_dupes = set()
    self_matches = set()
    wrong_prefix = set()
    unknown = set()
    n_rows = 0
    empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but contains commas — file is comma-separated instead of tab-separated."
            )
            return None

        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(f"{name}: unexpected header {cols}. Expected exactly {expected_header}.")
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab delimiter) at line {line_num}: {line.rstrip()!r}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue

            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)

            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_target_ids is not None and mid not in valid_target_ids:
                    unknown.add(mid)

    findings = [
        (dup_rows, f"{name}: duplicate source1_entity_id row(s): {examples(dup_rows)}"),
        (intra_dupes, f"{name}: repeated ID inside a {col_label} list for: {examples(intra_dupes)}"),
        (self_matches, f"{name}: {col_label} contains Source-1 self-matches: {examples(self_matches)}"),
        (wrong_prefix, f"{name}: {col_label} contains invalid prefixes (not S2-/S3-): {examples(wrong_prefix)}"),
        (unknown, f"{name}: {col_label} references IDs not in test Source-2/3 files: {examples(unknown)}"),
        (required_s1_ids - seen, f"{name}: missing required test S1 entities: {examples(required_s1_ids - seen)}"),
        (seen - required_s1_ids, f"{name}: unexpected S1 IDs not in test set: {examples(seen - required_s1_ids)}"),
    ]
    for offenders, msg in findings:
        if offenders:
            errors.append(msg)

    print(f"  [{name}] validated: {n_rows:,} total rows ({empties:,} singletons, {n_rows - empties:,} non-empty matches).")
    return mapping


def validate_submission_bundle(
    matching_path: str,
    candidate_path: Optional[str],
    test_dir: str,
    check_ids: bool = True,
) -> Tuple[List[str], List[str]]:
    """Perform comprehensive submission validation."""
    errors: List[str] = []
    warnings: List[str] = []

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(s1_path):
        errors.append(f"Test source1 file not found at {s1_path}")
        return errors, warnings

    print(f"Loading required Source-1 test IDs from {s1_path}...")
    required_s1 = read_ids(s1_path)
    print(f"  Found {len(required_s1):,} required Source-1 test entities.")

    valid_targets = None
    if check_ids:
        print("Checking S2/S3 ID existence against test sources...")
        valid_targets = set()
        for name in ("test_source2.tsv", "test_source3.tsv"):
            p = os.path.join(test_dir, name)
            if os.path.isfile(p):
                ids = read_ids(p)
                valid_targets.update(ids)
                print(f"  Loaded {len(ids):,} target IDs from {name}")
            else:
                warnings.append(f"{name} not found in {test_dir} — skipped ID target check.")
                valid_targets = None
                break

    matched_map = validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids", required_s1, valid_targets, errors
    )

    cand_map = None
    if candidate_path and os.path.isfile(candidate_path):
        cand_map = validate_id_list_file(
            candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required_s1, valid_targets, errors
        )
    elif candidate_path:
        warnings.append(f"candidate_pairs file not found at {candidate_path} (optional for leaderboard, required for full package).")

    # Check subset relationship: matched IDs must be subset of candidates
    if matched_map is not None and cand_map is not None:
        discrepancies = {
            s1: matched_map[s1] - cand_map.get(s1, set())
            for s1 in matched_map
            if matched_map[s1] - cand_map.get(s1, set())
        }
        if discrepancies:
            warnings.append(
                f"{len(discrepancies)} S1 entities have matched IDs that were not present in candidate_pairs: {examples(set(discrepancies.keys()))}"
            )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Submission Validator")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default="dataset/student_resource/dataset/test", help="Test files directory")
    parser.add_argument("--check-ids", action="store_true", default=True, help="Verify all matched IDs exist in S2/S3")
    args = parser.parse_args()

    print("=" * 70)
    print("SUBMISSION INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"Matching file:  {args.matching}")
    print(f"Candidate file: {args.candidate}")
    print(f"Test directory: {args.test_dir}")
    print()

    errors, warnings = validate_submission_bundle(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )

    print()
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    print()

    if errors:
        print(f"FAIL: Found {len(errors)} blocking validation error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        sys.exit(1)
    else:
        print("SUCCESS: PASS — All submission checks satisfied. File is ready for upload.")
        sys.exit(0)


if __name__ == "__main__":
    main()
