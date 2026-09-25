"""Submission Generation and Validation Module for Amazon ML Challenge 2026.

Produces BOTH required submission files as specified in problem_statement.txt & instructions.md §F2:
  1. output/matching_results.tsv (scored on leaderboard): source1_entity_id \\t matched_entity_ids
  2. output/candidate_pairs.tsv (blocking audit): source1_entity_id \\t candidate_entity_ids

Validates against the official competition validator:
  dataset/student_resource/utils/validate_submission.py
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Set
import polars as pl

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def generate_submission_bundle(
    matching_predictions: Dict[str, Set[str]],
    candidate_predictions: Dict[str, Set[str]],
    test_s1_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_official_validator: bool = True,
) -> Dict[str, str]:
    """Generate and validate the complete 2-file submission bundle."""
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    if test_s1_path is None:
        test_s1_path = os.path.join(
            REPO_ROOT, "dataset", "student_resource", "dataset", "test", "test_source1.tsv"
        )

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Reading test query IDs from {test_s1_path}...", flush=True)
    test_s1_df = pl.read_csv(
        test_s1_path,
        separator="\t",
        columns=["entity_id"],
        quote_char=None,
    )
    s1_ids = test_s1_df["entity_id"].to_list()
    total_test_s1 = len(s1_ids)

    print(f"Writing {total_test_s1:,} matching rows to {matching_path}...", flush=True)
    n_matching_preds = 0
    total_match_pairs = 0
    with open(matching_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matches = matching_predictions.get(s1_id, set())
            if matches:
                # Ensure all matched IDs are subsets of candidate IDs
                if s1_id in candidate_predictions:
                    candidate_predictions[s1_id].update(matches)
                f.write(f"{s1_id}\t{','.join(sorted(matches))}\n")
                n_matching_preds += 1
                total_match_pairs += len(matches)
            else:
                f.write(f"{s1_id}\t\n")

    print(f"Writing {total_test_s1:,} candidate rows to {candidate_path}...", flush=True)
    total_cand_pairs = 0
    with open(candidate_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = candidate_predictions.get(s1_id, set())
            if cands:
                f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
                total_cand_pairs += len(cands)
            else:
                f.write(f"{s1_id}\t\n")

    print("Two-file submission package written successfully.", flush=True)
    print(f"  matching_results.tsv : {total_match_pairs:,} matches across {n_matching_preds:,} S1 ({n_matching_preds/total_test_s1*100:.2f}%)", flush=True)
    print(f"  candidate_pairs.tsv  : {total_cand_pairs:,} candidates across {total_test_s1:,} S1", flush=True)

    if run_official_validator:
        validator_script = os.path.join(
            REPO_ROOT, "dataset", "student_resource", "utils", "validate_submission.py"
        )
        test_dir = os.path.join(REPO_ROOT, "dataset", "student_resource", "dataset", "test")
        if os.path.exists(validator_script):
            print(f"Running official competition validator: {validator_script}...", flush=True)
            python_exe = sys.executable
            cmd = [
                python_exe,
                validator_script,
                "--matching", matching_path,
                "--candidate", candidate_path,
                "--test-dir", test_dir,
            ]
            res = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "dataset", "student_resource"), capture_output=True, text=True)
            print(res.stdout, flush=True)
            if res.stderr:
                print(res.stderr, flush=True)
            if res.returncode != 0:
                raise RuntimeError(f"Official validator failed with exit code {res.returncode}")
            print("OFFICIAL VALIDATOR PASSED (Exit code 0).", flush=True)

    return {
        "matching_path": matching_path,
        "candidate_path": candidate_path,
        "total_test_s1": total_test_s1,
        "total_match_pairs": total_match_pairs,
        "total_cand_pairs": total_cand_pairs,
    }
