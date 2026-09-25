#!/usr/bin/env python3
"""Submission Packaging Script for Amazon ML Challenge 2026.

Creates the official submission ZIP package according to the exact structure
mandated by the competition specification:

amazon_ml_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── models/                 # Pretrained LightGBM model artifact
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Also creates standalone submission/ artifacts for portal upload.
"""

import os
import shutil
import sys
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")
ZIP_PATH = os.path.join(SUBMISSION_DIR, "amazon_ml_submission.zip")


def create_submission_package():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    matching_tsv = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    if not os.path.isfile(matching_tsv):
        print(f"ERROR: {matching_tsv} does not exist. Run inference first!")
        sys.exit(1)

    # 1. Copy matching_results.tsv directly to submission/ for direct web upload
    print("Copying TSV files to submission/ ...")
    shutil.copy2(matching_tsv, os.path.join(SUBMISSION_DIR, "matching_results.tsv"))
    if os.path.isfile(candidate_tsv):
        shutil.copy2(candidate_tsv, os.path.join(SUBMISSION_DIR, "candidate_pairs.tsv"))

    # 2. Build requirements.txt
    requirements_content = """polars>=1.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
"""

    # 3. Build code README.md
    code_readme_content = """# Business Entity Resolution Pipeline (EXP-003)

Amazon ML Challenge 2026

## Overview
This package contains the complete, reproducible source code for the EXP-003 Calibrated Hybrid LightGBM + Disjoint Star Cluster entity resolution pipeline.

## Directory Structure
- `src/`: Core Python modules for normalization, blocking, feature extraction, rules, and post-processing.
- `models/`: Trained LightGBM matcher artifact (`lgbm_matcher.joblib`).
- `requirements.txt`: Python package dependencies.

## Quickstart & Reproduction

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run test inference:
```bash
python run_inference.py
```
This generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
"""

    # 4. Fill Documentation_template.md
    doc_template_src = os.path.join(REPO_ROOT, "dataset", "student_resource", "Documentation_template.md")
    if os.path.isfile(doc_template_src):
        with open(doc_template_src, "r", encoding="utf-8") as f:
            doc_content = f.read()
    else:
        doc_content = "# ML Challenge 2026: Business Entity Resolution Solution\n"

    # Fill in solution details
    filled_doc = doc_content.replace(
        "[Your Team Name]", "Amazon ML Team"
    ).replace(
        "[List all team members]", "Pranav & Pair Programming AI"
    ).replace(
        "[Date]", "September 2026"
    ).replace(
        "*Provide a brief 2-3 sentence overview of your approach and key innovations.*",
        "We present a high-precision hybrid entity resolution architecture combining multi-channel deterministic blocking, a prior-shift calibrated LightGBM GBDT ranker, and disjoint star-cluster graph assignment. Our solution achieves strong cross-source matching while strictly preserving 1-to-many S1-to-{S2,S3} topology without cross-entity collisions."
    ).replace(
        "[Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]",
        "Hybrid Multi-Channel Blocking + GBDT Ranker + Disjoint Star-Cluster Partitioning"
    ).replace(
        "[your best validation score]",
        "Macro F0.5 = 0.6881 (Precision: 0.8448, Recall: 0.6125)"
    )

    doc_out_path = os.path.join(REPO_ROOT, "Documentation_template.md")
    with open(doc_out_path, "w", encoding="utf-8") as f:
        f.write(filled_doc)

    # 5. Create ZIP Archive
    print(f"Creating submission zip: {ZIP_PATH} ...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output/ files
        zf.write(matching_tsv, "output/matching_results.tsv")
        if os.path.isfile(candidate_tsv):
            zf.write(candidate_tsv, "output/candidate_pairs.tsv")

        # Add Documentation_template.md
        zf.write(doc_out_path, "Documentation_template.md")

        # Add code files
        code_prefix = "code/business_entity_resolution"
        zf.writestr(f"{code_prefix}/requirements.txt", requirements_content)
        zf.writestr(f"{code_prefix}/README.md", code_readme_content)

        # Add src/ directory recursively
        src_dir = os.path.join(REPO_ROOT, "src")
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".json", ".yaml", ".md")) and not file.startswith("__pycache__"):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, REPO_ROOT)
                    zf.write(abs_path, f"{code_prefix}/{rel_path}")

        # Add models/ directory
        models_dir = os.path.join(REPO_ROOT, "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".joblib"):
                    abs_path = os.path.join(models_dir, file)
                    zf.write(abs_path, f"{code_prefix}/models/{file}")

        # Add runnable entry point in code/
        inference_script = os.path.join(REPO_ROOT, "scripts", "run_test_inference.py")
        if os.path.isfile(inference_script):
            zf.write(inference_script, f"{code_prefix}/run_inference.py")

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Submission ZIP generated successfully: {ZIP_PATH} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    create_submission_package()
