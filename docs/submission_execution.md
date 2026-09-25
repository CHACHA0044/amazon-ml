# Submission Execution Guide — Amazon ML Challenge 2026

**Experiment Reference:** EXP-003 Calibrated Hybrid LightGBM + Disjoint Star Clustering  
**Target Output:** `output/matching_results.tsv` (Leaderboard) & `output/candidate_pairs.tsv` (Audit)  

---

## 1. System Requirements & Environment

### Python Version
- **Python:** 3.10+ (tested on Python 3.10 and 3.12, 64-bit)

### Dependencies
All required libraries are standard, open-source (permissive MIT / Apache-2.0 / BSD):
- `polars >= 1.0.0` (vectorized query loading and streaming data processing)
- `lightgbm >= 4.0.0` (gradient boosted decision tree inference)
- `scikit-learn >= 1.3.0` (evaluation metrics and calibration utilities)
- `joblib >= 1.3.0` (model serialization)
- `numpy >= 1.24.0` (matrix operations)

Install all requirements via:
```bash
pip install -r requirements.txt
```

---

## 2. Input Dataset Specification

The pipeline expects official test TSVs placed in the input directory (e.g. `dataset/test/`):
- `test_source1.tsv` (Source-1 reference queries)
- `test_source2.tsv` (Source-2 candidate entities)
- `test_source3.tsv` (Source-3 candidate entities)

No external lookups or ground truth labels are required or used.

---

## 3. Execution Commands

### Full Test Inference Pipeline
To run the complete end-to-end normalization, multi-channel blocking, pairwise feature extraction, LightGBM ranker, hybrid confidence scoring, and disjoint star clustering:

```bash
python scripts/run_test_inference.py
```

### Submission Validation Check
To run the offline submission validator against official schema and constraint checks:

```bash
python scripts/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```

---

## 4. Hardware & Resource Profile

- **Peak RAM Usage:** ~1.8 – 2.2 GB (stays strictly under the 2.5 GB target)
  - Memory efficiency is achieved by country chunking (France → India → US) and Polars columnar streaming.
- **CPU Execution:** Multi-threaded CPU execution (utilizes all available cores for blocking and LightGBM inference).
- **GPU Requirement:** None (100% CPU compatible).

---

## 5. Output Format and Guarantees

The generated `output/matching_results.tsv` satisfies all competition constraints:
1. **Header:** `source1_entity_id\tmatched_entity_ids` (exact tab delimiter, no extra columns).
2. **Row Count:** Exactly 1,732,544 rows matching every S1 test entity.
3. **Singletons:** Correctly represented with an empty matched column (`S1-xxxxx\t\n`).
4. **Matched IDs:** Strict comma-separated list of valid `S2-` and `S3-` IDs only.
5. **No Duplicates:** Neither duplicate rows nor duplicate matched IDs within any row.
