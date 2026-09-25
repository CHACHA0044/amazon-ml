# Submission Manifest — Amazon ML Challenge 2026

**Team Name:** Amazon ML Challenge Team  
**Submission Package:** `amazon_ml_submission.zip`  
**Evaluation Target:** Business Entity Resolution — Macro $F_{0.5}$  
**Date Generated:** September 2026  

---

## 1. Generated Package Artifacts

| Filename | Location | Description |
| :--- | :--- | :--- |
| `matching_results.tsv` | `submission/matching_results.tsv` | Final entity matching predictions for live leaderboard evaluation |
| `candidate_pairs.tsv` | `submission/candidate_pairs.tsv` | Blocking candidate audit mapping (verification & reduction audit) |
| `amazon_ml_submission.zip` | `submission/amazon_ml_submission.zip` | Full reproducible submission package containing code, model, outputs, & documentation |

---

## 2. Model & Experiment Provenance

- **Experiment ID:** `EXP-003` (Calibrated Hybrid LightGBM Matcher + Disjoint Star Cluster)
- **Model Architecture:**
  1. Multi-channel query-filtered inverted indexing (6 blocking channels: exact name, suffix name, address tokens, numeric tokens, rare tokens, country partition)
  2. Multi-tier deterministic rule scorer (10 precision rules with confidence gating)
  3. Prior-shift calibrated LightGBM GBDT ranker (19 pairwise features, trained on validation/train partition, prior ratio calibrated)
  4. Hybrid confidence fusion: $\text{Conf}_{\text{hybrid}} = \sqrt{P_{\text{LGBM}} \times \text{Conf}_{\text{rule}}}$
  5. Disjoint star-cluster assignment (Anchor threshold $\tau = 0.85$, $K_{\max} = 11$, 1-to-many S1-to-{S2,S3} bijection without collisions)
- **Training Data Used:** `dataset/student_resource/dataset/train/` (US & India records)
- **Test Data Used:** `dataset/student_resource/dataset/test/` (US, India, and France records)
- **Zero Label Leakage:** Test predictions generated purely through inference on unlabeled test sets (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`). No ground truth leakage or manual test overrides.

---

## 3. Validation & Benchmark Scores

> [!IMPORTANT]
> **OFFLINE VALIDATION ONLY — NOT LEADERBOARD SCORE**  
> The score below is our rigorous offline 5-fold cross-validation / holdout benchmark on training data. The official leaderboard test score will be determined by Unstop upon file evaluation.

- **Offline Validation Macro $F_{0.5}$:** **0.6881**
  - **Macro Precision:** **0.8448**
  - **Macro Recall:** **0.6125**
  - **US Sub-Region Macro $F_{0.5}$:** **0.6989**
  - **India Sub-Region Macro $F_{0.5}$:** **0.6720**

---

## 4. Verification & Validation Status

- **Validator Executed:** `scripts/validate_submission.py` & `dataset/student_resource/utils/validate_submission.py`
- **Integrity Checks:**
  - [x] TSV Tab-Delimited formatting (`\t`)
  - [x] Header matches exact specification (`source1_entity_id\tmatched_entity_ids`)
  - [x] Exact 1-to-1 row correspondence with test `test_source1.tsv` (1,732,544 rows)
  - [x] Correct singleton representation (empty string for non-matching entities)
  - [x] Valid ID prefixes only (`S2-`, `S3-`)
  - [x] Zero self-matches (`S1-` IDs strictly excluded from match lists)
  - [x] Zero intra-row duplicate IDs
  - [x] Matched entities strictly subset of candidate pairs
  - [x] ZIP file contains zero raw data, zero git files, zero cache files, zero credentials
