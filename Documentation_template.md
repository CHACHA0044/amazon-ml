# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Amazon ML Team  
**Team Members:** Pranav & Pair Programming AI  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present a high-precision hybrid entity resolution architecture combining multi-channel deterministic blocking, a prior-shift calibrated LightGBM GBDT ranker, and disjoint star-cluster graph assignment. Our solution achieves strong cross-source matching while strictly preserving 1-to-many S1-to-{S2,S3} topology without cross-entity collisions.

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** Hybrid Multi-Channel Blocking + GBDT Ranker + Disjoint Star-Cluster Partitioning  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Macro F0.5 = 0.6881 (Precision: 0.8448, Recall: 0.6125)
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
