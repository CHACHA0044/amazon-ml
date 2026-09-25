# Experiment Plan & Modeling Strategy: Amazon ML Challenge 2026

## 1. System Architecture Overview

The entity resolution engine operates as a high-throughput, multi-stage pipeline:

```text
Raw Tables (S1, S2, S3)
       │
       ▼
1. Preprocessing & Normalization (Unicode NFKD, ASCII folding, lowercase, token clean)
       │
       ▼
2. Multi-Channel Blocking / Candidate Generation (Country partition, Exact Hash, MinHash LSH, Character n-gram TF-IDF)
       │
       ▼
3. Candidate Union & Deduplication (Top-K candidates per S1, target recall > 98%)
       │
       ▼
4. Feature Engineering (Jaro-Winkler, Levenshtein, Token Jaccard/Overlap, Address component match, Length deltas)
       │
       ▼
5. Matching & Classification Model (Fast baseline -> LightGBM / CatBoost pairwise ranker)
       │
       ▼
6. Threshold Calibration & Post-Processing (F0.5 optimization, Disjoint cluster assignment)
       │
       ▼
Final Submission TSV (source1_entity_id \t matched_entity_ids)
```

---

## 2. Iterative Experiment Roadmap

### Phase 1: Baseline & Core Infrastructure (Current)
- **EXP-001 (Deterministic Baseline)**:
  - Exact normalized name + address matching within country partition.
  - Generates verified end-to-end submission pipeline and establishes the lower-bound reference $F_{0.5}$ score.
  - Target: Fast execution, 100% precision on exact matches, baseline score recorded.

### Phase 2: High-Recall Multi-Channel Blocking
- **EXP-002 (Inverted Index & Token-Level Blocking)**:
  - Inverted index on salient name tokens and address tokens.
  - Character 3-gram MinHash LSH for fuzzy variations and noisy OCR records.
  - Measure Blocking Recall@K ($K=10, 20, 50$) and pair reduction ratio.

### Phase 3: Machine Learned Pairwise Matcher
- **EXP-003 (Feature Extraction & Gradient Boosted Decision Trees)**:
  - Extract dense similarity features:
    - Name: Jaro-Winkler, Monge-Elkan, Levenshtein ratio, Token Sort ratio, Token Set ratio, Prefix match.
    - Address: Number matching, street token overlap, postal code match (French / US / India), missing address handling.
    - Global: String length differences, character non-ASCII ratio, candidate rank.
  - Train LightGBM classifier with early stopping on validation split.
  - Calibrate decision threshold explicitly to maximize $F_{0.5}$.

### Phase 4: Domain Adaptation & Post-Processing Optimization
- **EXP-004 (French Domain Generalization & Disjoint Assignment)**:
  - Benchmark on multilingual tokens / diacritics.
  - Apply greedy bipartite maximum-weight matching enforcing 0 multi-participation on $S_2$ and $S_3$.
  - Generate full test submission with strict validation.

---

## 3. Experiment Log & Benchmark Results

| Exp ID | Hypothesis | Changes | Blocking Recall | Val Prec | Val Recall | Val Macro $F_{0.5}$ | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| **EXP-000** | Predict Nothing Baseline | All empty sets | 0.0% | 0.00% | 0.00% | **0.0559** | Complete | Official lower-bound singleton score |
| **Baseline 1** | Exact Name Matching | Exact name block | 25.5% | 11.08% | 25.51% | **0.3576** | Complete | Catastrophic FP rate (89% FPs) |
| **Baseline 2** | Exact Name + Numeric | Exact name & shared digits | 16.3% | 98.14% | 16.26% | **0.3421** | Complete | Ultra-high precision, low recall |
| **Baseline 3** | Strong Name + Address | Name $\ge 0.6$ & Addr $\ge 0.5$ | 36.3% | 89.50% | 36.27% | **0.5475** | Complete | High precision rule baseline |
| **EXP-001** | Multi-Channel Tiered Scorer | 6-channel blocking + Tiered rules | **71.75%** | 72.87% | **63.51%** | **0.6427** | Complete | Strong multi-match baseline |
| **EXP-002** | Per-S1 Macro-Aware Selection | Policy C: Dynamic Ranked Window | **71.75%** | **79.69%** | **61.21%** | **0.6683** | Complete | US: **0.6908**, India: **0.6345** |
| **EXP-003** | Calibrated Hybrid LightGBM Ranker | GBDT + Rule Gating + Disjoint Cluster | **71.78%** | **84.48%** | **61.25%** | **0.6881** | Complete | US: **0.6989**, India: **0.6720** |
| **EXP-004** | Trigram MinHash LSH + Multilingual | Cross-lingual blocking + French generalization | Target $>85\%$ | Target $>85\%$ | Target $>75\%$ | Target $>0.75$ | Next | Recall scaling & Domain Adaptation |
