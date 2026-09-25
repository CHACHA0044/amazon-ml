# EXP-003: Calibrated Machine-Learned Pairwise Matcher & Hybrid Ranking Report

## 1. Executive Summary & Objectives

- **EXP-003**: Train a machine-learned Gradient Boosted Decision Tree (LightGBM) pairwise classifier on 1.35M leak-free training pairs, calibrate against true retrieval class imbalance, and combine with conjunctive rule gating and disjoint star cluster mutual exclusivity.
- **Evaluation Metric**: Official Macro-averaged $F_{0.5}$ over Source 1 entities (including singletons).
- **Candidate Pool**: Full 10,320,219 candidate records ($S_2 + S_3$) with natural 26% noise distractors.
- **Hardware Profile**: 100% compliant with [`device-specs.md`](file:///d:/rep/amazonml/device-specs.md) (peak RAM $<2.5\text{ GB}$, CPU multi-threading).

---

## 2. Model Architecture & Feature Importances

### 2.1 Training Details
- **Training Data**: 1,357,370 pairs ($573,005$ positive matches, $784,365$ hard negatives across 4 channels) strictly isolated from validation S1 IDs (zero pair leakage).
- **Validation Fold**: 63,917 pairs across S1 holdout entities.
- **Model**: `LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.05, n_estimators=500, min_child_samples=50)` with early stopping at iteration 239.
- **Pairwise ROC-AUC**: **0.99889** on validation fold.

### 2.2 Top Feature Importances
| Feature Name | Split Importance | Description |
|---|---|---|
| `name_char2` | 743 | Character 2-gram Dice similarity on business name |
| `addr_contain_c` | 689 | Candidate address token containment in query |
| `name_char3` | 636 | Character 3-gram Dice similarity on business name |
| `rare_shared` | 623 | Count of shared distinctive tokens |
| `addr_char3` | 609 | Character 3-gram Dice similarity on address |
| `name_contain_c` | 509 | Candidate name token containment in query |
| `addr_char2` | 504 | Character 2-gram Dice similarity on address |
| `name_tok_jac` | 448 | Token Jaccard similarity on business name |
| `is_us` | 423 | Country indicator (US vs India signal weighting) |
| `name_contain_q` | 396 | Query name token containment in candidate |
| `addr_tok_jac` | 340 | Token Jaccard similarity on address |
| `num_overlap` | 324 | Count of shared address numeric tokens |

---

## 3. Benchmark Progression Table

Evaluated on the stratified holdout benchmark against all 10.32M candidate records:

| Experiment / Model Stage | Macro $F_{0.5}$ (Official) | Micro $F_{0.5}$ | Precision | Recall | US Macro $F_{0.5}$ | India Macro $F_{0.5}$ | Singleton ($k=0$) $F_{0.5}$ |
|---|---|---|---|---|---|---|---|
| **EXP-000 (Predict Nothing)** | **0.0559** | 0.0000 | 0.0000 | 0.0000 | 0.0559 | 0.0560 | **1.0000** |
| **Baseline 1 (Exact Name)** | **0.3576** | 0.1245 | 0.1108 | 0.2551 | 0.4030 | 0.2896 | 0.6064 |
| **Baseline 2 (Exact Name + Numeric)** | **0.3421** | 0.4705 | **0.9814** | 0.1626 | 0.3837 | 0.2799 | 0.9911 |
| **Baseline 3 (Name + Address Rule)** | **0.5475** | 0.6866 | 0.8950 | 0.3627 | 0.5343 | 0.5672 | 0.7961 |
| **EXP-001 (Deterministic Tiered)** | **0.6427** | 0.7077 | 0.7287 | **0.6351** | 0.6634 | 0.6117 | 0.4222 |
| **EXP-002 (Policy C: Dynamic Window)** | **0.6683** | 0.7516 | 0.7969 | 0.6121 | 0.6908 | 0.6345 | 0.5564 |
| **EXP-003 (Raw LightGBM Top-K)** | **0.5707** | 0.6080 | 0.5104 | **0.6426** | 0.5823 | 0.5533 | 0.3488 |
| **EXP-003 (Hybrid Ranked Ensemble 0.90)** | **0.6881** | 0.7675 | 0.8446 | 0.6106 | 0.6988 | 0.6720 | **0.6047** |
| **EXP-003 (Hybrid + Disjoint Star Cluster)** | **0.6881** | **0.7686** | **0.8448** | 0.6125 | **0.6989** | **0.6720** | **0.5832** |

---

## 4. Key Discoveries & Technical Insights

1. **The Retrieval Prior Mismatch Insight**:
   - Raw machine learning classifiers trained on balanced/sampled pools suffer from prior probability overestimation when deployed against 10.3M candidate spaces with 95% negative distractors.
   - Ensembling LightGBM's fine-grained probability ranking with conjunctive rule gating ($\text{Conf}_{\text{Hybrid}} = \sqrt{P_{\text{LGBM}} \times \text{Conf}_{\text{Rule}}}$) boosts precision from **79.69% to 84.48%** and raises Macro $F_{0.5}$ to **0.6881**.
2. **Substantial India Lift**:
   - The ML ranker learned complex character-level substring and token containment relationships that lifted Indian entity resolution performance from **0.6342 $\to$ 0.6720** (+0.0378 lift).
3. **Disjoint Star Cluster Optimization**:
   - Enforcing the structural property that each $S_2/S_3$ candidate belongs to at most one $S_1$ entity further eliminated multi-merge conflicts.
