# Validation Strategy: Amazon ML Challenge 2026

## 1. Core Principles & Problem Structure

In entity resolution, standard random record splits can introduce catastrophic data leakage if entities from the same true identity cluster are split across train and validation folds.
Furthermore, because Source 2 and Source 3 entities in our dataset participate in at most one Source 1 cluster (star graph topology), validation must operate at the **entity cluster level** (Query $S_1$ and its associated true candidates).

---

## 2. Validation Split Architecture

### 2.1 Cluster-Level Stratified Holdout Split
We partition the training dataset ($2,206,821$ $S_1$ records and their corresponding $S_2, S_3$ records) into:

1. **Train Fold (80%)**: ~1,765,456 $S_1$ records
2. **Validation Fold (20%)**: ~441,365 $S_1$ records

### Stratification Criteria:
- **Country proportion**: Maintain exact 60% US / 40% India split.
- **Match cardinality distribution**: Equal representation of singletons ($k=0$), moderate matches ($k=1..5$), and high-multiplicity clusters ($k=6..11$).

### Candidate Index Isolation:
- During validation, candidate indices ($S_2$ and $S_3$) for the validation fold are indexed together with a proportional background pool of distractor entities (to faithfully replicate test-time retrieval difficulty).

---

## 3. Synthetic Out-of-Domain (OOD) Validation Split

To specifically evaluate model robustness on unseen languages/countries (such as the French test set):
- **OOD Split**: Hold out a subset of non-standard Indian regions or linguistically distinct subsets as a pseudo-OOD evaluation benchmark to test zero-shot transfer of string matching, character n-grams, and TF-IDF blocking before submitting against the French test partition.

---

## 4. Evaluation Metrics Pipeline

> **CORRECTION**: The snippet below is provided for historical reference ONLY. It implements the **micro-averaged** F_0.5, which contradicts the official `problem_statement.txt`. The official metric is the **macro-average of per-Source-1-entity F_0.5** (each S1 contributes equally, singletons included). The corrected implementation lives in `src/evaluation/metrics.py` (`compute_f05_score`), which validates against the problem statement's worked example (0.714). The official macro score is what every experiment must be optimized/reported against; the micro value is reported only to quantify divergence.

```python
def compute_f05_micro(predictions_dict, ground_truth_dict):
    """Deprecated: micro-averaged F_0.5 -- NOT the official metric.
    Official = mean over S1 entities of per-entity F_0.5 (see src/evaluation/metrics.py).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ...
```

---

## 5. Candidate Generation Recall Tracking

Because a downstream classifier cannot predict matches that the retrieval/blocking stage misses, we track **Blocking Recall@K**:

$$\text{Recall@K} = \frac{|\text{Retrieved Ground Truth Pairs}|}{|\text{Total Ground Truth Pairs}|}$$

Target candidate generation criteria:
- Recall@K $\ge 98.0\%$ with $K \le 50$ candidates per $S_1$ query.
- Maintain blocking execution time under 15 minutes locally.
