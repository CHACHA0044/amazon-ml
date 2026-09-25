# Problem Understanding: Amazon ML Challenge 2026

## 1. Executive Summary & Problem Formulation

### 1.1 The Real-World Task
The Amazon ML Challenge 2026 addresses large-scale **Entity Resolution (Record Linkage / Deduplication)** across heterogeneous business data catalogs. 
E-commerce and marketplace operations aggregate merchant records, suppliers, physical stores, and commercial partners from disparate source registries (Source 1, Source 2, Source 3). Due to variations in spelling, transliterations, typographical errors, formatting styles, missing fields, and noisy OCR/character encodings, identical real-world business entities appear with divergent names and addresses across catalogs.

The objective is to accurately map every query entity in **Source 1** to its corresponding matching entities in **Source 2** and **Source 3**.

### 1.2 Input-Output Specification

- **Input Registries:**
  - `source1.tsv`: Primary reference catalog containing query entities ($S_1$) with attributes `[entity_id, business_name, business_address, country]`.
  - `source2.tsv`: Candidate catalog 2 ($S_2$) with attributes `[entity_id, business_name, business_address, country]`.
  - `source3.tsv`: Candidate catalog 3 ($S_3$) with attributes `[entity_id, business_name, business_address, country]`.

- **Target Output:**
  - For every entity in `test_source1.tsv` ($N = 1,732,544$), predict the comma-separated list of matching entity IDs from $S_2$ and $S_3$.
  - Output format: TSV file `submission.tsv` with header `source1_entity_id\tmatched_entity_ids`.
  - For entities with zero matches (singletons), the output must be empty/blank after the tab delimiter (`S1-xxxx\t`).

### 1.3 Key Competition Constraints & Rules
1. **Zero External Data**: External APIs, web scrapers, geocoding engines, Google Maps, OpenStreetMap, LLM APIs, and pre-built commercial entity databases are strictly prohibited.
2. **Deterministic & Self-Contained**: All feature extraction, blocking, candidate retrieval, and inference must operate strictly on the supplied offline data and locally hosted models/libraries.
3. **Scale & Throughput**: Test dataset contains $1.73\text{M}$ query records ($S_1$), $4.89\text{M}$ records in $S_2$, and $5.08\text{M}$ records in $S_3$. Total test candidate space is $>1.72 \times 10^{13}$ pairs, demanding strict multi-stage indexing, blocking, and efficient ranking architectures.

---

## 2. Evaluation Metric: Macro-Averaged $F_{0.5}$ Score (CORRECTED)

> **CORRECTION (verified against official source)**: An earlier draft of this doc (and of `validation_strategy.md` / `src/evaluation/metrics.py`) described the score as **micro-averaged** over all pairs. That contradicted the authoritative `problem_statement.txt`, which explicitly states the score is a **macro-average**: F_0.5 is calculated **per Source 1 entity**, then averaged across **all** Source 1 entities. `src/evaluation/metrics.py` has been fixed to implement the official macro definition (`compute_f05_score`), with the micro version retained as `compute_f05_micro` for comparison only. Verified numerically against the worked example in the problem statement (0.714).

### 2.1 Mathematical Formulation (Official)
For each Source 1 entity, compute per-entity Precision and Recall over that entity's predicted and true match sets:

$$\text{Precision}_e = \frac{|\text{pred}_e \cap \text{true}_e|}{|\text{pred}_e|}, \qquad \text{Recall}_e = \frac{|\text{pred}_e \cap \text{true}_e|}{|\text{true}_e|}$$

$$F_{0.5,e} = \frac{1.25 \cdot \text{Precision}_e \cdot \text{Recall}_e}{0.25 \cdot \text{Precision}_e + \text{Recall}_e}$$

Then the official score is the unweighted mean over all S1 entities:

$$F_{0.5} = \frac{1}{N_{S1}} \sum_{e=1}^{N_{S1}} F_{0.5,e}$$

**Singleton conventions (from the spec):**
- `pred_e` empty, `true_e` empty → `F_{0.5,e} = 1.0` (correctly predicted singleton)
- `pred_e` non-empty, `true_e` empty → `0.0` (false merge on a singleton)
- `pred_e` empty, `true_e` non-empty → `0.0` (missed matches)

### 2.2 Strategic Implications of $F_{0.5}$ (Macro)
- **Every S1 entity contributes equally**, regardless of match cardinality. High-cardinality clusters do NOT dominate the score the way they do under micro-averaging.
- **False Positive Penalty (per-entity)**: On an entity with few true matches, one extra predicted ID sharply lowers that entity's `F_{0.5,e}`. E.g. true=1, pred=2 → `F_{0.5,e} ≈ 0.714`. FP on singleton entities is a full 1.0 → 0.0 swing.
- **High-Precision Thresholding**: Because the mean is dominated by entities (most have 2–6 true matches), a thresholding scheme must be precision-safe at the per-entity level, and singletons (5.585% × 1.0 credit) must be deliberately protected.
- **Macro vs Micro divergence was measured on real training GT**: e.g. "oracle + 1 extra FP per entity" → macro 0.8074 vs micro 0.8209; "drop 1 true match per multi-match entity" → macro 0.9250 vs micro 0.9352. Micro systematically inflates scores relative to the official metric.

---

## 3. Structural Properties & Ground Truth Topology

From the verified audit of `train_ground_truth.tsv` (2,206,821 records):

| Property | Value | Percentage / Note |
|---|---|---|
| Total $S_1$ Query Entities | 2,206,821 | 100% |
| Singletons (0 matches) | 123,247 | 5.585% |
| Exactly 1 Match | 119,157 | 5.400% |
| Multi-Match ( $\ge 2$ matches) | 1,964,417 | 89.015% |
| Matches in Both $S_2$ and $S_3$ | 1,776,047 | 80.480% |
| Total Pairwise Matches | 7,638,365 | Mean: 3.46 matches / $S_1$ |
| Max Matches for single $S_1$ | 11 | (Max $S_2$: 5, Max $S_3$: 6) |
| $S_2$ Multi-Participation | **0** | Every $S_2$ record maps to $\le 1$ $S_1$ record |
| $S_3$ Multi-Participation | **0** | Every $S_3$ record maps to $\le 1$ $S_1$ record |

### Crucial Insight: Disjoint Cluster / 1-to-N Rooted Star Graph
The empirical multi-participation count for $S_2$ and $S_3$ is exactly **zero**. This proves that the underlying data-generating process forms disjoint star clusters rooted at each real-world business entity. A candidate $S_2$ or $S_3$ entity cannot belong to multiple $S_1$ entities simultaneously. This enables post-processing optimization via mutually exclusive assignment (e.g. bipartite matching / greedy conflict resolution).

---

## 4. Domain & Distribution Shifts: The France Anomaly

### 4.1 Country Distribution Shift
- **Training Set**: 
  - `US`: ~60.0%
  - `India`: ~40.0%
  - `France`: **0.0% (Completely absent)**
- **Test Set**: 
  - `India`: ~46.8%
  - `US`: ~38.3%
  - `France`: **14.97% (259,452 query records in $S_1$, ~703k in $S_2$, ~731k in $S_3$)**

### 4.2 Handling the Unseen French Domain
Models trained solely on US and Indian lexical patterns, street terms ("Rd", "St", "Nagar", "Marg"), and character n-grams could degrade on French addresses ("Rue", "Boulevard", "Avenue", "Allée", "Cedex", postal codes).
**Solution Architecture:**
1. Language-agnostic string normalization and unicode diacritics decomposition (`NFKD` normalization converting `é -> e`, `ç -> c`, etc.).
2. Country-partitioned blocking (candidates are strictly partitioned within the same country).
3. Cross-lingual / character n-gram TF-IDF and MinHash LSH representations that generalize seamlessly to French naming conventions without requiring French training labels.
