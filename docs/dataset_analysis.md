# Dataset Analysis & Audit Report: Amazon ML Challenge 2026

## 1. Dataset Inventory & Integrity Audit

The raw dataset was comprehensively audited using chunked, memory-safe streaming scanners. Below are the verified metrics across all 7 TSV files:

| File Name | File Role | Size on Disk | Total Rows | Unique IDs | Duplicate IDs | ID Prefix Check |
|---|---|---|---|---|---|---|
| `train/train_source1.tsv` | Train Query ($S_1$) | 200.3 MB | 2,206,821 | 2,206,821 | 0 | 100% `S1-` |
| `train/train_source2.tsv` | Train Candidates ($S_2$) | 466.6 MB | 5,034,616 | 5,034,616 | 0 | 100% `S2-` |
| `train/train_source3.tsv` | Train Candidates ($S_3$) | 480.4 MB | 5,285,603 | 5,285,603 | 0 | 100% `S3-` |
| `train/train_ground_truth.tsv` | Train Labels | 121.1 MB | 2,206,821 | 2,206,821 | 0 | Matched $S_1$ Keys |
| `test/test_source1.tsv` | Test Query ($S_1$) | 166.9 MB | 1,732,544 | 1,732,544 | 0 | 100% `S1-` |
| `test/test_source2.tsv` | Test Candidates ($S_2$) | 485.9 MB | 4,887,273 | 4,887,273 | 0 | 100% `S2-` |
| `test/test_source3.tsv` | Test Candidates ($S_3$) | 482.6 MB | 5,082,316 | 5,082,316 | 0 | 100% `S3-` |

---

## 2. Missing Value Analysis

| Split & Table | `entity_id` Nulls | `business_name` Nulls | `business_address` Nulls | `country` Nulls | Missing % (`business_address`) |
|---|---|---|---|---|---|
| `train_source1` | 0 | 0 | 0 | 0 | **0.000%** |
| `train_source2` | 0 | 0 | 168,967 | 0 | **3.356%** |
| `train_source3` | 0 | 0 | 175,916 | 0 | **3.328%** |
| `test_source1` | 0 | 0 | 0 | 0 | **0.000%** |
| `test_source2` | 0 | 0 | 129,408 | 0 | **2.648%** |
| `test_source3` | 0 | 0 | 136,098 | 0 | **2.678%** |

### Observations:
1. `source1` records (both train and test) have **100% complete fields** with zero missing values.
2. `source2` and `source3` have ~2.6% to 3.4% missing addresses.
3. Imputation strategy: Replace missing `business_address` with empty string `""` and generate a binary indicator flag `is_address_missing`. When address is missing, similarity scoring must rely exclusively on high-precision name matching.

---

## 3. Country & Geographic Distribution

| Country | Train $S_1$ | Train $S_2$ | Train $S_3$ | Test $S_1$ | Test $S_2$ | Test $S_3$ |
|---|---|---|---|---|---|---|
| **US** | 1,323,633 (60.0%) | 3,016,817 (59.9%) | 3,170,056 (60.0%) | 663,106 (38.3%) | 1,871,330 (38.3%) | 1,945,701 (38.3%) |
| **India** | 883,188 (40.0%) | 2,017,799 (40.1%) | 2,115,547 (40.0%) | 809,986 (46.8%) | 2,312,565 (47.3%) | 2,405,000 (47.3%) |
| **France** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **259,452 (15.0%)** | **703,378 (14.4%)** | **731,615 (14.4%)** |

### Hard Country Boundary Constraint
Across the entire training ground truth, **100% of matches occur within the same country**. There are 0 cross-country matches.
Therefore, blocking and candidate matching can be strictly partitioned by `country`, reducing the test candidate space by $>60\%$ immediately with zero recall loss:

$$\text{Search Space} = (S_{1,\text{US}} \times [S_{2,\text{US}} \cup S_{3,\text{US}}]) + (S_{1,\text{IN}} \times [S_{2,\text{IN}} \cup S_{3,\text{IN}}]) + (S_{1,\text{FR}} \times [S_{2,\text{FR}} \cup S_{3,\text{FR}}])$$

---

## 4. Text Length & Noise Profiling

| Table | Name Length (Mean / P50 / P95) | Addr Length (Mean / P50 / P95) | Non-ASCII Names (%) | Non-ASCII Addrs (%) |
|---|---|---|---|---|
| `train_source1` | 24.0 / 24 / 37 | 52.1 / 41 / 103 | 0.00% | 0.03% |
| `train_source2` | 25.1 / 25 / 40 | 47.8 / 37 / 97 | 15.19% | 9.50% |
| `train_source3` | 25.2 / 25 / 42 | 48.3 / 42 / 92 | 11.48% | 9.02% |
| `test_source1` | 23.8 / 24 / 36 | 57.2 / 50 / 105 | 2.35% | 4.26% |
| `test_source2` | 25.7 / 25 / 42 | 51.8 / 43 / 99 | 18.99% | 14.75% |
| `test_source3` | 25.7 / 25 / 42 | 50.1 / 44 / 95 | 14.51% | 14.35% |

### Text Quality Observations:
- `source2` and `source3` exhibit deliberate or natural synthetic noise (diacritics, OCR substitution, leet-speak character substitutions, unicode artifacts).
- Preprocessing must include Unicode normalization (NFKD), ASCII transliteration/folding, lowercasing, punctuation stripping, and whitespace collapse to bridge noisy variants.

---

## 5. Ground Truth Match Cardinality Distribution

The distribution of the number of matched entities per Source 1 record ($k$):

| Match Count ($k$) | Number of $S_1$ Records | Percentage of $S_1$ |
|---|---|---|
| $k = 0$ (Singletons) | 123,247 | 5.585% |
| $k = 1$ | 119,157 | 5.400% |
| $k = 2$ | 375,212 | 17.002% |
| $k = 3$ | 530,841 | 24.055% |
| $k = 4$ | 484,115 | 21.937% |
| $k = 5$ | 321,957 | 14.589% |
| $k = 6$ | 164,868 | 7.471% |
| $k = 7$ | 63,968 | 2.898% |
| $k = 8$ | 18,680 | 0.846% |
| $k = 9$ | 4,205 | 0.191% |
| $k = 10$ | 534 | 0.024% |
| $k = 11$ | 37 | 0.002% |

- **Mean Matches per $S_1$**: 3.46 ($S_2$: 1.67, $S_3$: 1.79)
- **Max Matches per $S_1$**: 11 ($S_2$: 5, $S_3$: 6)

---

## 6. Leakage & Overlap Audit

- Train vs Test $S_1$ overlap: **0**
- Train vs Test $S_2$ overlap: **0**
- Train vs Test $S_3$ overlap: **0**
- Internal cross-source ID collisions: **0**
- All IDs are pure partition keys without leakage into ground truth labels.
