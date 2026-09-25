# Deep Data Discovery — Amazon ML Challenge 2026 Entity Resolution

Second, targeted investigation on top of the earlier `dataset_analysis.md` /
`validation_strategy.md`. All numbers below are computed on real data
(`work/*.parquet`, produced by `scripts/discovery/*`) and are reproducible.
Numbers from sampled pools are flagged as such. Raw `dataset/` files were
touched **only in read mode**.

---

## 0. Scope and reproducibility

Artifacts produced and where they live:

| artifact | location |
|---|---|
| preprocessed records | `work/train_records.parquet`, `work/test_records.parquet` |
| ground truth long | `work/gt_long.parquet` (s1_id, matched_id, msrc) |
| all positive pairs | `work/pos_pairs_full.parquet` (7,638,365 rows) |
| pair pool (600k pos sample + 821k sampled neg, 4 channels) | `work/pair_pool.parquet` |
| analysis JSONs | `work/analysis/01_..13_*.json` |
| scripts | `scripts/discovery/01_pair_similarity.py` … `13_splitcheck.py` |

Official metric (re-verified): **macro-average F0.5 over Source-1 entities**,
each S1 is one unit, singletons included (empty vs empty → 1.0; any prediction
on a true-empty S1 → 0.0). Implementation: `src/evaluation/metrics.py`
(`compute_f05_score`) matches the problem statement example (0.7143). A
"predict nothing" baseline scores **0.0558** (== singleton fraction 123,247 /
2,206,821).

---

## 1. Ground-truth structure

- GT = **7,638,365** pairs across **2,083,574** matched S1; **123,247** singleton S1 (5.59%).
- Matches per S1: mean **3.67**, median **4**, p90 6, p99 8, max **11**.
  Histogram: 1→119k, 2→375k, 3→531k, 4→484k, 5→322k, 6→165k, 7→64k, 8→19k, 9→4k, 10+/11→0.6k. **>50% of matched entities have ≥4 matches** → recall cannot be "top-1".
- Source mix: S2-only 143k, S3-only 164k, **both 1,776k** S1.
- **Distractors:** ~26.6% of S2 (1,340,997) and ~25.4% of S3 (1,340,857) records appear in **zero** GT pairs → ~2.68M unlinkable candidate records. A candidate that "looks plausible" is often genuinely noise.
- Each candidate (S2/S3) id appears at most once in GT (matched_id unique per msrc).

---

## 2. Metric — what it means operationally

- Baseline (predict nothing) F0.5 = **0.0558**; a per-entity "oracle + 1 FP
  everywhere = 0.8074"; dropping 1 true match per entity = **0.9250**
  (0.5-weighted precision + high cardinality floors). So each missed match hurts
  ~0.07-0.07 points, each extra FP on a singleton or low-precision entity drives
  to 0.
- Consequence: optimise **per-entity**, prefer precise acceptance high-signal
  pairs, and be very conservative about predicting anything for low-evidence S1.

---

## 3. Corruption process on true matches (02_corruption.json, 13% sample)

Name differences S1 vs matched S2/S3 (per-pair types, exclusive):

| name type | share | notes |
|---|---|---|
| exact copy | 4.6% | byte-identical |
| case only | 6.1% | |
| punctuation only | 15.0% | |
| corp suffix only | 2.8% | Ltd/Pvt/LLC/Inc/SARL churn |
| same tokens, order changed | 7.1% | | 
| token subset (words added/removed) | 27.2% | e.g. "Studio" → "The Studio & Salon" |
| heavy (renames/edits/combos) | 37.2% | mean name token-Jaccard 0.32 |

Address differences:

| addr type | share | notes |
|---|---|---|
| exact | 2.2% | |
| case-only | 5.0% | |
| norm-map-only (abbrev/punct) | 1.0% | |
| order | 4.3% | |
| token subset | 10.9% | |
| missing (one side empty) | 4.4% | both empty ≈ 0 |
| heavy | 72.0% | mean leftover addr token-Jaccard **0.53** |

Key points:
- **~37% of true pairs have substantially rewritten names** (India heavier:
  heavy-bucket token-Jaccard 0.25 vs US 0.38). Exact-name matching alone tops
  out at ~29% recall.
- **Addresses mutate the most textually but keep token overlap** (0.53) — the
  primary fuzzy signal.
- India has far more corporate-suffix churn (22.5k vs 5.6k samples) and more
  reorder — Indian names are the least reliable.

---

## 4. Feature discrimination (01_pair_sim_summary.json — pool: 600k pos + 821k neg)

AUC on the mixed pool (positives + name-block/addr-block/rare-token/random negatives):

| feature | pos mean | neg mean | AUC |
|---|---|---|---|
| addr token Jaccard | 0.60 | 0.11 | **0.89** |
| addr contain (either dir) | 0.71-0.77 | 0.12-0.16 | **0.89** |
| addr char3 | 0.64 | 0.12 | **0.88** |
| addr char2 | 0.70 | 0.19 | **0.87** |
| numeric-token overlap | 0.86 | 0.11 | **0.79** |
| name char2 | 0.72 | 0.43 | 0.67 |
| name token Jaccard | 0.66 | 0.41 | 0.66 |
| rare shared name tokens | 0.76 | 0.41 | 0.61 |
| name eq (norm) | 0.29 | 0.31 | 0.49 (no signal) |

Probes (pool):
- name eq **&** addr-char2≥0.5 → 99.9% positive (144.6k pool pairs).
- name eq alone → only 40.6% positive — names alone are **not** discriminating.
- num overlap ≥1 → 85.6% positive.
- Both name+addr strongly confusable negatives are rare (**656** / 821k sampled).

Takeaway hierarchy: **address → numeric tokens → name**, with rare-token-name
as a weaker auxiliary channel.

---

## 5. Ambiguity / hard-negative analysis (03_ambiguity.json)

Among same-name (norm-equal) candidates in the pool (423.6k), **40.6% are
negatives** (in-sample). But among those same-name *negative* candidates:
- addr token Jaccard ≥ 0.5: **0.03%**
- share any numeric token: **0.28%**
- addr char2 ≥ 0.6: **0.05%**

So: same-name candidates are plentiful and ambiguous (this drives FP risk), but
the *decision* is nearly decidable by the address. Among same-address negatives
(71.3k), only **1.4%** have name-token Jaccard ≥ 0.5. Random negatives have
essentially no name/addr/numeric surface.

→ High-signal (name-strong AND addr-strong) pairs are matches almost surely;
the failure mode to engineer for is *"name-strong, address-weak"* collisions.

---

## 6. Evidence hierarchy on positives (04_evidence.json)

On the 600k positive sample:

- **addr_strong** (addr tok-jac ≥ 0.5 **OR** shared numeric token): covers
  **83.3%** of positives.
- **numeric** (≥1 shared digit token): 66.1%.
- **name_strong** (name tok-jac ≥ 0.8 AND containment ≥ 0.8): 40.1%.
- Overlap = 32.9%; name-only-strong 7.2%; addr-only-strong **50.5%**; neither **9.5%**.
- When **name is weak** (59.9% of positives), address still rescues 84.2%, numeric
  67.0%, addr-char2≥0.6 76.1%.
- When **address is weak** (16.7%), name alone rescues only 43.2%.

Conclusion: address (+numbers) is the backbone; name is a tie-breaker and a
single strong-name match must be checked against address.

---

## 7. Address decomposition (05_address.json)

- **US:** 92.4% of addresses start with a house-number-like token; 15.4% have a
  5-digit trailing token; median 6 tokens/address.
- **India:** 90.8% start numeric; 12.0% have 6-digit pincode, 13.8% a 5-digit
  trailing token; median **11 tokens**/address (long locality chains).
- On matched pairs (both have digit tokens, 87.2% of pairs):
  - first numeric token (house/plot #) agrees **54.8%**
  - last numeric token (postal) agrees **57.1%**
  → digits are themselves corrupted in a meaningful fraction of matches; agreement
  on them is therefore a *high-precision* but not high-recall signal (still, the
  strongest single number: 66% of positives share ≥1 digit token at all).

---

## 8. Name structure (06_name.json)

- Token counts: India mean 3.70 (median 4, p90 5); US mean 3.49 (median 3, p90 5).
- India top tokens: ltd (668k), pvt (554k), india, llp, co, services, solutions,
  brothers, trading, technologies, international, foundation, corp, global, tech,
  industries, enterprises, consultants, technology, consultancy, developers,
  ventures, traders, care, marketing.
- US top tokens: llc (356k), inc (238k), and, s, c, l, care, of, associates,
  corp, center, group, partners, p …
- Common **last tokens**: India ltd/llp/co/corp/trust/group/society; US
  llc/inc/c/corp/group/pc/pllc/lp.
- Implication: suffixes + generic business words carry ~no signal → strip/map
  them (already in `normalize_name` suffix map) and weight rare/mid tokens in
  name similarity; "physical therapy", "mumbai india pvt ltd", "cc" behave like
  stopword blocks.

---

## 9. Country difference (07_08.json)

| | India | US |
|---|---|---|
| name token-jac AUC | 0.57 | 0.72 |
| name char2 AUC | 0.58 | 0.74 |
| addr token-jac AUC | **0.92** | 0.87 |
| addr char2 AUC | 0.90 | 0.85 |
| numeric overlap AUC | **0.81** | 0.77 |
| rare-shared AUC | 0.60 | 0.61 |

India: names weaker (heavier rewriting), address+numbers stronger.
→ country-specific thresholds/weights are justified.

---

## 10. Domain shift / France (07_08.json)

- Test: S1 India 809,986 / US 663,106 / **France 259,452**; S2 4,887,273; S3 5,082,316.
- **Only 9.4% of test name-blocks and 23.5% of test name tokens appeared in
  train** → most test entities are new; France is wholly novel.
- Rare-token statistics learned on train will cover only ~a quarter of test
  tokens; name-block exact-matching will miss most matches on test unless
  combined with fuzzy/address/numeric channels. Treat train vocab as
  illustrative, not exhaustive.

---

## 11. Blocking / candidate-generation reality (10_blocking.json)

- Name block (norm, within country): group p50 = 1, p90 = 3, p99 = 9,
  max 1072 ("physical therapy"); 60.6k blocks >10.
- Address block: mostly singletons (9.2M of 10.5M groups), p99 = 3; **empty
  address forms a giant group (222k)** — must drop empties before blocking.
- Digit-token blocking explodes: "2nd" → 145k records, etc. Numeric overlap must
  be applied on *pairs*, not as a block key for common digits; zips/pincodes as
  keys OK after length checks.
- **Coverage floor:** only ~1.04M of 2.08M matched S1 have *any* candidate
  sharing their exact name block (India 367k / US 672k of 2.08M). A single
  exact-name block channel would cap recall near ~50% of *matches*, and much
  lower for S1 coverage. Multi-channel generation (name block + rare token +
  numeric + address block + fuzzy) is required.

---

## 12. Deterministic-rule coverage sanity — official macro F0.5 on pool (09_rules.json)

Per-rule recall (fraction of positives), negative-candidate hit rate, and
macro F0.5 computed over covered S1 (pool-bounded; ordering not absolutes):

| rule | pos recall | neg-hit rate | F0.5 (covered S1) |
|---|---|---|---|
| R1 name norm-equal | 28.6% | 30.7% | 0.228 |
| R2 name eq & addr eq | 1.7% | 0.0% | 0.037 (tiny recall) |
| R3 name eq & num overlap | 18.5% | 0.09% | 0.339 |
| R6 name≥0.6 & addr≥0.5 | 45.0% | 0.29% | **0.652** |
| R7 addr≥0.4 & num | 57.5% | 7.9% | 0.613 |
| R8 addr-char2≥0.8 | 41.3% | 8.8% | 0.501 |
| R9 num≥1 & name≥0.5 | 51.7% | 0.35% | **0.678** |
| R11 rare-token & addr≥0.3 | 46.6% | 0.81% | 0.531 |
| R10 any-strong (loose union) | 86.3% | 39.9% | 0.400 (FP blow-up) |

Insight: crisp **conjunctions** of name+address or name+number dominate loose
unions; false positives come from name-alone or fuzzy-addr-alone acceptance.
The realistic best-of-combos here ≈ 0.68 F0.5 *on the pool's covered S1*; a
learner with proper thresholds and per-S1 selection is the main path past this.

---

## 13. Match cardinality (11_cardinality.json)

See §1. Extra detail: unmatched candidate population (26.6% S2 / 25.4% S3) is
the precision enemy; any accepted pair to an "unmatched" record is an FP by
definition.

---

## 14. Cross-source corroboration (12_crosssource.json)

For S1 with both S2 and S3 matches (1,733,536 S1, 4.4M cross-pairs):
- Same normalized name: **14.4%** of cross-pairs; at-least-one agreement per S1:
  **35.4%**; full (name+addr) agreement: **0.06%**.

→ S2 and S3 usually write the same entity differently. Cross-source agreement
is a *rare luxury*, not a fallback. Do not condition a model on "S2 and S3
separately confirmed".

---

## 15. Validation hygiene (13_splitcheck.json)

- No pre-built train/valid/test split exists in the repo (only docs references).
- S1 ids: train vs test overlap **0**; no GT candidate id leaks into test.
- Recommendation: build CV on **S1 entities** (group positives per S1, split by
  S1, never by pairs); because macro is per-entity, keep per-S1 match counts
  balanced across folds.

---

## 16. Top discoveries (numbered)

1. **Official metric is macro per-S1 F0.5**; singleton-safe policy matters; baseline 0.0558.
2. **Match cardinality is high** (median 4) — must predict multi-matches.
3. **~26% of candidate records are unlinkable noise** — precision must reject them.
4. **~37% of true pairs have rewritten names** (India worst) → exact-name recall ceiling ≈ 29%.
5. **Address is the strongest field** (tok-jac AUC 0.89); names weak/lossy.
6. **Numeric overlap is the precision anchor**: name-strong + addr-strong ⇒ ~100% match; hard both-strong negatives ≈ 656/821k.
7. **Name equality is dangerously ambiguous** (~40% negatives among same-name candidates), resolved almost exactly by address proximity.
8. **Addresses keep ~0.5 token overlap even when "heavy"** — robust fuzzy base.
9. **House-number/postal agreement ~55-57%** — high-precision, moderate-recall numeric checks.
10. **Name blocking covers only ~half of matches** (1.04M/2.08M matched S1); multi-channel generation required.
11. **Empty-address and common-digit blocking blow up** (222k / 145k groups) — exclude empties, avoid bare "2nd"-style keys.
12. **India vs US differ in signal mix** (India: names weak, address/numeric strong).
13. **Heavy train→test domain shift** (9.4% block overlap; France is new).
14. **Cross-source corroboration is rare (14%)** — don't rely on S2↔S3 agreement.
15. **Crisp conjunctive rules beat loose unions** (R9 num+name F0.5≈0.68 vs R10 union 0.40).

---

## 17. Hypotheses for EXP-001 / EXP-002

- **EXP-001 (blocking + crisp scoring).** Candidate generation: (a) exact name
  block (country-scoped, suffix-mapped), (b) shared rare/medium name token
  (df-bounded), (c) shared numeric token (house#/postal, length-checked),
  (d) exact address block (empty addresses excluded). Score candidates with a
  small calibrated stacker over {addr token-jac, addr char2/3, name char2,
  name token-jac, numeric overlap, rare-shared, length diffs} trained
  per-country; accept on **conjunctive high-precision rules first, address-anchored
  thresholds second, name-only rarely**. Hypothesis: reaches >0.6 macro F0.5 on
  matched-S1 sub-population while keeping singleton FPs ≈ 0.
- **EXP-002 (per-entity selection).** Greedy per-S1 candidate selection with
  macro-aware objectives: (i) high-precision anchor tier (name&addr strong,
  or numeric+addr) predicting all matches found, (ii) medium tier (addr-strong),
  (iii) optional low tier predicted only when entity otherwise has 0 predictions
  (protects recall on multi-match entities without risking singleton FPs).
  Punish guesses on low-evidence S1 (target ≈ predict-nothing for nothing-like).
- **Checklist before build:** final France/Indian/US threshold split; whether to
  add French suffix/abbr maps (SARL/SAS/SARL maps already present); minhash-LSH
  for fuzzy name channel vs brute force on small blocks; sample doubling for
  hard-negative coverage of the addr-strong tier.

## 18. Risks & open questions for a human

- Pooled F0.5 numbers are orderings, not absolutes (600k/7.6M positive sample;
  4 negative channels). Full-system eval needed.
- France: no train entities; name/address vocab shift; French-language business
  suffixes and punctuation (accent stripping already handled).
- ~26% noise candidates make "high recall at any cost" dangerous under F0.5.
- missing GT coverage might itself be incomplete for the very ambiguous records.
- Decide: per-country models vs single model with country features; whether low
  evidence ⇒ predict nothing is acceptable given singletons count 1.0 anyway.