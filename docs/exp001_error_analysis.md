# Error Analysis & Diagnosis: EXP-001 & EXP-002

## 1. Executive Summary

This error analysis examines the empirical false positive (FP) and false negative (FN) failure modes observed during the evaluation of 10,001 stratified validation queries against the 10.32 million candidate dataset.

---

## 2. Major False Positive (FP) Patterns & Root Causes

### Error Pattern A: Generic Business Names with Locality False Matches
- **Example Case**: `S1-363533908` ("corner hair studio", address "1028 42nd street il washington park") produced 5 false positive candidate matches across other generic salons.
- **Root Cause**: The name consists of low-information generic tokens ("corner", "hair", "studio"). When address street numbers or short tokens partially match by coincidence in dense cities, the deterministic rule gave excessive weight to the name equality.
- **Remedy for EXP-003**: 
  - Downweight name similarity inversely by total token frequency (TF-IDF weighted name similarity).
  - Impose an adaptive address similarity threshold ($>0.75$) when name character length is short ($\le 15$ characters) or generic.

### Error Pattern B: Shared Corporate Brand Names in Different Cities
- **Example Case**: `S1-411457951` ("hunter sumter mcdonnell chicago group") matched candidates in Texas with similar corporate names.
- **Root Cause**: Suffix normalization stripped "chicago group", leading to an exact name match on the core entity name, while the city token "chicago" was not recognized as a city constraint.
- **Remedy for EXP-003**:
  - Add city/state extraction and mismatch penalty features.

### Error Pattern C: Singleton False Merges
- **Example Case**: `S1-192033397` (True Singleton) matched candidate `S2-524824680`.
- **Root Cause**: The entity had no true matches in the candidate pool, but a distractor candidate had partial fuzzy address and numeric agreement. Under macro $F_{0.5}$, this turned a $1.0$ score into $0.0$.
- **Remedy for EXP-003**:
  - Calibrate an explicit **Singleton Confidence Gate**: if maximum pair confidence for an S1 entity is below $0.85$, predict an empty match set.

---

## 3. Major False Negative (FN) Patterns & Root Causes

### Error Pattern D: Heavily Rewritten Names Missed in Candidate Blocking
- **Example Case**: `S1-481270277` ("classic logistics pvt ltd" vs true candidates with names completely rephrased or missing common tokens).
- **Root Cause**: Candidate generation blocking recall is currently **71.75%**. The remaining 28.25% of positive pairs were never retrieved by the 6 candidate channels because both name and address were heavily corrupted simultaneously.
- **Remedy for EXP-003**:
  - Add character 3-gram MinHash LSH blocking channel on name prefixes to retrieve fuzzy name matches with zero exact token overlap.
  - Add street number + city/state composite blocking keys.

### Error Pattern E: Missing Candidate Addresses
- **Example Case**: True positive candidates with `addr_missing == 1` (~3.3% of S2/S3 records).
- **Root Cause**: When the candidate address is null, address similarity is 0. If the name is even slightly modified (e.g. token subset), deterministic rules cautiously reject it to avoid false positives.
- **Remedy for EXP-003**:
  - Implement a dedicated "Missing Address" feature branch in the machine-learned ranker (LightGBM) trained specifically on name-distinctiveness features.

---

## 4. Prioritized Action Plan for EXP-003 (Machine-Learned Ranker)

| Priority | Targeted Bottleneck | Planned Solution | Expected Score Lift |
|---|---|---|---|
| **1** | Blocking Recall Ceiling (71.75%) | Trigram MinHash LSH + City/Pincode Composite Blocking | +0.05 to +0.08 $F_{0.5}$ |
| **2** | Generic Name False Positives | TF-IDF Token Weighting & City Mismatch Penalty | +0.03 to +0.05 $F_{0.5}$ |
| **3** | Machine-Learned Scoring (LightGBM) | Gradient Boosted Decision Tree on 24 pairwise features | +0.04 to +0.07 $F_{0.5}$ |
| **4** | Singleton Shield & Post-Processing | Mutual exclusivity & Bipartite matching (0 multi-participation) | +0.02 to +0.03 $F_{0.5}$ |
