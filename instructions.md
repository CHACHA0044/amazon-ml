# Amazon ML Challenge 2026 — Antigravity / Claude Working Instructions

## 0. Purpose

This file defines the operating rules for the AI coding/reasoning agent working through Antigravity on the Amazon ML Challenge 2026 project.

This is an **operating instruction file**, not the first task prompt and not the solution. The actual task-specific prompt will be supplied separately.

The goal is to build a highly competitive, scientifically rigorous ML solution while preserving reproducibility, preventing leakage, respecting competition rules, and using local/AWS compute efficiently.

---

## 1. Workspace

The project root is the local Amazon ML Challenge repository.

The workspace currently contains, or may contain:

- `dataset/` — approximately 2.3 GB of competition data
- `problem_statement.txt` — copied competition/problem information
- `amazon ml problem statement.mp4` — supplied problem-statement recording
- `aws builder center guide.pdf` — supplied AWS guidance
- `README.md`
- `.gitignore`
- Git metadata
- Potential private credential/infrastructure files

Always verify the actual contents instead of assuming them.

### Critical dataset rule

`dataset/` contains local competition data.

The agent MUST:

- never commit or push the dataset to Git
- never move, delete, overwrite, or modify original raw data in place
- avoid unnecessary full copies
- use streaming/chunked/memory-efficient inspection
- preserve the original directory structure
- store derived metadata/analysis outside the raw dataset

If any dataset file is already tracked by Git, stop and report it before destructive Git operations.

---

## 2. Security

The workspace may contain private files such as `.pem` keys.

The agent MUST:

- never read, print, summarize, upload, commit, or expose private-key contents
- never commit `.pem`, `.key`, credentials, access tokens, secret keys, `.env`, or equivalent secrets
- ensure sensitive file patterns are ignored by Git
- never hard-code AWS credentials
- never ask the user to paste secret credentials into chat
- report only authentication success/failure when credentials are involved

---

## 3. Source-of-Truth Hierarchy

When determining competition requirements, use this priority:

1. Official competition/problem-statement material supplied by Amazon/the competition platform
2. Official recording supplied with the challenge
3. Official AWS/competition documentation supplied with the project
4. Actual dataset structure and observed data
5. Repository code and experiment results
6. General ML knowledge

Do not silently replace competition-specific information with generic assumptions.

If sources conflict, identify the conflict and determine which source is authoritative. If something is unknown, explicitly mark it unknown rather than inventing it.

---

## 4. Understand Before Modeling

Do NOT immediately train a sophisticated model.

First understand:

- exact problem definition
- inputs
- outputs
- target
- evaluation metric
- train/test structure
- labels
- submission format
- restrictions
- external-data/API/model rules
- dataset schema
- row counts
- data types
- identifiers
- missing values
- duplicates
- relationships between files
- temporal/entity structure
- possible leakage

Do not generalize from a small sample without verifying against the full dataset.

---

## 5. Dataset Inspection

The dataset is approximately 2.3 GB. Do not blindly load everything into RAM.

Use appropriate tools such as:

- filesystem metadata
- chunked pandas
- Polars
- DuckDB
- Parquet metadata
- streaming
- targeted sampling

First produce a dataset inventory containing, where applicable:

- path
- extension
- file size
- row count
- columns
- data types
- likely train/test/reference role
- labels
- identifiers
- relationships to other files

Never alter raw files during this process.

---

## 6. Build a Dataset Map

Create a documented map of the actual data relationships, for example:

```text
Competition Dataset
├── Training data
├── Test data
├── Reference/master data
└── Auxiliary data
```

This is only a template. Do not assume those categories exist.

Determine how the actual files relate to each other.

---

## 7. Problem Understanding

After reading the problem statement, establish:

### Problem
What real-world problem is being solved?

### Inputs
What data is available?

### Output
Exactly what must be predicted/submitted?

### Evaluation
What metric determines performance?

### Constraints
What restrictions are explicit?

### Hidden-test implications
What can reasonably be inferred about hidden evaluation?

Clearly label inference and hypothesis; never present speculation as a rule.

---

## 8. EDA

EDA must answer questions that can change the modeling strategy.

Prioritize:

### Structure
- row/column counts
- types
- cardinality
- missingness
- uniqueness
- duplicates

### Distributions
- target distribution
- feature distributions
- skew/long tails
- rare categories
- outliers

### Relationships
- feature-target relationships
- group behavior
- entity behavior
- interactions
- conditional distributions

### Quality
- malformed values
- inconsistent formatting
- conflicting records
- suspicious identifiers
- placeholders
- anomalies

### Leakage
Investigate:

- target-derived fields
- post-outcome information
- duplicates across splits
- IDs encoding labels
- reference tables revealing answers
- temporal leakage
- preprocessing leakage
- train/test contamination

---

## 9. Validation Is Critical

Never optimize blindly against a local score.

Determine whether the data is:

- iid
- grouped
- temporal
- hierarchical
- duplicated
- imbalanced
- vulnerable to near-duplicate leakage

Consider random, stratified, group, temporal, or cross-validation schemes according to the actual data-generating process.

Do not select a split merely because it produces a higher score.

If uncertainty exists, maintain multiple validation views and explain them.

---

## 10. Baseline First

Build a simple, fast baseline before sophisticated modeling.

Preserve the baseline permanently as a reference.

It must establish:

- an end-to-end pipeline
- a measurable score
- a sanity check
- a benchmark for future experiments

Every improvement must be compared against it.

---

## 11. Experiment Discipline

Every meaningful experiment should record:

- experiment ID
- code/config version
- data assumptions/version
- validation scheme
- preprocessing
- features
- model
- hyperparameters
- runtime
- memory/compute
- metric(s)
- result
- interpretation
- next hypothesis

Use a format such as:

```text
EXP-001
Hypothesis:
Change:
Validation:
Score:
Runtime:
Result:
Interpretation:
Next step:
```

Distinguish:

- **Observed** — directly measured
- **Inferred** — supported conclusion
- **Hypothesized** — not yet tested

Never present a hypothesis as fact.

---

## 12. Modeling

Do not assume the solution must use a large neural network.

Select model families from the actual data and metric.

Depending on the problem, consider:

- linear models
- tree/boosting models
- ranking
- retrieval/candidate generation
- text representations
- embeddings
- neural networks
- hybrid systems
- ensembles
- post-processing
- threshold optimization

Do not use deep learning merely because the challenge is called ML.

---

## 13. Feature Engineering

Consider, where relevant:

- numerical transformations
- categorical encodings
- frequency features
- aggregations
- group/entity statistics
- temporal features
- text normalization
- string similarity
- character/token features
- embeddings
- interactions

Every feature must be checked for leakage and unseen-data behavior.

Prefer features that are reproducible and computationally reasonable.

---

## 14. If the Task Is Entity Resolution

Do not assume this until confirmed by the actual problem statement.

If confirmed, explicitly investigate:

```text
Raw records
    ↓
Normalization
    ↓
Candidate generation / blocking
    ↓
Pairwise/listwise features
    ↓
Matching model
    ↓
Confidence / thresholding
    ↓
Set-level consistency
    ↓
Final prediction
```

Measure candidate-generation recall separately from final matching performance.

Investigate exact matches, normalized matches, field similarities, string similarities, missing fields, conflicting fields, duplicates, one-to-one constraints, false-positive/false-negative tradeoffs.

A matcher cannot recover a true match that candidate generation never produces.

---

## 15. Metrics

Reproduce the official metric locally where possible.

Understand:

- exact definition
- averaging method
- class imbalance
- threshold behavior
- ranking behavior
- asymmetric error costs
- edge cases

If the official evaluator is unavailable, document the proxy metric.

Never claim a proxy is identical to hidden evaluation without evidence.

---

## 16. Leaderboard

If a public leaderboard exists, treat it as an external signal.

Do not:

- overfit to tiny leaderboard movements
- make untracked changes
- infer causality from one score change
- sacrifice reproducibility

Use:

```text
Local validation
+
Robust validation where appropriate
+
Leaderboard feedback
```

as separate evidence sources.

Record submissions and observed scores.

---

## 17. Compute Strategy

Use the cheapest adequate compute first:

```text
Local machine
    ↓
Optimized local/parallel processing
    ↓
AWS CPU if needed
    ↓
SageMaker processing/training
    ↓
GPU only if justified
```

Before expensive AWS work:

- estimate runtime
- estimate credit/cost usage
- estimate information gained
- confirm the experiment is worth running

Stop idle AWS resources.

---

## 18. AWS

AWS may be used for S3, larger data processing, SageMaker notebooks, processing jobs, training jobs, or other justified services.

The available AWS credits are finite.

Do not create resources simply because they are available.

Where practical, support both:

```text
LOCAL DATA
```

and:

```text
AWS/S3 DATA
```

through configuration rather than hard-coded paths.

---

## 19. Antigravity Agent Workflow

Before modifying code:

1. inspect the relevant files
2. understand the current state
3. identify dependencies
4. formulate the intended change
5. make the smallest coherent change
6. run appropriate checks
7. inspect the result
8. report what changed

Avoid large rewrites without measurement.

Prefer incremental, reversible changes.

---

## 20. No Hallucinated Results

Never claim:

- a score not actually measured
- a training run not executed
- a dataset property not inspected
- an AWS resource not verified
- a leaderboard result not observed
- an improvement not demonstrated

If something cannot be verified, say so.

---

## 21. Code Quality

Competition code should be:

- modular
- reproducible
- configurable
- testable
- efficient
- reasonably documented

Avoid:

- giant monolithic notebooks
- hard-coded absolute paths
- duplicated preprocessing
- hidden global state
- unnecessary dependencies
- unexplained magic numbers

Separate exploration from reusable pipeline code.

---

## 22. Suggested Repository Structure

Adapt as appropriate:

```text
amazon-ml/
├── README.md
├── .gitignore
├── docs/
│   ├── problem_understanding.md
│   ├── dataset_analysis.md
│   ├── validation.md
│   └── experiments.md
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── notebooks/
├── configs/
└── outputs/
```

Do not create unnecessary directories.

The raw `dataset/` directory remains untracked.

---

## 23. Documentation

Eventually maintain:

- problem understanding
- dataset understanding
- validation rationale
- baseline
- experiment history
- final approach
- reproduction instructions

Documentation must reflect actual measured behavior.

---

## 24. Human + Agent Responsibilities

The human team remains responsible for:

- strategic decisions
- competition submissions
- AWS account decisions
- credential management
- final validation
- competition compliance

The agent provides:

- analysis
- implementation
- experiment design
- debugging
- optimization ideas
- documentation
- reproducibility

For destructive, expensive, credential-sensitive, or competition-sensitive actions, ask for confirmation rather than acting blindly.

---

## 25. Problem Statement Recording

The supplied MP4 is an important source.

Until it has been transcribed/inspected, do not invent information from it.

Once a transcript is available, extract:

- explicit requirements
- technical hints
- AWS instructions
- dataset information
- submission requirements
- restrictions
- important clarifications

Separate explicit statements from inference.

---

## 26. AWS Builder Center Guide

Inspect the supplied AWS guide when AWS setup becomes relevant.

Extract:

- account/setup requirements
- Builder Center requirements
- credits/resources
- recommended services
- setup procedure
- restrictions
- competition-specific guidance

Do not create resources merely because the guide mentions them.

---

## 27. Current Phase

The project is currently in:

# PHASE 0 — Environment and Source Understanding

Immediate objectives:

1. Verify repository state.
2. Verify `.gitignore`.
3. Protect dataset and credentials.
4. Read the supplied problem statement.
5. Inspect the recording/transcript when available.
6. Inspect the AWS guide.
7. Inventory the dataset.
8. Build a data map.
9. Understand the evaluation problem.
10. Only then begin baseline modeling.

Do not jump directly to complex modeling.

---

## 28. First Dataset Investigation

The first actual dataset investigation must be a **read-only audit**.

It should:

- recursively inventory files
- identify formats
- measure sizes
- inspect schemas
- determine row counts efficiently
- identify likely train/test/reference files
- inspect representative samples
- summarize missingness
- summarize cardinality
- identify IDs
- identify labels/targets if present
- identify file relationships
- flag potential leakage
- avoid modifying raw data

Produce a concise machine-readable report plus a human-readable summary.

Do not train a serious model until this audit is sufficiently complete.

---

## 29. Optimization Philosophy

The objective is not to write the most code.

The objective is:

> Maximize reliable competition performance per unit of time and compute.

Prioritize experiments according to expected impact, confidence in the hypothesis, and experiment cost.

Prefer experiments that answer important unknowns.

---

## 30. Final Solution Requirements

Before final submission, the project should have:

- reproducible preprocessing
- reproducible inference
- validated pipeline/model
- documented assumptions
- documented validation
- experiment history
- final configuration
- final submission-generation code
- sanity checks
- no dataset commits
- no credentials in Git
- reproducible execution

Generate the final submission from code wherever possible.

---

## 31. Absolute Rules

1. Never commit the competition dataset.
2. Never expose credentials.
3. Never overwrite raw competition data.
4. Never fabricate results.
5. Never fabricate dataset properties.
6. Never claim an improvement without measurement.
7. Never assume the evaluation metric.
8. Never ignore leakage.
9. Never optimize only for noisy leaderboard feedback.
10. Never spend AWS credits without justification.
11. Never make destructive changes without confirmation.
12. Never treat speculation as fact.
13. Never add external data unless explicitly permitted by the rules.
14. Never use external information to bypass a competition restriction.
15. Preserve reproducibility.

---

## 32. Working Mindset

Treat this as a serious ML research and engineering project:

```text
Understand
    ↓
Inspect
    ↓
Measure
    ↓
Hypothesize
    ↓
Baseline
    ↓
Experiment
    ↓
Validate
    ↓
Analyze
    ↓
Improve
    ↓
Stress-test
    ↓
Finalize
```

Do not jump directly from:

```text
Problem statement → complex model
```

The solution strategy must emerge from evidence in the actual problem and data.

---

## End of Operating Instructions

These are operating instructions only.

They do NOT contain the final task prompt.

The task-specific prompt will be supplied separately after the environment and source materials have been prepared.

---

## Update Log — 2026-09-25 03:19 IST

### Purpose of this update

This is the first findings update appended to the operating instructions. It records verified discoveries from (a) the initial read-only dataset audit, (b) official-metric reproduction/verification, and (c) the deep data discovery investigation (`scripts/discovery/*`, `work/analysis/*.json`, `docs/deep_data_discovery.md`). All content above this section remains in force and is unchanged. This section is strictly additive; it does not delete, rewrite, or replace any earlier rule. Where a new finding updates the interpretation of an earlier statement, it is listed under "Conflict Resolution" below with explicit labels.

### New Findings Added

**F1. The task is confirmed to be large-scale record-level Entity Resolution (ER).** [VERIFIED — `problem_statement.txt`]
- Source 1 is the deduplicated reference; for each S1 entity predict all matching S2/S3 ids. Source 1 may match zero, one, or many records. A candidate (S2/S3) id appears in at most one S1 cluster (0 multi-participation).
- Evidence: verified against `train_ground_truth.tsv` (see F6) and the pipeline requirements in `problem_statement.txt`.

**F2. The submission stage produces TWO tab-separated files, not one.** [VERIFIED — `problem_statement.txt`]
- `output/matching_results.tsv` (scored on the leaderboard): `source1_entity_id <tab> matched_entity_ids` (comma-separated, or empty for singletons).
- `output/candidate_pairs.tsv` (NOT scored; used by organizers to audit blocking/candidate quality and verify the pipeline): must contain the exact candidate set fed to the final matching model just before scoring — not an earlier blocking pass.
- Hard validation rules: every S1 entity in the test set appears exactly once; empty list for singletons; only S2-/S3- ids that exist in the test set; no duplicate ids in a list; no duplicate S1 rows; every id in `matching_results.tsv` must appear in `candidate_pairs.tsv` (a matched id that was never a candidate is flagged as a pipeline bug).
- Local validator provided by the competition: `utils/validate_submission.py` (stdlib-only; run from the student resource directory; checks both files, exit 0 = PASS).
- Operational consequence: candidate-set construction is now a hard deliverable, not just an internal stage — it must be regenerable and auditable alongside the final matches.
- Source: `problem_statement.txt` (Output Format / Final Submission Package sections).

**F3. `country` is an open set of string labels; France appears only in the test set.** [VERIFIED — `problem_statement.txt`, `docs/dataset_analysis.md`]
- Train covers US (~60%) and India (~40%); France = 0 entries in train S1/S2/S3; test adds France (S1 259,452; S2 703,378; S3 731,615, ~15% of test S1).
- Instruction from the statement: do not hard-code, filter, or one-hot the pipeline to only {US, India}; every test entity, France included, must appear in the submission.
- 100% of train matches occur within the same country → country partitioning of blocking/candidate matching is safe (also in `docs/dataset_analysis.md` §3).

**F4. Model-license and size cap for the final model.** [VERIFIED — `problem_statement.txt`]
- Final model must be MIT/Apache-2.0 licensed and up to 8 billion parameters. External data lookups (registries, geocoding, entity-resolution APIs, etc.) are strictly prohibited and disqualifying — all evidence already in `problem_statement.txt`; restated here only as a build constraint for any future agent.

**F5. Official metric is macro-averaged F0.5 per S1 entity (already reproduced locally).** [VERIFIED — `problem_statement.txt` worked example 0.714; `src/evaluation/metrics.py`]
- Official score = simple mean of per-S1-entity F0.5; singletons included; all S1 entities contribute equally (not pair-weighted). `compute_f05_score` in `src/evaluation/metrics.py` implements this and matches the statement's worked example. Micro F0.5 is retained only as `compute_f05_micro` for divergence reporting (micro inflates vs. macro on real data).
- Baseline "predict nothing" F0.5 = **0.0558** (== singleton fraction 123,247 / 2,206,821). Per-entity "oracle + 1 FP everywhere" ≈ 0.8074; "drop 1 true match per entity" ≈ 0.9250.
- Consequence: an FP on a singleton moves that entity from 1.0 → 0.0; each missed match costs ~0.07–0.09 on its entity. Precision-heavy, entity-level behavior must drive all threshold/model choices.

**F6. Ground-truth structure: multi-match is the norm, and candidate participation is 1-to-N star topology.** [VERIFIED — `docs/dataset_analysis.md` §5; `work/analysis/11_cardinality.json`; `scripts/discovery/11_cardinality.py`]
- 7,638,365 GT pairs across 2,083,574 matched S1; 123,247 singletons (5.585%).
- Matches per S1: mean 3.67, median 4, p90 6, p99 8, max 11; >50% of matched entities have ≥4 matches → prediction must be multi-match; "top-1" or single-match assumptions are wrong.
- S2-only 143k S1, S3-only 164k, both 1,776,047.
- S2 and S3 multi-participation exactly 0 → disjoint rooted-star clusters; enables mutually-exclusive greedy/bipartite post-processing (already noted in `docs/problem_understanding.md` §3).

**F7. ~26% of candidate records are unlinkable distractor noise.** [VERIFIED — `docs/deep_data_discovery.md` §1; `work/analysis/11_cardinality.json`]
- 26.6% of S2 (1,340,997) and 25.4% of S3 (1,340,857) records appear in zero GT pairs (~2.68M records). Under macro F0.5, accepting a pair to any such record is an FP by definition → candidate "plausibility" ≠ match; the matcher must actively reject noise.

**F8. Name corruption is severe on true matches; exact-name recall is capped ≈29%.** [OBSERVED — 13% hash sample n=992,763; `work/analysis/02_corruption.json`; `scripts/discovery/02_corruption.py`]
- True-pair name differences (exclusive types): exact copy 4.6%, case-only 6.1%, punctuation-only 15.0%, corporate-suffix-only 2.8%, token reorder 7.1%, token-subset 27.2%, heavy rewrite 37.2% (heavy bucket mean name token-Jaccard 0.32; India 0.25 vs US 0.38).
- Address differences: exact 2.2%, case 5.0%, norm-map 1.0%, order 4.3%, token-subset 10.9%, one-side-missing 4.4%, heavy 72.0% — but heavy addresses retain mean leftover token-Jaccard ~0.53.
- India dominates corporate-suffix churn and reordering (names are least reliable there).

**F9. Feature discrimination (pool = 600k positives + 821k negatives, 4 channels).** [OBSERVED — `work/analysis/01_pair_sim_summary.json`; `scripts/discovery/01_pair_similarity.py`]
- AUC order: addr token-Jaccard 0.89, addr containment 0.89, addr char3 0.88, addr char2 0.87, numeric-token overlap 0.79, name char2 0.67, name token-Jaccard 0.66, rare-shared name tokens 0.61, normalized name equality 0.49 (no usable signal).
- Probes: name-eq AND addr-char2≥0.5 → 99.86% positive (n=144,606); numeric overlap≥1 → 85.6% positive; name-eq alone → only 40.55% positive.
- Evidence hierarchy conclusion (see F10/F11): address → numeric → name; rare-token-name is a weaker auxiliary channel.

**F10. Same-name candidates are plentiful and ambiguous, but address almost fully resolves the decision.** [OBSERVED — `work/analysis/03_ambiguity.json`; `scripts/discovery/03_04_ambiguity_evidence.py`]
- Same normalized-name pool pairs (423,603): 40.6% negatives. Among those same-name negatives: addr token-Jaccard≥0.5 = 0.03%, any shared numeric token = 0.28%, addr char2≥0.6 = 0.05%. Among same-address negatives (71.3k): name token-Jaccard≥0.5 = 1.4%.
- Both name-strong AND address-strong negatives in the sampled pool: **656 / 821,287**. → "name-strong, address-weak" collisions are the real failure mode to engineer for.

**F11. Evidence hierarchy on the 600k positive sample.** [OBSERVED — `work/analysis/04_evidence.json`]
- addr_strong (addr token-jac≥0.5 OR any shared numeric token) covers 83.3% of positives; numeric 66.1%; name_strong (name token-jac≥0.8 AND containment≥0.8) 40.1%; overlap 32.9%; addr-only-strong 50.5%; neither 9.5%. When name is weak (59.9% of positives) address rescues 84.2%; when address is weak (16.7%) name rescues only 43.2%.
- Conclusion: address (+numbers) is the backbone; name is a tie-breaker; strong-name matches still need an address check.

**F12. Digit agreement is high-precision but moderate-recall.** [OBSERVED — `work/analysis/05_address.json`; `scripts/discovery/05_address.py`]
- On matched pairs with digit tokens on both sides (87.2% of pairs): first numeric token (house/plot #) agrees 54.8%; last numeric token (postal) agrees 57.1%. US: 92.4% addresses start with a house-number-like token; India: 90.8% (median 11 tokens/address vs US 6).
- Digits are themselves corrupted in a meaningful fraction of matches; numeric agreement is a strong precision anchor but not high-recall.

**F13. Crisp conjunctive score rules dominate loose unions.** [OBSERVED — pool-bounded ordering, `work/analysis/09_rules.json`; `scripts/discovery/09_rules.py`]
- F0.5 over covered S1 (official macro): R9 (num≥1 & name≥0.5) ≈ 0.678; R6 (name≥0.6 & addr≥0.5) ≈ 0.652; R7 (addr≥0.4 & num) ≈ 0.613; R10 loose "any-strong" union ≈ 0.400 with negative-hit rate 39.9%. R2 (name eq & addr eq) tiny recall (1.7%).
- These are pool-bounded and serve as an ordering, not an absolute system score (see F18).

**F14. Country-specific signal asymmetry (India vs US).** [OBSERVED — `work/analysis/07_08_country_domainshift.json`; `scripts/discovery/07_08_country_domainshift.py`]
- India: name token-jac AUC 0.57 vs US 0.72; India addr token-jac AUC 0.92 vs US 0.87; India numeric-overlap AUC 0.81 vs US 0.77. India names are weaker and addresses/numbers stronger → country-specific thresholds/weights are justified.

**F15. Heavy train→test domain shift; France is wholly novel.** [VERIFIED / OBSERVED — `work/analysis/07_08_country_domainshift.json`]
- Only 9.4% of test name-blocks and 23.5% of test name tokens appeared in train. Rare-token statistics learned on train will cover only roughly a quarter of test tokens. France has no train entities at all. Conclusion: treat train vocabulary as illustrative, not exhaustive; do not rely on memorized names or token df values computed only on train.

**F16. Single-channel exact-name blocking has a recall ceiling near 50% of matches.** [OBSERVED — `work/analysis/10_blocking.json`; `scripts/discovery/10_blocking.py`]
- Only ~1.04M of 2.08M matched S1 have any candidate sharing their exact normalized name block within country (a caps recall far below the 98% candidate-generation recall target for the exact-name channel alone). Multi-channel generation (exact-name + shared rare/medium token + shared numeric + address block, empties excluded, plus a fuzzy channel) is therefore required, not optional.
- Name block sizes (within country): p50 = 1, p90 = 3, p99 = 9, max 1072 ("physical therapy"); 60.6k blocks > 10.
- Address blocks mostly singletons (9.2M of 10.5M groups) but the EMPTY address forms a single 222k-record group; bare digit tokens explode as keys ("2nd" → 145k records). Empty addresses must be excluded from address-block keys; numeric overlap should be applied pair-wise and digit keys given length checks (zips/pincodes only).

**F17. Cross-source corroboration (S2 vs S3) is rare.** [OBSERVED — `work/analysis/12_crosssource.json`; `scripts/discovery/12_crosssource.py`]
- For S1 with both S2 and S3 matches (1,733,536 S1, 4.4M cross-pairs): same normalized name on only 14.4% of cross-pairs; at-least-one per-S1 agreement 35.4%; full (name+addr) agreement 0.06%. Do not condition a model on "S2 and S3 separately confirm".

**F18. Validation hygiene and result-status caveats.** [VERIFIED — `work/analysis/13_splitcheck.json`; `scripts/discovery/13_splitcheck.py`; `work/analysis/01_pair_sim_summary.json`]
- No pre-built train/valid split artifacts exist in the repo (a `split_exists` grep only matched docs/self-references). Train vs test S1 id overlap = 0; no GT candidate id appears in test. → CV must be built at S1-entity level (group positives per S1, split by S1 never by individual pairs), with per-S1 match counts balanced across folds (macro metric is per-entity).
- Pooled F0.5 numbers above are **orderings, not absolute scores**: positives are a 600k sample of 7.6M; negatives come from 4 sampled channels; F0.5 was computed over covered (recalled) S1 only. Full-system evaluation is required before treating any number as a score.
- Analysis-tooling note: a tie-averaging bug in the AUC computation in `scripts/discovery/pairfeat.py` was fixed (tie ranks written via `ranks[order[s:e]]`) and summaries recomputed; `work/analysis/01_pair_sim_summary.json` is authoritative.

### Evidence / Source

- `problem_statement.txt` — F1, F2, F3, F4, F5 formula & example; open-set country rule; validation rules; license/size cap.
- `src/evaluation/metrics.py` — official macro `compute_f05_score`, singleton conventions, micro comparison retained (F5).
- `docs/dataset_analysis.md` — inventory, missingness (2.6–3.4% missing addresses in S2/S3, none in S1), country distribution, cardinality, 100% same-country matches, 0 cross-split overlap (F3, F5, F6, F18).
- `docs/deep_data_discovery.md` — consolidated report of the second investigation (F6–F18 summaries, EXP-001/EXP-002 hypotheses).
- `scripts/discovery/01_pair_similarity.py` + `01_recompute_summary.py` + `backfill_rare_shared.py` + `work/analysis/01_pair_sim_summary.json` — pool build, feature AUCs, probes, rare-shared backfill (F9, F12, F18).
- `scripts/discovery/02_corruption.py` + `work/analysis/02_corruption.json` — corruption taxonomy, 13% sample (F8).
- `scripts/discovery/03_04_ambiguity_evidence.py` + `03_ambiguity.json` / `04_evidence.json` — ambiguity and evidence hierarchy (F10, F11).
- `scripts/discovery/05_address.py` + `05_address.json` — address decomposition, house/postal agreement (F12).
- `scripts/discovery/06_name.py` + `06_name.json` — name token structure per country (suffix/generic-token stopword behavior).
- `scripts/discovery/07_08_country_domainshift.py` + `07_08_country_domainshift.json` — country AUCs and train/test overlap (F14, F15).
- `scripts/discovery/09_rules.py` + `09_rules.json` — deterministic rule F0.5 ordering (F13).
- `scripts/discovery/10_blocking.py` + `10_blocking.json` — block-size stats and ceilings (F16).
- `scripts/discovery/11_cardinality.py` + `11_cardinality.json` — match cardinality and distractor fractions (F6, F7).
- `scripts/discovery/12_crosssource.py` + `12_crosssource.json` — cross-source agreement (F17).
- `scripts/discovery/13_splitcheck.py` + `13_splitcheck.json` — split hygiene (F18).
- `work/pair_pool.parquet` (1,421,264 rows), `work/pos_pairs_full.parquet` (7,638,365 rows), `work/gt_long.parquet` — reproducible artifacts backing the above.
- All raw `dataset/` files were only ever read; nothing in `dataset/` was modified.

### Impact on Modeling / Experiments

- The preferred architecture (already outlined in `docs/experiment_plan.md` and `docs/deep_data_discovery.md` §17) is: normalization → multi-channel blocking/candidate generation (country-partitioned) → pairwise/listwise features → scored matcher → per-entity thresholded selection → disjoint-assignment post-processing. All of it reproducible from code; two-file submission output.
- **EXP-001 — "blocking + crisp scoring" [HYPOTHESIS, not yet validated as a full system].** Candidate generation: (a) exact name block (country-scoped, suffix-mapped; note the ≈50% coverage floor so other channels are mandatory), (b) shared rare/medium name token (df-bounded), (c) shared numeric token (house#/postal, length-checked), (d) exact address block (empty addresses excluded; optionally fuzzy address channel). Score with a small calibrated model over {addr token-jac, addr char2/3, name char2, name token-jac, numeric overlap, rare-shared, length diffs}, accept via conjunctive high-precision rules first, address-anchored thresholds second, name-only rarely. Hypothesis: >0.6 macro F0.5 on matched-S1 sub-population with singleton FPs ≈ 0.
- **EXP-002 — "per-entity selection" [HYPOTHESIS, not yet validated].** Greedy per-S1 selection with macro-aware tiers: (i) high-precision anchor tier (name&addr strong, or numeric+addr) predicting all found matches, (ii) medium tier (addr-strong), (iii) low tier only when the entity otherwise has 0 predictions (protects recall on multi-match entities without risking singleton FPs). Punish guesses on low-evidence S1 (singletons already earn 1.0 when empty is predicted).
- **Reinterpretation of earlier plan targets [UPDATED FINDING].** The `validation_strategy.md` "blocking Recall@K ≥ 98% with K ≤ 50" target remains in force conceptually but is unreachable by an exact-name-only block channel (verified ceiling ≈ 50%); it is a target for the combined multi-channel candidate set, and candidate-generation recall must be measured separately from matching quality (already required by §14 above).
- Per-country thresholds/features are supported by F14. France has no train data; rely on language-agnostic features (token/char similarity, numeric agreement, conjunctions) and pseudo-OOD holdouts of non-standard train subsets as a proxy; do not tune France-specific thresholds on absent data.
- Name similarity should down-weight corporate suffixes and generic business words (verified stopword-like behavior in `06_name.json`); rare/mid-token weighting is supported (rare-shared AUC 0.61, useful only as auxiliary).
- The fixed-pool caveats (F18) mean the above F0.5 numbers are directional; a full-system validation run on S1-level CV is required before selecting final thresholds.

### New Instructions

1. Use `compute_f05_score` (official macro per-S1) as the single reporting metric for every experiment; report `compute_f05_micro` only as a divergence diagnostic. [F5]
2. Treat §14 above as confirmed: ER pipeline applies in full; always measure candidate-generation recall separately from matching performance, and keep the final candidate set auditable (it is a submission deliverable). [F1, F2]
3. Predict multi-matches per S1; never assume a single (top-1) match; the expected match count under GT is ~3–4 for matched entities. [F6]
4. Build blocking from multiple channels (exact-name block + shared rare/medium token + shared numeric + address block, country-partitioned); do not rely on exact-name blocking alone. Exclude empty addresses from address-block keys; apply digit tokens as keys only with length checks (e.g. zips/pincodes); apply common-digit numeric overlap pair-wise, not as a block key. [F16]
5. Weight evidence as address → numeric → name, with rare-token-name as an auxiliary channel; never accept name-equality alone (≈40% negative among same-name candidates); when accepting a name-strong match, require an address signal. [F9, F10, F11]
6. Implement acceptance as crisp conjunctions (name+address, name+number) rather than loose "any-signal" unions; loose unions empirically explode FP (R10 neg-hit 39.9%). [F13]
7. Wherever the code branches by country, keep `country` an open set (string label); do not hard-code {US, India}; ensure France entities are emitted. Use country-aware thresholds/features but country-agnostic text handling (unicode/NFKD normalization already handles accents). [F3, F14, F15]
8. Validate at S1-entity level: group positives per S1, split by S1 (never by pairs), and balance per-S1 match counts across folds; there is no pre-built split; treat pooled F0.5 numbers as orderings until a full CV run exists. [F18]
9. Never condition a model on S2↔S3 "confirmation" (agreement is <15% and full agreement ≈0.1%). [F17]
10. Respect train-to-test domain shift: do not trust exact-name blocking recall or train-computed token df / rare-token tables to transfer to test (9.4% block overlap); keep the rare-token dynamic or recompute-with-fallback safe. [F15]
11. Build the two-file submission end-to-end and locally validated: generate `matching_results.tsv` and `candidate_pairs.tsv` (candidate set = exact model input, matching ⊆ candidates, one row per S1, S2-/S3- ids only, no duplicates), then run the provided `utils/validate_submission.py` before any upload. [F2]
12. Respect model constraints: MIT/Apache-2.0 licensed, ≤ 8B parameters; no external data/APIs/geocoding/registry lookups. [F4]

### Conflict Resolution

- **C-1 (§27 "Current Phase" = PHASE 0) vs. actual repository state.** The operating instructions were drafted during Phase 0, but repository evidence now shows Phase 0–Phase 3 largely executed: environment+source understanding done, read-only datasets audited (`docs/dataset_analysis.md`), official metric verified (`src/evaluation/metrics.py`), and the second deep data discovery completed (`docs/deep_data_discovery.md`, `work/analysis/*.json`). [VERIFIED] The original §27 text is preserved unmodified; the authoritative current picture is this Update Log, `docs/`, and `work/`. Future agents should treat the project as being in the baseline/experiment phase (EXP-001/EXP-002 hypotheses pending), not Phase 0.
- **C-2 (§14 "Do not assume this is entity resolution until confirmed") vs. F1.** The task is now explicitly confirmed as entity resolution by `problem_statement.txt`. [VERIFIED] §14's caution is retained verbatim; its condition is now satisfied, so its pipeline guidance applies.
- **C-3 (`docs/validation_strategy.md` "blocking Recall@K ≥ 98%, K ≤ 50") vs. F16.** Not weakened; recontextualized. The 98% target must be met by the combined multi-channel candidate set, not by any single exact-name channel (verified exact-name coverage ≈ 50% of matched S1). [UPDATED FINDING]
- **C-4 (earlier micro-averaged F0.5 descriptions) vs. F5.** Earlier docs described the metric as micro; corrected implementations (`src/evaluation/metrics.py`, `docs/problem_understanding.md`, `docs/validation_strategy.md`) and the official statement confirm macro per-S1. instructions.md §15 (reproduce official metric) is unaffected; `compute_f05_micro` is retained only for comparison. This conflict was already resolved inside the docs; recorded here for traceability.

### Open Questions / Unknowns

- [RESOLVED] Official submission evaluated on Unstop Leaderboard: Baseline score = 0.526. Top leaderboard benchmark = ~0.985+.
- [RESOLVED] High-speed C++17 OpenMP engine on AWS EC2 r5.2xlarge executes 1.73M queries in 3.6 minutes with <800 MB RAM and passes 100% official format validation.
- [UNKNOWN] Optimal MinHash LSH / Trigram candidate blocking configuration to push blocking recall from 71.8% to >95%.
- [UNKNOWN] French domain-specific postal and abbreviation lexicon tuning to close out-of-distribution performance gap.

---

## 29. Update Log & Leaderboard Results (September 25, 2026)

### 29.1 Official Submission & Leaderboard Status
- **Submission 1 (Evaluated)**:
  - **Timestamp**: September 25, 2026, 04:24 PM IST
  - **Status**: Evaluated & Scored
  - **Official Macro $F_{0.5}$ Score**: **0.526**
- **Current Top Leaderboard Benchmarks**:
  - **Rank 1**: Banana — **0.985884**
  - **Rank 2**: Midnight Blizzard — **0.984644**
  - **Rank 3**: KL_converge — **0.984350**
  - **Rank 4**: MavericK — **0.984**
  - **Rank 5**: Medallion — **0.984**
  - **Rank 6**: Grinders — **0.983**

---

### 29.2 System Enhancements & Completed Infrastructure

```text
+-----------------------------------------------------------------------------------+
|                        AMAZON ML CHALLENGE 2026 PIPELINE                          |
|                                                                                   |
|  1. Preprocessing & Unicode Normalization (NFKD, Diacritic Stripping, Lowercase)  |
|                                    │                                              |
|                                    ▼                                              |
|  2. 6-Channel Multi-Index Candidate Blocking (Exact Name, Suffix, Addr, Numbers)  |
|                                    │                                              |
|                                    ▼                                              |
|  3. Standalone C++17 OpenMP Native Inference Engine (src/native/matcher.cpp)      |
|     - Lazy 2-gram/3-gram Dice string SIMD features (< 800 MB RAM)                 |
|     - 239-Tree Compiled LightGBM Predictor (src/native/lgbm_model.h)              |
|     - Conjunctive Rule Gating & Prior Calibration                                 |
|     - Disjoint Star-Cluster Partitioning (Eliminates Multi-Merge Conflicts)       |
|                                    │                                              |
|                                    ▼                                              |
|  4. AWS EC2 Batch Worker (r5.2xlarge in us-east-1): Runtime = 3.6 minutes         |
|                                    │                                              |
|                                    ▼                                              |
|  5. Final Verified Outputs:                                                       |
|     - matching_results.tsv (1,732,544 rows, 100% compliant, PASS on validator)   |
|     - candidate_pairs.tsv (blocking audit dataset)                                |
|     - amazon_ml_submission.zip (Complete code, models, and methodology package)   |
+-----------------------------------------------------------------------------------+
```

---

### 29.3 Root-Cause Analysis: Closing the Gap from 0.526 to 0.985+

1. **Candidate Blocking Recall Bottleneck**:
   - Current exact token and suffix blocking achieves ~71.8% recall on holdout pairs. Missing candidates cannot be recovered by the ranker.
   - **Fix (EXP-004)**: Introduce MinHash LSH on character 3-grams and phonetic Soundex/Metaphone keys to elevate blocking recall to $>96\%$.
2. **Precision Weighting in $F_{0.5}$ ($2\times$ Precision Bias)**:
   - False positive merges catastrophically penalize the macro-average.
   - **Fix**: Calibrate anchor thresholds ($\ge 0.90$) and enforce strict numerical/postal code agreements on address matches.
3. **France & Multi-lingual Domain Shift**:
   - The test set contains France (unseen in training data).
   - **Fix**: Standardize French address tokens (*Rue, Boulevard, Avenue, Cedex*, 5-digit postal code matchers).

---

### 29.4 Quick Reproduction & Execution Guide

1. **Export LightGBM Model to C++**:
   ```bash
   python scripts/export_lgbm_to_cpp.py
   ```
2. **Launch Cloud Batch Worker on AWS**:
   ```bash
   python scripts/aws/launch_cpp_batch_compute.py
   ```
3. **Collect & Validate Submission Artifacts**:
   ```bash
   python scripts/aws/collect_outputs.py
   ```
4. **Validate Output TSVs Locally**:
   ```bash
   python dataset/student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/student_resource/dataset/test
   ```

