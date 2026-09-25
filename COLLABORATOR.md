# COLLABORATOR.md

Handy reference for anyone working in this repository. Short version:

- **Read the findings, follow the rules, never commit secrets.**
- If you are unsure about a step, ask before acting.

---

## 1. What This Repository Is

This is the codebase for the team's entry to the **Amazon ML Challenge 2026 — Business Entity Resolution** problem. It contains:

- A reproducible **multi-channel blocking → rule-gated LightGBM ranker → disjoint star-cluster** entity resolution pipeline (EXP-003 is the current best experiment).
- Analytical/discovery scripts and documentation used during research.
- A complete AWS/SageMaker workflow for running test inference in the cloud.

The primary deliverable is a submission file: `matching_results.tsv` mapping each Source-1 entity to its matched Source-2/Source-3 entity IDs.

---

## 2. Problem & Metric Context

- **Task:** entity resolution across three data sources (S1, S2, S3).
- **Metric:** official **macro per-entity F0.5** — precision-weighted (precision matters 2x recall).
- **Key constraint:** predictions must be **strict 1-to-many S1 → {S2, S3}** with no cross-entity collisions and no self-matches.
- Best offline validation: **Macro F0.5 = 0.6881** (P 0.8448 / R 0.6125) on a stratified, cluster-isolated, leakage-free holdout.

See `docs/` (problem_understanding, experiment_plan, exp001/exp003 reports, validation_strategy) for full background.

---

## 3. Repository Layout

| Path | Contents |
| :--- | :--- |
| `src/` | Core modules: data loading/splitting, candidate generation (blocking), features, models, rules, submission generation |
| `scripts/` | Runners: discovery analyses, EXP-001/002/003 benchmarks, test inference, AWS pipeline, validation, packaging |
| `scripts/aws/` | AWS/SageMaker workflow (`aws_config.py`, upload, submit, monitor, collect) |
| `docs/` | Research, design, and experiment documentation |
| `experiments/` | Per-experiment metrics and generated run artifacts |
| `work/` | Derived data + analysis outputs (mostly git-ignored, e.g. `*.parquet`) |
| `models/` | Trained model artifacts (git-ignored) |
| `output/`, `submission/`, `submissions/` | Generated inference/submission artifacts (git-ignored) |

---

## 4. Tracked vs. Local-Only Content (Git & Data Policy)

The following are **never committed** — they are large, competition-restricted, account-specific, or secret. They live only on local machines:

- `dataset/` and `6ab10eb3b23ba_student_resource.zip` — competition data (2GB+, data-sharing restricted).
- Media files (`*.mp4`, etc.) — e.g. the problem statement video.
- Derived/work files: `work/*.parquet`, `models/*.joblib`, `output/`, `outputs/`.
- Generated submission artifacts: `submission/*.tsv`, `submission/*.zip`, `submissions/`.
- Private keys, credentials, certificates — `*.pem`, `*.key`, `.aws/`, `credentials`, `.env*`.
- Generated AWS run artifacts referencing live resources — `experiments/*/s3_manifest.json`, `latest_job.json`, `submission_validation.json`, `local_aws_config.json`, `checkpoints/`.

If you add new tracked code that generates large/derived files, extend `.gitignore` accordingly. When in doubt, ask.

---

## 5. Security Policy — READ THIS FIRST

1. **Never commit secrets.** This includes access keys, tokens, passwords, private keys, and account identifiers.
2. **Do not hard-code AWS account IDs, bucket names, or role ARNs in committed code.** Use the config mechanism in §8.
3. There is a private RSA key file (`pra-navpair-key22.pem`) historically present at the repo root on local machines. It is ignored by `.gitignore`. **Do not read it, move it, print it, or commit it.**
4. Do not paste live URL/S3 console links that expose account IDs into issues, PRs, or docs.
5. If you believe a secret was committed, tell the team immediately; fix it in the same working session (rotate + rewrite history if needed).

---

## 6. Environment Setup

- **Python:** 3.12.x. A local venv lives in the repo as `.venv/` (git-ignored) via `uv` (uv 0.11.x).
- Install dependencies: `uv pip install -r requirements.txt` (polars, lightgbm, scikit-learn, joblib, numpy, pyarrow, boto3, botocore).
- The dataset must be present locally under `dataset/student_resource/dataset/{train,test}` to run the pipeline.
- All code is **path-independent**: repo-relative paths are derived from `__file__` at runtime — never hard-code machine-specific absolute paths (e.g. `C:\Users\<you>\amazonml`).

---

## 7. Working with AWS

This team uses the **new AWS experience** (sign-in via a social provider on `aws login`):

- **Profile:** `default` · **Selected Region:** `us-east-1`.
- Authenticate with `aws login --region us-east-1 --profile default` (long-lived, renew periodically). Never paste the resulting credentials into files.
- If you hit sudden "Access Denied" on operations that previously worked, ask about the **spend limit** and check AWS Settings → Billing before assuming an IAM problem.
- Regional resources (Lambda, SageMaker, etc.) are created **only in the selected Region**; do not create them elsewhere.

---

## 8. AWS Resources & Configuration (Integration Point)

All SageMaker scripts read their AWS configuration from `scripts/aws/aws_config.py`. It resolves values in this order:

1. **Environment variables:**
   - `AMAZONML_REGION` (defaults to `AWS_REGION`/`AWS_DEFAULT_REGION`/`us-east-1`)
   - `AMAZONML_SAGEMAKER_BUCKET` *(required to upload/submit/collect)*
   - `AMAZONML_SAGEMAKER_PREFIX` (default `amazon-ml-challenge`)
   - `AMAZONML_SAGEMAKER_ROLE_ARN` *(required to submit a job)*
2. **Git-ignored local file** `experiments/EXP003-AWS/local_aws_config.json`:

   ```json
   {
     "region": "us-east-1",
     "bucket": "sagemaker-<region>-<account-id>",
     "s3_prefix": "amazon-ml-challenge",
     "role_arn": "arn:aws:iam::<account-id>:role/<your-sagemaker-execution-role>"
   }
   ```

A collaborator on a new machine copies that local JSON (or sets env vars) with **their own** account's bucket/role. The shared SageMaker execution role must allow SageMaker processing, S3 read/write on the bucket, and CloudWatch Logs for `/aws/sagemaker/ProcessingJobs`.

---

## 9. Cost Guardrails

- AWS runs are provided through the AWS experience (plan/spend-limit dependent). Respect the configured **spend limit** — if you think you exceeded it, check AWS Settings → Billing before debugging further.
- The SageMaker processing job is intentionally conservative:
  - Instance: `ml.m5.2xlarge` (8 vCPUs / 32 GB RAM, CPU only) × 1
  - Hard stop: `MaxRuntimeInSeconds = 7200` (2 hours)
  - EBS volume: 40 GB
- Do not raise these guardrails without team agreement.

---

## 10. SageMaker Workflow

Pipeline for cloud test inference (EXP003-AWS):

1. **Upload data** → `python scripts/aws/upload_dataset.py` — uploads raw TSVs, model artifact, and preprocessed checkpoint to S3 with MD5 checksums; writes `s3_manifest.json`.
2. **Submit job** → `python scripts/aws/submit_processing_job.py` — uploads the entrypoint (`scripts/aws/sagemaker_entrypoint.py`), creates the processing job, saves `latest_job.json`.
3. **Monitor** → `python scripts/aws/monitor_job.py [job_name]` — polls status and streams CloudWatch logs.
4. **Collect** → `python scripts/aws/collect_outputs.py` — downloads outputs, runs the submission validator, packages the Unstop submission ZIP, writes `docs/aws_exp003_run.md`.

The entrypoint (`sagemaker_entrypoint.py`) installs missing packages inside the container, loads the model + checkpoint from S3, runs per-country streaming inference, and writes `matching_results.tsv`, `candidate_pairs.tsv`, `run_summary.json`.

---

## 11. Running Experiments Locally

- Data discovery: `python scripts/discovery/01_pair_similarity.py` and siblings (writes `work/analysis/*.json`).
- EXP-001/002 benchmark suite: `python scripts/run_exp001_exp002.py`.
- EXP-003 (LightGBM ranker): `python scripts/run_exp003.py`.
- Local test inference: `python scripts/run_test_inference.py` (RAM-safe streaming, writes `output/`).
- Smoke test (fast subset): `python scripts/test_smoke.py`.
- Packaging local submission ZIP: `python scripts/package_submission.py`.

These scripts derive paths from their own location, so they work from any machine layout.

---

## 12. Validation & Submission

- Run `python scripts/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv` to check:
  - exact TSV headers/delimiters, 100% S1 row coverage (1,732,544 rows), valid `S2-`/`S3-` prefixes, no self-matches, no duplicate rows/IDs, matched ⊆ candidates, and ID existence in the test source files.
- The official validator also ships in the dataset bundle (`dataset/student_resource/utils/validate_submission.py`).
- Leaderboard upload expects a standalone `matching_results.tsv` (tab-separated) plus a ZIP package (see `docs/submission_spec.md` and `submission/MANIFEST.md`).

---

## 13. Git & GitHub Workflow

- Remote: `origin` → `github.com/<org>/amazon-ml` (see `git remote -v`). Branch: `main`.
- **Push only `main` → `origin/main`.** Never `git push --all`, `--mirror`, or `--force`.
- Keep the commit history free of competition data and secrets. Before pushing: review `git status`, `git diff --cached`, and confirm the staged set contains no `.parquet`/`.joblib`/`.pem`/`.env`/`.mp4`/`.zip`/account IDs.
- Style: concise lowercase-prefixed commit messages (e.g. `docs: explain AWS job workflow`, `fix: remove absolute paths`).
- If the official competition data must never leak, keep it out of any public clone/PR.

---

## 14. Guidance Levels & Conventions

- Preferred help level when sharing/assisting: **MEDIUM** — flag issues and ask a couple of clarifying questions when something seems off, but otherwise just execute.
- Prefer **LOW/MEDIUM** verification over over-explaining; escalate to **HIGH** only for security-sensitive or irreversible operations (pushing, spending money on AWS, deleting files).
- Ask before: changing git history, force-pushing, deleting data/artifacts, or modifying AWS resources beyond the documented workflow.
- Keep documentation truthful; if a doc references files, paths, or commands, they should exist and work.