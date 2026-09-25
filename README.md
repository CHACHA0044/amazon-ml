# Amazon ML Challenge 2026

Entity resolution across three business data sources (S1 → {S2, S3}), evaluated by **macro per-entity F0.5**. The current pipeline (EXP-003) is a hybrid **multi-channel blocking → rule-gated LightGBM ranker → disjoint star-cluster assignment** and achieves an offline validation Macro F0.5 of **0.6881** (precision 0.8448, recall 0.6125).

## Project Structure

    dataset/        Competition dataset - NOT tracked in Git
    src/            Core pipeline modules (data, blocking, features, models, submission)
    scripts/        Experiment runners, discovery analyses, AWS workflow, validation, packaging
    scripts/aws/    SageMaker cloud-inference workflow (upload / submit / monitor / collect)
    docs/           Research, design, and experiment documentation
    experiments/    Per-experiment metrics and run artifacts
    work/           Derived data and analysis outputs (mostly git-ignored)
    models/         Trained model artifacts (git-ignored)
    output/         Generated inference results (git-ignored)
    submission/     Submission manifest + generated package (artifacts git-ignored)

## Dataset

The competition dataset is intentionally kept local and excluded from Git because of its size and competition/data-sharing considerations. It is not part of this repository; placing it under `dataset/student_resource/dataset/{train,test}` is required to run the pipeline (see Reproducibility).

## Security & Shared-Code Rules

- **Never commit secrets**, credentials, private keys, or AWS account identifiers.
- Code is path-independent: it derives repo-relative paths from `__file__` — never hard-code absolute paths.
- AWS/SageMaker settings are resolved at runtime from environment variables or a git-ignored local config (see `scripts/aws/aws_config.py` and `COLLABORATOR.md`).
- Review `git status` / staged diffs before pushing; only `main` → `origin/main` is pushed (no `--all`, `--mirror`, `--force`).

See `COLLABORATOR.md` for the full working agreement.

## Reproducibility

1. Place the competition dataset under `dataset/student_resource/dataset/{train,test}`.
2. Create a venv and install dependencies (`polars`, `lightgbm`, `scikit-learn`, `joblib`, `numpy`, `pyarrow`, `boto3`).
3. Reproduce EXP-003 / re-run experiments:
   - `python scripts/run_exp003.py` — EXP-003 LightGBM benchmark on the validation split.
   - `python scripts/test_smoke.py` — fast pipeline smoke test on a small slice.
4. Generate the test-set submission locally:
   - `python scripts/run_test_inference.py` — writes `output/matching_results.tsv` + `output/candidate_pairs.tsv`.
   - `python scripts/validate_submission.py` — full integrity validation (format, coverage, no leakage).
   - `python scripts/package_submission.py` — builds `submission/amazon_ml_submission.zip`.
5. Or run test inference on AWS SageMaker (see `scripts/aws/` and `docs/aws_exp003_run.md`):
   - `python scripts/aws/upload_dataset.py` → `python scripts/aws/submit_processing_job.py` → `python scripts/aws/monitor_job.py` → `python scripts/aws/collect_outputs.py`.

All scripts compute paths from their own location, so they run from any clone/machine.