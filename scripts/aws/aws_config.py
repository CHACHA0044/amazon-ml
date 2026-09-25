"""Resolve AWS / SageMaker configuration without hard-coding account-specific values.

Never commit account IDs, bucket names, or role ARNs to source code. This module
resolves configuration from, in priority order:

1. Environment variables
2. A git-ignored local JSON config file: ``experiments/EXP003-AWS/local_aws_config.json``
   or ``<repo>/.amazonml-aws-config.json``

Environment variables (documented for collaborators):

- ``AMAZONML_REGION``              (default: ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` / ``us-east-1``)
- ``AMAZONML_SAGEMAKER_BUCKET``    required, e.g. ``sagemaker-<region>-<account-id>``
- ``AMAZONML_SAGEMAKER_PREFIX``    (default: ``amazon-ml-challenge``)
- ``AMAZONML_SAGEMAKER_ROLE_ARN``  required to submit jobs

Example local config file (git-ignored):

.. code-block:: json

    {
      "region": "us-east-1",
      "bucket": "sagemaker-us-east-1-<account-id>",
      "s3_prefix": "amazon-ml-challenge",
      "role_arn": "arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-..."
    }
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_local_config() -> dict:
    for name in ("experiments/EXP003-AWS/local_aws_config.json", ".amazonml-aws-config.json"):
        path = REPO_ROOT / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


_LOCAL_CFG = _load_local_config()


def _resolve(env_names, local_key, default=None):
    for key in env_names:
        value = os.environ.get(key)
        if value:
            return value
    value = _LOCAL_CFG.get(local_key)
    return value if value else default


REGION = _resolve(["AMAZONML_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"], "region", "us-east-1")
BUCKET = _resolve(["AMAZONML_SAGEMAKER_BUCKET"], "bucket")
S3_PREFIX = _resolve(["AMAZONML_SAGEMAKER_PREFIX"], "s3_prefix", "amazon-ml-challenge")
ROLE_ARN = _resolve(["AMAZONML_SAGEMAKER_ROLE_ARN"], "role_arn")


def require_config(*names):
    """Raise SystemExit if any required config value is missing."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise SystemExit(
            "Missing SageMaker configuration: " + ", ".join(missing)
            + ". Set the corresponding AMAZONML_* environment variable(s) "
            "or create experiments/EXP003-AWS/local_aws_config.json."
        )