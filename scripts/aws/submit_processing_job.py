"""Submit a SageMaker Processing Job for EXP-003 test inference with cost guardrails.

Follows Phase 4 requirements:
- Cost guardrails: CPU instance (ml.m5.2xlarge: 8 vCPUs, 32 GB RAM, ~$0.46/hr)
- Hard timeout: MaxRuntimeInSeconds=7200 (2 hours)
- Standard SageMaker Scikit-learn container
- Standard S3 input/output channel configuration
- Tags: Project=AmazonMLChallenge, Experiment=EXP003-AWS, Purpose=CompetitionSubmission
"""
import json
import os
import sys
import time

import boto3

from aws_config import REGION, BUCKET, S3_PREFIX, ROLE_ARN, REPO_ROOT, require_config

require_config("BUCKET", "ROLE_ARN")

# AWS Managed Scikit-learn Processing Container in us-east-1
SKLEARN_IMAGE_URI = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

INSTANCE_TYPE = "ml.m5.2xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 40
MAX_RUNTIME_SECONDS = 7200  # 2 hours hard timeout guardrail


def print_guardrail_notice():
    print("=" * 70)
    print("PHASE 4 — AWS SAGEMAKER COST & COMPUTE SPECIFICATION")
    print("=" * 70)
    print(f"AWS Region:         {REGION}")
    print(f"Instance Type:      {INSTANCE_TYPE} (8 vCPUs, 32 GB RAM)")
    print(f"Instance Count:     {INSTANCE_COUNT}")
    print(f"EBS Volume Size:    {VOLUME_SIZE_GB} GB")
    print(f"Max Runtime Limit:  {MAX_RUNTIME_SECONDS}s ({MAX_RUNTIME_SECONDS/3600:.1f} hours hard timeout)")
    print(f"Estimated Cost:     ~$0.46/hr (~$0.92 max for 2 hrs, covered by $140 credits)")
    print(f"Estimated Workload: 1,732,544 test S1 queries across France, India, US")
    print(f"Expected Outputs:   matching_results.tsv, candidate_pairs.tsv, run_summary.json")
    print(f"Reason for Selection: 32 GB RAM avoids the memory paging that stalled local execution")
    print(f"                    Single-node CPU eliminates distributed complexity while finishing fast")
    print("=" * 70)


def upload_entrypoint(s3_client, local_script_path: str) -> str:
    s3_key = f"{S3_PREFIX}/code/sagemaker_entrypoint.py"
    print(f"Uploading entrypoint script to s3://{BUCKET}/{s3_key}...", flush=True)
    s3_client.upload_file(local_script_path, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"


def submit_processing_job(job_name: str = None) -> str:
    print_guardrail_notice()

    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    local_entrypoint = os.path.join(REPO_ROOT, "scripts", "aws", "sagemaker_entrypoint.py")
    entrypoint_s3_uri = upload_entrypoint(s3, local_entrypoint)

    if not job_name:
        job_name = f"amazonml-exp003-{int(time.time())}"

    # S3 Inputs
    processing_inputs = [
        {
            "InputName": "code",
            "AppManaged": False,
            "S3Input": {
                "S3Uri": entrypoint_s3_uri,
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "models",
            "AppManaged": False,
            "S3Input": {
                "S3Uri": f"s3://{BUCKET}/{S3_PREFIX}/models/",
                "LocalPath": "/opt/ml/processing/input/models",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "preprocessed",
            "AppManaged": False,
            "S3Input": {
                "S3Uri": f"s3://{BUCKET}/{S3_PREFIX}/experiments/EXP003-AWS/checkpoints/01_preprocessed/",
                "LocalPath": "/opt/ml/processing/input/preprocessed",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
    ]

    # S3 Output
    processing_output_config = {
        "Outputs": [
            {
                "OutputName": "inference_outputs",
                "S3Output": {
                    "S3Uri": f"s3://{BUCKET}/{S3_PREFIX}/outputs/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
                "AppManaged": False,
            }
        ]
    }

    # Resource Config
    processing_resources = {
        "ClusterConfig": {
            "InstanceCount": INSTANCE_COUNT,
            "InstanceType": INSTANCE_TYPE,
            "VolumeSizeInGB": VOLUME_SIZE_GB,
        }
    }

    # App Specification
    app_specification = {
        "ImageUri": SKLEARN_IMAGE_URI,
        "ContainerEntrypoint": [
            "python3",
            "/opt/ml/processing/input/code/sagemaker_entrypoint.py",
        ],
    }

    # Stopping Condition (Hard Guardrail)
    stopping_condition = {
        "MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS,
    }

    # Tags
    tags = [
        {"Key": "Project", "Value": "AmazonMLChallenge"},
        {"Key": "Experiment", "Value": "EXP003-AWS"},
        {"Key": "Purpose", "Value": "CompetitionSubmission"},
    ]

    print(f"\nSubmitting SageMaker Processing Job: {job_name}...", flush=True)
    response = sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig=processing_output_config,
        ProcessingResources=processing_resources,
        StoppingCondition=stopping_condition,
        AppSpecification=app_specification,
        RoleArn=ROLE_ARN,
        Tags=tags,
    )

    print(f"Processing Job Created! ARN: {response['ProcessingJobArn']}")
    print(f"View in AWS Console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/processing-jobs/{job_name}")

    # Save job metadata locally
    job_meta_path = os.path.join(REPO_ROOT, "experiments", "EXP003-AWS", "latest_job.json")
    os.makedirs(os.path.dirname(job_meta_path), exist_ok=True)
    with open(job_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "job_name": job_name,
                "job_arn": response["ProcessingJobArn"],
                "region": REGION,
                "bucket": BUCKET,
                "instance_type": INSTANCE_TYPE,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )
    print(f"Job metadata saved to {job_meta_path}")
    return job_name


if __name__ == "__main__":
    submit_processing_job()
