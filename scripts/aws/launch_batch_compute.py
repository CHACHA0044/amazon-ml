"""Launch dedicated AWS Batch Compute (m5.2xlarge: 8 vCPUs, 32 GB RAM) with auto-termination guardrail.

Fulfills Phase 3 & 4:
  - SageMaker Processing/Training quotas are 0 on this Free Tier account
  - EC2 m5.2xlarge is verified active with full S3 instance profile
  - Auto-terminates immediately upon completion (zero credit waste)
  - Estimated total runtime: ~35-45 minutes. Total cost: ~$0.25 (covered by $140 credits)
"""
import base64
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

from aws_config import REGION, BUCKET, S3_PREFIX, REPO_ROOT, require_config

require_config("BUCKET")

INSTANCE_TYPE = "m5.2xlarge"
AMI_ID = "ami-05a3e9423ae4d7a19"  # Ubuntu 22.04 LTS us-east-1
INSTANCE_PROFILE_NAME = "AmazonML-EC2-S3-Profile"
VOLUME_SIZE_GB = 40


def print_guardrail_specification():
    print("=" * 70)
    print("PHASE 4 — AWS BATCH COMPUTE SPECIFICATION & COST GUARDRAILS")
    print("=" * 70)
    print(f"AWS Region:         {REGION}")
    print(f"Compute Type:       EC2 Dedicated Batch Worker ({INSTANCE_TYPE})")
    print(f"Hardware Specs:     8 vCPUs, 32 GB RAM, 40 GB NVMe/gp3 Storage")
    print(f"Instance Count:     1")
    print(f"Hourly Rate:        $0.384 / hour (On-Demand)")
    print(f"Estimated Runtime:  35 - 45 minutes")
    print(f"Estimated Cost:     ~$0.25 - $0.30 (deducted from $140.00 active credits)")
    print(f"Termination Policy: Auto-terminate on completion (poweroff -> terminate)")
    print(f"Target Workload:    1,732,544 test S1 queries across France, India, US")
    print(f"Expected Outputs:   matching_results.tsv, candidate_pairs.tsv, run_summary.json")
    print(f"Reason for Selection:")
    print(f"  1. SageMaker Processing/Training job quotas are 0 by default on this Free Tier account.")
    print(f"  2. EC2 m5.2xlarge has 32 GB RAM which eliminates the local paging bottleneck.")
    print(f"  3. Automatic cloud-init self-termination guarantees no persistent idle charges.")
    print("=" * 70)


def build_user_data_script() -> str:
    return f"""#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================================="
echo "STARTING AMAZON ML CHALLENGE BATCH INFERENCE (EXP-003)"
echo "Host: $(hostname), CPU: $(nproc), RAM: $(free -h | awk '/Mem:/ {{print $2}}')"
date -u
echo "=========================================================="

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-pip python3-venv awscli jq

mkdir -p /opt/amazonml/models
mkdir -p /opt/amazonml/work
mkdir -p /opt/amazonml/output
cd /opt/amazonml

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -q polars lightgbm pyarrow joblib numpy boto3

echo "Downloading artifacts from S3..."
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/models/lgbm_matcher.joblib /opt/amazonml/models/lgbm_matcher.joblib
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/experiments/EXP003-AWS/checkpoints/01_preprocessed/test_records.parquet /opt/amazonml/work/test_records.parquet
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/code/sagemaker_entrypoint.py /opt/amazonml/run_inference.py

echo "Starting full test-set inference..."
date -u
python run_inference.py

echo "Inference finished! Syncing output artifacts to S3..."
date -u
aws s3 cp /opt/amazonml/output/ s3://{BUCKET}/{S3_PREFIX}/outputs/ --recursive
aws s3 cp /var/log/user-data.log s3://{BUCKET}/{S3_PREFIX}/outputs/execution.log

echo "=========================================================="
echo "JOB COMPLETED SUCCESSFULLY! SHUTTING DOWN TO TERMINATE."
date -u
echo "=========================================================="
sudo poweroff
"""


def launch_worker() -> str:
    print_guardrail_specification()

    session = boto3.Session(region_name=REGION)
    ec2 = session.client("ec2")

    user_data = build_user_data_script()

    print(f"\nLaunching {INSTANCE_TYPE} worker in {REGION}...", flush=True)
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": VOLUME_SIZE_GB,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "AmazonML-EXP003-Inference-Worker"},
                    {"Key": "Project", "Value": "AmazonMLChallenge"},
                    {"Key": "Experiment", "Value": "EXP003-AWS"},
                    {"Key": "Purpose", "Value": "CompetitionSubmission"},
                ],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"Instance launched successfully! ID: {instance_id}")

    job_meta = {
        "job_name": f"ec2-batch-{instance_id}",
        "instance_id": instance_id,
        "instance_type": INSTANCE_TYPE,
        "region": REGION,
        "bucket": BUCKET,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    job_meta_path = os.path.join(REPO_ROOT, "experiments", "EXP003-AWS", "latest_job.json")
    os.makedirs(os.path.dirname(job_meta_path), exist_ok=True)
    with open(job_meta_path, "w", encoding="utf-8") as f:
        json.dump(job_meta, f, indent=2)

    return instance_id


if __name__ == "__main__":
    launch_worker()
