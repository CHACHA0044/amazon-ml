"""Launch High-Performance Dedicated C++ OpenMP Batch Compute on AWS EC2."""
import base64
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

from aws_config import REGION, BUCKET, S3_PREFIX, REPO_ROOT, require_config

require_config("BUCKET")

INSTANCE_TYPE = "r5.2xlarge"  # 8 vCPUs, 32 GB RAM
AMI_ID = "ami-05a3e9423ae4d7a19"  # Ubuntu 22.04 LTS us-east-1
INSTANCE_PROFILE_NAME = "AmazonML-EC2-S3-Profile"
VOLUME_SIZE_GB = 40


def build_cpp_user_data_script() -> str:
    return f"""#!/bin/bash
export AWS_DEFAULT_REGION={REGION}
export AWS_REGION={REGION}

exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================================="
echo "STARTING AMAZON ML CHALLENGE C++17 OPENMP BATCH INFERENCE"
echo "Host: $(hostname), CPU: $(nproc), RAM: $(free -h | awk '/Mem:/ {{print $2}}')"
date -u
echo "=========================================================="

export DEBIAN_FRONTEND=noninteractive
fallocate -l 16G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=16384
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

apt-get update -y
apt-get install -y build-essential g++ awscli

mkdir -p /opt/amazonml/src/native
mkdir -p /opt/amazonml/dataset/test
mkdir -p /opt/amazonml/output
cd /opt/amazonml

echo "1. Downloading raw test files and C++ sources from S3..."
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/raw/test/test_source1.tsv /opt/amazonml/dataset/test/test_source1.tsv
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/raw/test/test_source2.tsv /opt/amazonml/dataset/test/test_source2.tsv
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/raw/test/test_source3.tsv /opt/amazonml/dataset/test/test_source3.tsv
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/src/native/matcher.cpp /opt/amazonml/src/native/matcher.cpp
aws s3 cp s3://{BUCKET}/{S3_PREFIX}/src/native/lgbm_model.h /opt/amazonml/src/native/lgbm_model.h

echo "2. Compiling C++17 OpenMP binary with -O3 -march=native..."
g++ -O3 -march=native -std=c++17 -fopenmp -Isrc/native src/native/matcher.cpp -o matcher

echo "3. Executing native C++ matcher across all 8 cores..."
date -u
./matcher /opt/amazonml/dataset/test /opt/amazonml/output

echo "4. Syncing generated submission artifacts to S3..."
date -u
aws s3 cp /opt/amazonml/output/ s3://{BUCKET}/{S3_PREFIX}/outputs/ --recursive
aws s3 cp /var/log/user-data.log s3://{BUCKET}/{S3_PREFIX}/outputs/execution.log

echo "=========================================================="
echo "C++ INFERENCE COMPLETED SUCCESSFULLY! SHUTTING DOWN."
date -u
echo "=========================================================="
sudo poweroff
"""


def launch_cpp_worker() -> str:
    session = boto3.Session(region_name=REGION)
    ec2 = session.client("ec2")

    user_data = build_cpp_user_data_script()

    print(f"Launching {INSTANCE_TYPE} (8 vCPUs) C++ Worker in {REGION}...", flush=True)
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
                    {"Key": "Name", "Value": "AmazonML-Cpp-Inference-Worker"},
                    {"Key": "Project", "Value": "AmazonMLChallenge"},
                    {"Key": "Experiment", "Value": "EXP003-Cpp-AWS"},
                ],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"Instance launched successfully! ID: {instance_id}")

    job_meta = {
        "job_name": f"ec2-cpp-{instance_id}",
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
    launch_cpp_worker()
