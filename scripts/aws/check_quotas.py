"""Test SageMaker Training and Processing instance quotas in the project region."""
import boto3
from botocore.exceptions import ClientError

from aws_config import REGION, BUCKET, ROLE_ARN, require_config

require_config("BUCKET", "ROLE_ARN")

sm = boto3.client("sagemaker", region_name=REGION)
image_uri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

instances = [
    "ml.m5.xlarge",
    "ml.m4.xlarge",
    "ml.c5.xlarge",
    "ml.m5.2xlarge",
    "ml.c5.2xlarge",
    "ml.c5.4xlarge",
    "ml.m5.4xlarge",
    "ml.t3.medium",
    "ml.t3.large",
    "ml.t3.xlarge",
    "ml.t3.2xlarge",
]

print("=== CHECKING PROCESSING JOB QUOTAS ===")
for inst in instances:
    try:
        sm.create_processing_job(
            ProcessingJobName="quota-test-dummy",
            AppSpecification={"ImageUri": image_uri},
            RoleArn=ROLE_ARN,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": inst,
                    "VolumeSizeInGB": 10,
                }
            },
            StoppingCondition={"MaxRuntimeInSeconds": 60},
        )
        print(f"  PROCESSING {inst}: ACCEPTED!")
        sm.stop_processing_job(ProcessingJobName="quota-test-dummy")
    except ClientError as e:
        msg = e.response["Error"]["Message"]
        if "is 0 Instances" in msg:
            print(f"  PROCESSING {inst}: QUOTA 0")
        else:
            print(f"  PROCESSING {inst}: {e.response['Error']['Code']} - {msg}")

print("\n=== CHECKING TRAINING JOB QUOTAS ===")
for inst in instances:
    try:
        sm.create_training_job(
            TrainingJobName="quota-test-dummy-train",
            AlgorithmSpecification={
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
            },
            RoleArn=ROLE_ARN,
            OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/quotas-test/"},
            ResourceConfig={
                "InstanceType": inst,
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 60},
        )
        print(f"  TRAINING {inst}: ACCEPTED!")
        sm.stop_training_job(TrainingJobName="quota-test-dummy-train")
    except ClientError as e:
        msg = e.response["Error"]["Message"]
        if "is 0 Instances" in msg:
            print(f"  TRAINING {inst}: QUOTA 0")
        else:
            print(f"  TRAINING {inst}: {e.response['Error']['Code']} - {msg}")
