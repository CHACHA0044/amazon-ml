"""Test SageMaker instance quotas in eu-north-1 (probe only; not the project region)."""
import boto3
from botocore.exceptions import ClientError

from aws_config import BUCKET, ROLE_ARN, require_config

require_config("BUCKET", "ROLE_ARN")

sm = boto3.client("sagemaker", region_name="eu-north-1")
image_uri = "669576153137.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

instances = [
    "ml.m5.xlarge",
    "ml.m5.2xlarge",
    "ml.c5.xlarge",
    "ml.c5.2xlarge",
    "ml.m4.xlarge",
    "ml.t3.medium",
]

print("=== CHECKING PROCESSING JOB QUOTAS IN EU-NORTH-1 ===")
for inst in instances:
    try:
        sm.create_processing_job(
            ProcessingJobName="quota-test-dummy-eunorth",
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
        print(f"  eu-north-1 PROCESSING {inst}: ACCEPTED!", flush=True)
        sm.stop_processing_job(ProcessingJobName="quota-test-dummy-eunorth")
    except ClientError as e:
        msg = e.response["Error"]["Message"]
        code = e.response["Error"]["Code"]
        if "is 0 Instances" in msg:
            print(f"  eu-north-1 PROCESSING {inst}: QUOTA 0", flush=True)
        else:
            print(f"  eu-north-1 PROCESSING {inst}: {code} - {msg}", flush=True)

print("\n=== CHECKING TRAINING JOB QUOTAS IN EU-NORTH-1 ===")
for inst in instances:
    try:
        sm.create_training_job(
            TrainingJobName="quota-test-dummy-eunorth-train",
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
        print(f"  eu-north-1 TRAINING {inst}: ACCEPTED!", flush=True)
        sm.stop_training_job(TrainingJobName="quota-test-dummy-eunorth-train")
    except ClientError as e:
        msg = e.response["Error"]["Message"]
        code = e.response["Error"]["Code"]
        if "is 0 Instances" in msg:
            print(f"  eu-north-1 TRAINING {inst}: QUOTA 0", flush=True)
        else:
            print(f"  eu-north-1 TRAINING {inst}: {code} - {msg}", flush=True)
