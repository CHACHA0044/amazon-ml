"""Monitor EC2 batch worker and wait for outputs in S3."""
import json
import os
import sys
import time

import boto3

from aws_config import REGION, BUCKET, S3_PREFIX, REPO_ROOT

def monitor(instance_id: str = None):
    if not instance_id:
        meta_path = os.path.join(REPO_ROOT, "experiments", "EXP003-AWS", "latest_job.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                instance_id = json.load(f)["instance_id"]
        else:
            raise FileNotFoundError("Could not find latest_job.json")

    session = boto3.Session(region_name=REGION)
    ec2 = session.client("ec2")
    s3 = session.client("s3")

    print(f"Monitoring EC2 batch worker {instance_id} in {REGION}...")
    print(f"Target S3 output path: s3://{BUCKET}/{S3_PREFIX}/outputs/")
    
    t_start = time.time()
    last_state = None

    while True:
        elapsed = time.time() - t_start
        # 1. Check EC2 status
        try:
            desc = ec2.describe_instances(InstanceIds=[instance_id])
            state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
        except Exception as e:
            state = f"Error: {e}"

        if state != last_state:
            print(f"[{time.strftime('%H:%M:%S')}] Instance State: {state} (Elapsed: {elapsed/60:.1f} min)", flush=True)
            last_state = state

        # 2. Check S3 for output files
        try:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{S3_PREFIX}/outputs/")
            contents = resp.get("Contents", [])
            filenames = [os.path.basename(c["Key"]) for c in contents if c["Key"] != f"{S3_PREFIX}/outputs/"]
            if filenames:
                print(f"[{time.strftime('%H:%M:%S')}] S3 Outputs detected: {filenames}", flush=True)
                if "matching_results.tsv" in filenames and "run_summary.json" in filenames:
                    print("\nInference run completed and outputs uploaded to S3!", flush=True)
                    return True
        except Exception as e:
            print(f"S3 check error: {e}", flush=True)

        if state in ["terminated", "stopped", "shutting-down"]:
            print(f"\nInstance reached {state} state.", flush=True)
            # Final S3 check
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{S3_PREFIX}/outputs/")
            filenames = [os.path.basename(c["Key"]) for c in resp.get("Contents", [])]
            print(f"Final S3 objects in outputs: {filenames}")
            return "matching_results.tsv" in filenames

        time.sleep(30)


if __name__ == "__main__":
    inst_id = sys.argv[1] if len(sys.argv) > 1 else None
    monitor(inst_id)
