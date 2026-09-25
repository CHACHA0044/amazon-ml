"""Monitor SageMaker Processing Job and stream CloudWatch logs until completion."""
import json
import os
import sys
import time

import boto3

from aws_config import REGION, REPO_ROOT


def get_latest_job_name() -> str:
    meta_path = os.path.join(REPO_ROOT, "experiments", "EXP003-AWS", "latest_job.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)["job_name"]
    raise FileNotFoundError(f"Job metadata not found at {meta_path}")


def monitor_job(job_name: str = None):
    if not job_name:
        job_name = get_latest_job_name()

    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    cw_logs = session.client("logs")

    print(f"Monitoring SageMaker Processing Job: {job_name} ({REGION})...")
    log_group_name = "/aws/sagemaker/ProcessingJobs"

    last_status = None
    stream_tokens = {}

    while True:
        desc = sm.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]

        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Job Status: {status}", flush=True)
            last_status = status

        # Try to stream CloudWatch logs
        try:
            streams = cw_logs.describe_log_streams(
                logGroupName=log_group_name,
                logStreamNamePrefix=job_name,
                orderBy="LogStreamName",
            ).get("logStreams", [])

            for s in streams:
                s_name = s["logStreamName"]
                kwargs = {
                    "logGroupName": log_group_name,
                    "logStreamName": s_name,
                    "startFromHead": True,
                }
                if s_name in stream_tokens:
                    kwargs["nextToken"] = stream_tokens[s_name]

                try:
                    events_resp = cw_logs.get_log_events(**kwargs)
                    for event in events_resp.get("events", []):
                        msg = event["message"].rstrip()
                        print(f"  [CloudWatch] {msg}", flush=True)
                    stream_tokens[s_name] = events_resp.get("nextForwardToken")
                except Exception:
                    pass
        except Exception:
            pass

        if status in ["Completed", "Failed", "Stopped"]:
            if status == "Failed":
                print(f"ERROR: Job failed! Failure reason: {desc.get('FailureReason', 'Unknown')}")
            else:
                print(f"\nJob finished with status: {status}!")
            return status

        time.sleep(15)


if __name__ == "__main__":
    job_name = sys.argv[1] if len(sys.argv) > 1 else None
    monitor_job(job_name)
