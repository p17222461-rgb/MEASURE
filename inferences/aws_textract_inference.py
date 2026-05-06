import boto3
import asyncio
import json
from pathlib import Path
import os

textract = boto3.client("textract")
s3 = boto3.client("s3")

BUCKET = ""

async def start_textract_job(s3_key):
    response = textract.start_document_analysis(
        DocumentLocation={
            "S3Object": {"Bucket": BUCKET, 
                         "Name": s3_key}
        },
        FeatureTypes=["TABLES", "LAYOUT"]
    )
    return response["JobId"]

async def wait_for_job(job_id):
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]

        if status in ("SUCCEEDED", "FAILED"):
            return response
        
        await asyncio.sleep(2)


async def process_image(s3_key):
    print(f"Starting Textract job for: {s3_key}")
    job_id = await start_textract_job(s3_key)

    print(f"Waiting for job completion: {job_id}")
    result = await wait_for_job(job_id)
    os.makedirs("textract_results", exist_ok=True)
    output_filename = Path(s3_key).stem + ".json"
    output_path = os.path.join("textract_results", output_filename)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {output_path}")


async def main():
    paginator = s3.get_paginator("list_objects_v2")
    png_objects = []

    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith(".png"):
                png_objects.append({
                    "key": key,
                    "last_modified": obj["LastModified"]
                })

    png_objects.sort(
        key=lambda x: x["last_modified"],
        reverse=True
    )

    recent_pngs = png_objects[:105]
    for obj in recent_pngs:
        print(f"  • {obj['key']} ({obj['last_modified']})")

    semaphore = asyncio.Semaphore(5)

    async def sem_task(s3_key: str):
        async with semaphore:
            await process_image(s3_key)

    await asyncio.gather(
        *(sem_task(obj["key"]) for obj in recent_pngs)
    )

if __name__ == "__main__":
    asyncio.run(main())