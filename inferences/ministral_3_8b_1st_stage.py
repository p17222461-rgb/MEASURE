import os
import ollama
from utils import pydantic_schema

PROMPT = """
Extract structured information from this resume image.
Return ONLY the extracted information in JSON format following the provided schema.
Do NOT translate, rewrite, or summarize any text: extract exactly as shown.
Maintain the original language for all extracted text.
Do not hallucinate any information not visible in the image.

Your response must strictly follow this JSON structure:

{
  "personal_info": {
    "full_name": 
    "email": 
    "phone": 
    "location": 
    "github": 
    "linkedin": 
    "address":
    "identification_number":
  },
  "skills": [

  ],
  "work_experience": [
    {
      "company": 
      "role": 
      "start_date": 
      "end_date": 
      "employment_description": 
      "location": 
    },
    {
      "company": 
      "role": 
      "start_date": 
      "end_date":
      "employment_description": 
      "location": 
    }
  ],
  "education": {
    "degrees": [
      {
        "institution": 
        "degree": 
        "start_date": 
        "end_date":
        "location": 
      }
    ],
    "certifications": [
      {
        "name":
        "institution": 
        "date_obtained": 
        "license": 
      }
    ]
  }
}

Now extract the resume data from the provided images and return ONLY the JSON object.
"""

def process_one_resume(image_b64: str, filename: str, model: str, num_ctx: int) -> dict:
    json_schema = pydantic_schema.ResumeData.model_json_schema()

    response = ollama.chat(
        model=model,
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        messages=[
            {
                "role": "system",
                "content": "You are a precise resume parser. Return ONLY valid JSON that follows the schema."
            },
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],          # single image as base64 string
            },
        ],
        format=json_schema,
    )

    raw = response.message.content
    if not raw or not raw.strip():
        raise ValueError(f"Model returned empty output for '{filename}'")

    validated = pydantic_schema.ResumeData.model_validate_json(raw)
    return validated.model_dump()


def run(job_input: dict) -> dict:
    """
    Expected job_input keys:
        - records  : list[dict]   Each record: { "filename": "cv_name", "image_b64": "..." }
        - model    : str          (optional, default "ministral-3:8b")
        - num_ctx  : int          (optional, default 32768)

    Returns:
        { cv_id: {"status": "ok",    "data": {...}}
                | {"status": "error", "error": "..."} }
    """
    model   = job_input.get("model",   os.environ.get("OLLAMA_MODEL",   "ministral-3:8b"))
    num_ctx = int(job_input.get("num_ctx", os.environ.get("OLLAMA_NUM_CTX", 32768)))
    records = job_input.get("records", [])

    if not records:
        raise ValueError("Job payload must include 'records'.")

    results = {}
    for rec in records:
        filename  = rec["filename"]
        image_b64 = rec["image_b64"]
        print(f"  Processing: {filename}")

        try:
            parsed = process_one_resume(image_b64, filename, model, num_ctx)
            results[filename] = {"status": "ok", "data": parsed}
            print(f"    ✓ Done")
        except Exception as e:
            results[filename] = {"status": "error", "error": str(e)}
            print(f"    ✗ Error: {e}")

    ok  = sum(1 for v in results.values() if v["status"] == "ok")
    err = sum(1 for v in results.values() if v["status"] == "error")
    print(f"\nJob finished — OK: {ok}, Errors: {err}")
    return results