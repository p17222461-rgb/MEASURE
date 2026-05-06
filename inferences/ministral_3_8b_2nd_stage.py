import os
import re
import ollama
from collections import defaultdict
from utils import pydantic_schema

prompt_w_context_between_pages = """
You are given multiple images that together form a single resume (CV).
The pages are provided in order and must be interpreted as ONE document.

Extract structured information from the COMPLETE resume using ALL pages.
Information may appear on any page.

Rules:
- Return ONLY valid JSON. No markdown, no code fences, no extra text.
- Do NOT translate, rewrite, or summarize text.
- Preserve the original language exactly as shown.
- Do NOT hallucinate or infer missing information. Use empty string "" for missing fields.
- Do NOT duplicate entries across pages.
- If a section continues across pages, merge it correctly.
- The output must represent the full resume as a single entity.
- Known languages must be included in the skills section. 

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


def extract_cv_id(filename: str) -> str | None:
    match = re.match(r"(.*)_page_(\d+)$", str(filename), re.IGNORECASE)
    return match.group(1) if match else None


def extract_page_number(filename: str) -> int:
    match = re.match(r"(.*)_page_(\d+)$", str(filename), re.IGNORECASE)
    return int(match.group(2)) if match else 0


def group_pages_by_cv(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        cv_id = extract_cv_id(row.get("filename", ""))
        if cv_id:
            grouped[cv_id].append(row)

    for cv_id in grouped:
        grouped[cv_id] = sorted(grouped[cv_id], key=lambda r: extract_page_number(r["filename"]))

    return dict(grouped)


def process_resume_images(cv_name: str, pages: list[dict], model: str, num_ctx: int) -> dict:
    json_schema = pydantic_schema.ResumeData.model_json_schema()

    # Ollama expects raw base64 strings in the images list
    images_b64 = [page["image_b64"] for page in pages]

    print(f"  [{cv_name}] pages: {len(pages)}, model: {model}, ctx: {num_ctx}")

    response = ollama.chat(
        model=model,
        options={
            "temperature": 0.0,
            "num_ctx": num_ctx,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise resume parser.\n"
                    "You will receive MULTIPLE images that together form ONE complete CV.\n"
                    "Use information across ALL pages.\n"
                    "Return ONLY valid JSON that follows the schema."
                ),
            },
            {
                "role": "user",
                "content": prompt_w_context_between_pages,
                "images": images_b64,
            },
        ],
        format=json_schema,
    )

    raw = response.message.content
    if not raw or not raw.strip():
        raise ValueError(f"Model returned empty output for CV '{cv_name}'")

    validated = pydantic_schema.ResumeData.model_validate_json(raw)
    return validated.model_dump()


# ── Entry point called by main_end_to_end.py ─────────────────────────────────

def run(job_input: dict) -> dict:
    """
    Expected job_input keys:
        - records  : list[dict]   Each record: { "filename": "cv_name_page_N", "image_b64": "..." }
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

    grouped = group_pages_by_cv(records)
    print(f"\n{len(grouped)} CV(s) found, model: {model}, ctx: {num_ctx}")

    results = {}
    for cv_name, pages in grouped.items():
        print(f"\nProcessing: {cv_name} ({len(pages)} page(s))")
        try:
            parsed = process_resume_images(cv_name, pages, model, num_ctx)
            results[cv_name] = {"status": "ok", "data": parsed}
            print(f"  ✓ Done")
        except Exception as e:
            results[cv_name] = {"status": "error", "error": str(e)}
            print(f"  ✗ Error: {e}")

    ok  = sum(1 for v in results.values() if v["status"] == "ok")
    err = sum(1 for v in results.values() if v["status"] == "error")
    print(f"\nJob finished — OK: {ok}, Errors: {err}")

    return results