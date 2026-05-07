# MEASURE: Multi-stage Evaluation for Assessing Structured Understanding in Resume Extraction

MEASURE is a multi-stage benchmark and evaluation framework for resume parsing that systematically evaluates multiple Document Understanding (DU) components under a unified framework. It covers **layout analysis**, **reading order**, **text extraction**, and **semantic understanding**.

The benchmark evaluates **fourteen systems** across four categories: seven OCR engines, one end-to-end DU model, five general-purpose Vision-Language Models (VLMs), and one commercial resume parsing service (RChilli).

---

### Dataset

The MEASURE dataset consists of **112 manually annotated real-world resumes** (218 page images), sourced from an internal database of technology-role candidates (Machine Learning, Data Science, Software Engineering, and Data Engineering). Resumes are in English or Spanish and were selected to maximize variation in visual layout and formatting.

Annotations were performed using **Label Studio** with SuryaOCR-assisted pre-annotation, and cover:

- **Item-level labels**: NAME, ADR, MAI, NMR, LAN, LOC, DAT, DUR, ORG, ROL, EDU, CFR, O
- **Section-level labels**: Personal Information, Employment Information, Employment Description, Education, Skills, Summary, O
- **Reading order indices** per bounding box
- **BIO schema** for multi-line entity spans

> ⚠️ The dataset is currently undergoing anonymization and is not yet publicly available. Upon acceptance, an anonymized version will be released. The full codebase for inference and evaluation is available here and can be run on any similarly structured CV data.
> 

---

### Systems Evaluated

| Type | Models |
| --- | --- |
| OCR Engines | AWS Textract, SuryaOCR, EasyOCR, DotsOCR, Tesseract, Mistral Basic OCR, DotsMOCR |
| End-to-end DU | Donut |
| NER Models | LayoutLMv3, LiLT |
| Reading-order | LayoutReader (combined with SuryaOCR, EasyOCR, and Tesseract bboxes) |
| VLMs | GPT-4.1-mini, Gemini-2.5-Flash, Gemini-2.5-Pro, LiquidAI/LFM2.5-VL-1.6B, Ministral-3:8B |
| Commercial Parser | RChilli |

---

### Evaluation Metrics

**Layout Analysis**: Mean Average Precision (mAP) at IoU thresholds of 0.50 and 0.75, using 11-point interpolation. Evaluated on systems that expose bounding boxes (SuryaOCR, EasyOCR, Tesseract, AWS Textract). Word-level outputs are reconstructed to line level using a vertical tolerance factor (ytol=0.6) before evaluation.

**Reading Order**: Normalized Damerau–Levenshtein Distance (NDLD) at word level, and corpus-level BLEU (4-gram). The predicted text sequence is sorted by model output order and compared against the ground-truth reading-order index sequence.

**Text Extraction**: Word Error Rate (WER) and Character Error Rate (CER). WER uses normalized text (lowercase, no diacritics, no punctuation). CER uses raw text without punctuation removal.

**Semantic Understanding (VLMs)**: BERTScore computed per section type (Personal Information, Skills, Education, Employment Information, Employment Description), averaged across sections per resume, then averaged across all resumes.

---

### VLM Evaluation Settings

VLMs were evaluated in two settings to analyze the effect of document-level context:

- **1st setting**: Each resume page is processed independently (page-by-page inference)
- **2nd setting**: All pages of a resume are provided in a single inference call (full document context)

Most VLMs were prompted to generate structured JSON output following a predefined schema (defined in `utils/pydantic_schema.py`).

**Compute**: LiquidAI ran on a MacBook (Apple M3 Pro, 18GB RAM, ~10 hours). Ministral-3:8B ran on RunPod (single L40 GPU, 48GB VRAM, ~12 minutes). GPT and Gemini API inference took ~30 minutes each.

---

### Repository Structure

```
MEASURE/
├── evaluation_pipeline.ipynb     # Main evaluation: OCR, layout, reading order, NER
├── vlms_evaluation.ipynb         # VLM semantic evaluation (BERTScore)
├── pyproject.toml                # Dependencies and package metadata
├── data/                         # Dataset artifacts (not included — see Dataset note above)
│   ├── resumes_images/           # Document page images
│   ├── ground_truth/             # Label Studio exports / GT CSV files
│   └── ocr_outputs/              # JSON/CSV outputs from OCR engines
├── inferences/
│   ├── aws_textract_inference.py         # Submit docs to AWS Textract via S3
│   ├── liquidai_inference.ipynb          # LiquidAI/LFM2.5-VL-1.6B inference
│   ├── ministral_3_8b_1st_stage.py       # Ministral-3:8B — 1st evaluation setting
│   ├── ministral_3_8b_2nd_stage.py       # Ministral-3:8B — 2nd evaluation setting
│   ├── RChilli_analysis.ipynb            # RChilli inference and evaluation
│   └── vlms_api_inference.ipynb          # GPT-4.1-mini, Gemini-2.5-Pro/Flash inference
└── utils/
    ├── df_to_structure_json.py    # GT/prediction → structured JSON for end-to-end metrics
    ├── helpers.py                 # PDF-to-image conversion, Label Studio cleanup
    ├── inferences.py              # LayoutLMv3, Donut, LiLT, LayoutReader inference classes
    ├── layout_metrics.py          # Line-building, IoU matching, mAP computation
    ├── ocr_engines.py             # EasyOCR, Tesseract, Mistral OCR wrappers
    ├── ocr_metrics.py             # Text normalization, WER/CER scoring
    ├── ocr_output_norma.py        # Output normalization for all OCR engines
    ├── ocr_to_labelstudio.py      # OCR → Label Studio annotation format
    ├── pydantic_schema.py         # JSON output validation schema for VLMs
    ├── reading_order_metrics.py   # NDLD, BLEU, sequence comparison utilities
    ├── suryaOCR.sh                # Surya OCR CLI inference script
    └── v3/                        # LayoutReader/LayoutLMv3 submodule
        ├── helpers.py             # boxes2inputs, prepare_inputs, parse_logits
        ├── eval.py                # LayoutLMv3 evaluation utilities
        ├── train.py               # Training script
        ├── train.sh               # Training shell wrapper
        └── ds_config.json         # Dataset config for training`
```


The `utils/v3/` submodule is adapted from [FreeOCR-AI/layoutreader](https://github.com/FreeOCR-AI/layoutreader), originally developed by Pang Hantian, which ports LayoutReader into the HuggingFace Transformers environment.

---

### Notebooks

**`evaluation_pipeline.ipynb`** — Main evaluation workflow covering:

- OCR inference across all engines
- Ground truth and prediction normalization
- Layout analysis (mAP@IoU -> mAP@0.5, mAP@0.75)
- Reading order evaluation (NDLD, corpus BLEU) for OCR engines and LayoutReader combinations
- Text extraction evaluation (WER, CER)
- NER evaluation (LayoutLMv3, LiLT) and end-to-end DU (Donut)

**`vlms_evaluation.ipynb`** — VLM semantic evaluation covering:

- Section-by-section BERTScore comparison (Personal Info, Skills, Education, Employment Info, Employment Description)
- Per-resume and corpus-level aggregation
- Comparison across GPT-4.1-mini, Gemini-2.5-Pro, Gemini-2.5-Flash, Ministral-3:8B, LiquidAI, and RChilli

---

### Dependencies

Cloud API access is required for:

- **Mistral Basic OCR** — Mistral API key
- **AWS Textract** — AWS credentials and S3 bucket
- **GPT-4.1-mini** — OpenAI API key
- **Gemini-2.5-Pro/Flash** — Google AI API key
- **RChilli** — RChilli API credentials
- **Ministral-3:8B (2nd stage)** — RunPod instance or equivalent GPU
