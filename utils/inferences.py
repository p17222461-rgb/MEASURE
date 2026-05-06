from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple
from PIL import Image
import torch
import pandas as pd
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForTokenClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import LayoutLMv3ForTokenClassification
from utils.v3.helpers import boxes2inputs, prepare_inputs, parse_logits

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _xywh_pct_to_px_xyxy(box_pct, W, H):
    x, y, w, h = [float(v) for v in box_pct]
    l = x * W / 100.0
    t = y * H / 100.0
    r = (x + w) * W / 100.0
    b = (y + h) * H / 100.0
    return [l, t, r, b]

def _norm_xyxy_llmv3(x1, y1, x2, y2, W, H):
    def n(v, dim):
        return int(max(0, min(1000, round(1000.0 * float(v) / float(dim)))))
    return [n(x1, W), n(y1, H), n(x2, W), n(y2, H)]

def coerce_box_to_llmv3(box, W, H):
    l, t, r, b = _xywh_pct_to_px_xyxy(box, W, H)
    return _norm_xyxy_llmv3(l, t, r, b, W, H)


class LayoutLmv3Inference:
    def __init__(
        self,
        df_ground_truth,
        images_base_path: str,
        model_name: str = "microsoft/layoutlmv3-base",
        sections_col: str = "ro_section",
        text_col: str = "ro_transcription",
        bbox_col: str = "ro_bboxes",
        apply_ocr: bool = False,
        device: Optional[str] = None,
        *,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        labels: List[str],
        processor: Any = None,
        model: Any = None,
    ):
        self.df = df_ground_truth
        self.images_base_path = str(images_base_path)
        self.sections_col = sections_col
        self.text_col = text_col
        self.bbox_col = bbox_col

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if id2label is None or label2id is None or labels is None:
            raise ValueError(
                "provide id2label, label2id and labels to build LayoutLmv3Inference."
            )
        self.id2label = id2label
        self.label2id = label2id
        self.labels = labels

        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=apply_ocr)
        else:
            self.processor = processor

        if model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def run(self) -> List[Dict[str, Any]]:
        results = []
        unique_filenames = self.df["filename"].unique()
        for filename in tqdm(unique_filenames, desc="evaluating documents"):
            try:
                pre = self._preprocess_for_inference(filename)

                with torch.no_grad():
                    enc = pre["encoding"]
                    inputs = {
                        "input_ids": enc["input_ids"].to(self.device),
                        "attention_mask": enc["attention_mask"].to(self.device),
                        "bbox": enc["bbox"].to(self.device),
                        "pixel_values": enc["pixel_values"].to(self.device)
                    }
                    outputs = self.model(**inputs)
                    pred_labels = self._decode_predictions_per_word(outputs.logits, enc, self.id2label)

                n = min(len(pre["words"]), len(pred_labels))
                pred_labels = pred_labels[:n]
                gt_texts = pre["words"][:n]
                gt_bboxes = pre["boxes_llm"][:n]
                gt_sections = pre["sections"][:n] if pre["sections"] else ["O"] * n

                results.append({
                    "filename": filename,
                    "gt_texts": gt_texts,
                    "gt_bboxes": gt_bboxes,
                    "gt_sections": gt_sections,
                    "pred_labels": pred_labels,
                })
            except Exception as e:
                print(f"Error en {filename}: {e}")
                continue
        return results

    def _preprocess_for_inference(self, filename: str) -> Dict[str, Any]:
        row = self.df.loc[self.df["filename"] == filename]
        if row.empty:
            raise ValueError(f"No se encontró filename='{filename}' en el DataFrame.")
        row = row.iloc[0]

        image_path = Path(self.images_base_path) / filename
        if not image_path.exists():
            alt = Path(str(image_path) + ".png")
            if alt.exists():
                image_path = alt
            else:
                raise FileNotFoundError(f"No existe la imagen: {image_path} (ni {alt})")

        image = Image.open(image_path).convert("RGB")

        W, H = int(row["width"]), int(row["height"])
        words_raw = list(row[self.text_col])
        bboxes_raw = list(row[self.bbox_col])
        sections = list(row[self.sections_col]) if self.sections_col in row else []

        boxes_llm = [coerce_box_to_llmv3(b, W, H) for b in bboxes_raw]

        encoding = self.processor(
            images=image,
            text=words_raw,
            boxes=boxes_llm,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        )

        return dict(
            encoding=encoding,
            image=image,
            words=words_raw,
            boxes_llm=boxes_llm,
            sections=sections,
            filename=filename,
            W=W, H=H
        )

    @staticmethod
    def _decode_predictions_per_word(logits: torch.Tensor, encoding, id2label: Dict[int, str]) -> List[str]:
        logits = logits.squeeze(0)
        word_ids = encoding.word_ids(0)

        buckets = {}
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            buckets.setdefault(w_id, []).append(tok_idx)

        pred_labels = []
        max_wid = max(buckets.keys()) if buckets else -1
        for w_id in range(max_wid + 1):
            tok_ids = buckets.get(w_id, [])
            if not tok_ids:
                pred_labels.append(id2label.get(0, "O"))
                continue
            sub_logits = logits[tok_ids, :]
            pooled = sub_logits.mean(dim=0)
            label_id = int(torch.argmax(pooled).item())
            pred_labels.append(id2label[label_id])
        return pred_labels


class DonutInference:

    def __init__(
        self,
        ckpt: str = "naver-clova-ix/donut-base",
        device: Optional[str] = None,
        max_length: int = 1536,
        num_beams: int = 1,
        patterns: tuple = (".png", ".jpg")):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.ckpt = ckpt
        self.max_length = max_length
        self.num_beams = num_beams
        self.patterns = patterns
        self.processor = DonutProcessor.from_pretrained(ckpt)
        self.model = VisionEncoderDecoderModel.from_pretrained(ckpt).to(self.device)

    def infer_image(self, image_path: str) -> str:
        image_path = str(image_path)
        if not os.path.exists(image_path):
            alt = image_path + ".png"
            if os.path.exists(alt):
                image_path = alt
            else:
                raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        prompt_token = self._find_task_prompt_token(self.processor.tokenizer)
        prompt = prompt_token if prompt_token is not None else ""

        task_prompt_ids = self.processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=task_prompt_ids,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return text

    def infer_directory(self, image_dir: str) -> pd.DataFrame:
        image_dir = Path(image_dir)
        paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in self.patterns])

        rows = []
        for p in tqdm(paths, desc="evaluating images with Donut"):
            try:
                raw = self.infer_image(p)
                rows.append({"filename": p.stem, "raw_output": [raw]})
            except Exception as e:
                rows.append({"filename": p.stem, "raw_output": [f"<ERROR: {e}>"]})

        df = pd.DataFrame(rows, columns=["filename", "raw_output"])
        df["words"] = df["raw_output"].apply(self._extract_words_from_raw)
        return df

    @staticmethod
    def _find_task_prompt_token(tokenizer):
        specials = tokenizer.additional_special_tokens or []
        candidates = [t for t in specials if t.startswith("<s_")]
        if candidates:
            return candidates[0]
        if "<s>" in specials:
            return "<s>"
        return None

    @staticmethod
    def _extract_words_from_raw(raw_output: List[str]) -> List[str]:
        if isinstance(raw_output, list):
            text = " ".join(str(t) for t in raw_output)
        else:
            text = str(raw_output)
        return [w for w in text.split() if w]
    

class LiLTInference:
    def __init__(
        self,
        df_ground_truth,
        images_base_path: str,
        model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base",
        sections_col: str = "ro_section",
        text_col: str = "ro_transcription",
        bbox_col: str = "ro_bboxes",
        device: Optional[str] = None,
        *,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        labels: Optional[List[str]] = None,
        processor: Any = None,
        model: Any = None,
    ):
        self.df = df_ground_truth
        self.images_base_path = str(images_base_path)
        self.sections_col = sections_col
        self.text_col = text_col
        self.bbox_col = bbox_col

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if id2label is None or label2id is None:
            raise ValueError("Debes proveer id2label y label2id para LiLTInference.")
        self.id2label = id2label
        self.label2id = label2id
        self.labels = labels 
        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = processor

        if model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def run(self) -> List[Dict[str, Any]]:
        results = []
        unique_filenames = self.df["filename"].unique()

        for filename in tqdm(unique_filenames, desc="LiLT · Evaluando documentos"):
            try:
                pre = self._preprocess_for_inference(filename)

                with torch.no_grad():
                    enc = pre["encoding"]
                    inputs = {
                        "input_ids": enc["input_ids"].to(self.device),
                        "attention_mask": enc["attention_mask"].to(self.device),
                        "bbox": enc["bbox"].to(self.device)
                    }
                    outputs = self.model(**inputs)
                    pred_labels = self._decode_predictions_per_word(outputs.logits, enc, self.id2label)

                n = min(len(pre["words"]), len(pred_labels))
                results.append({
                    "filename": pre["filename"],
                    "gt_texts": pre["words"][:n],
                    "gt_sections": (pre["sections"][:n] if pre["sections"] else ["O"] * n),
                    "pred_labels": pred_labels[:n],
                })

            except Exception as e:
                print(f"Error en {filename}: {e}")
                continue

        return results

    def _preprocess_for_inference(self, filename: str) -> Dict[str, Any]:
        row = self.df.loc[self.df["filename"] == filename]
        if row.empty:
            raise ValueError(f"No se encontró filename='{filename}' en el DataFrame.")
        row = row.iloc[0]

        image_path = Path(self.images_base_path) / filename
        if not image_path.exists():
            alt = Path(str(image_path) + ".png")
            if not alt.exists():
                raise FileNotFoundError(f"No existe la imagen: {image_path} (ni {alt})")
            image_path = alt

        _ = Image.open(image_path).convert("RGB")

        W, H = int(row["width"]), int(row["height"])
        words_raw = list(row[self.text_col])
        bboxes_raw = list(row[self.bbox_col])
        sections = list(row.get(self.sections_col, [])) if self.sections_col in row else []

        boxes_llm = [coerce_box_to_llmv3(b, W, H) for b in bboxes_raw]

        words_clean, boxes_clean, secs_clean = [], [], []
        for i, (t, box) in enumerate(zip(words_raw, boxes_llm)):
            if t is None:
                continue
            t_str = str(t).strip()
            if t_str == "":
                continue
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            words_clean.append(t_str)
            boxes_clean.append([int(v) for v in box])
            if sections:
                secs_clean.append(str(sections[i]).strip() if i < len(sections) else "O")

        if len(words_clean) == 0:
            raise ValueError(f"{filename}: sin pares (texto,bbox) válidos")
        if len(words_clean) != len(boxes_clean):
            raise ValueError(f"{filename}: longitudes distintas tras limpieza")

        encoding = self.processor(
            text=words_clean,
            boxes=boxes_clean,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False
        )

        return dict(
            encoding=encoding,
            words=words_clean,
            boxes_llm=boxes_clean,
            sections=secs_clean if sections else [],
            filename=filename,
            W=W, H=H
        )

    @staticmethod
    def _safe_word_ids(encoding, batch_index: int = 0):
        try:
            return encoding.word_ids(batch_index)
        except Exception:
            if hasattr(encoding, "encodings") and len(encoding.encodings) > batch_index:
                return encoding.encodings[batch_index].word_ids
        return None

    @staticmethod
    def _decode_predictions_per_word(logits: torch.Tensor, encoding, id2label: Dict[int, str]) -> List[str]:
        logits = logits.squeeze(0)
        word_ids = LiLTInference._safe_word_ids(encoding, 0)

        buckets = {}
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            buckets.setdefault(w_id, []).append(tok_idx)

        pred_labels = []
        max_wid = max(buckets.keys()) if buckets else -1
        for w_id in range(max_wid + 1):
            tok_ids = buckets.get(w_id, [])
            if not tok_ids:
                pred_labels.append(id2label.get(0, "O"))
                continue
            sub_logits = logits[tok_ids, :]
            pooled = sub_logits.mean(dim=0)
            label_id = int(torch.argmax(pooled).item())
            pred_labels.append(id2label[label_id])
        return pred_labels


class LayoutReaderInference:
    def __init__(self,
                 model_name: str = "hantian/layoutreader",
                 device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        self.model.eval().to(self.device)

    def run(self,
            input_dir: str,
            external_ocr_frames: Sequence[pd.DataFrame],
            text_column_pred: str = "text",
            filename_col: str = "filename",
            bbox_col: str = "bbox",
            on_missing: str = "skip") -> pd.DataFrame:
        input_dir = Path(input_dir)
        imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not imgs:
            raise FileNotFoundError(f"No images found in {input_dir.resolve()}")

        rows = []
        for img_path in tqdm(imgs, desc="LayoutReader: predicting reading order"):
            try:
                texts, boxes, size = self._fetch_lists_for_image(
                    img_path,
                    external_ocr_frames,
                    filename_col=filename_col,
                    bbox_col=bbox_col,
                    text_col=text_column_pred,
                )
                if not texts or not boxes:
                    msg = f"[WARN] No predictions row for {img_path.name} in provided dataframes."
                    if on_missing == "error":
                        raise ValueError(msg)
                    print(msg)
                    continue

                norm_boxes = self._norm_boxes_0_1000(boxes, size)
                orders = self._predict_reading_order(norm_boxes)

                rows.append({
                    "filename": img_path.stem,
                    "text": texts,
                    "bbox": boxes,
                    "predicted_order": orders,
                    "words_LR_order": self._words_in_lr_order(texts, orders),
                })
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")

        df = pd.DataFrame(
            rows,
            columns=["filename", "text", "bbox", "predicted_order", "words_LR_order"]
        ).sort_values(by=["filename"], ignore_index=True)
        return df

    @staticmethod
    def _clip_box_xyxy(box, W, H):
        l, t, r, b = box
        l = int(max(0, min(W, round(l))))
        t = int(max(0, min(H, round(t))))
        r = int(max(0, min(W, round(r))))
        b = int(max(0, min(H, round(b))))
        if r <= l:
            r = min(W, l + 1)
        if b <= t:
            b = min(H, t + 1)
        return [l, t, r, b]

    @classmethod
    def _clip_boxes_xyxy(cls, boxes: List[List[float]], W: int, H: int) -> List[List[int]]:
        return [cls._clip_box_xyxy(b, W, H) for b in boxes]

    @staticmethod
    def _norm_boxes_0_1000(pixel_boxes: List[List[int]], size: Tuple[int, int]) -> List[List[int]]:
        W, H = size
        sx = 1000.0 / max(1, W)
        sy = 1000.0 / max(1, H)
        out = []
        for l, t, r, b in pixel_boxes:
            ln = int(round(l * sx)); tn = int(round(t * sy))
            rn = int(round(r * sx)); bn = int(round(b * sy))
            ln = max(0, min(1000, ln)); tn = max(0, min(1000, tn))
            rn = max(0, min(1000, rn)); bn = max(0, min(1000, bn))
            if rn <= ln:
                rn = min(1000, ln + 1)
            if bn <= tn:
                bn = min(1000, tn + 1)
            out.append([ln, tn, rn, bn])
        return out

    @staticmethod
    def _reorder_text_by_order(text_list: List[str], order: List[int]) -> List[str]:
        return [text_list[j] for j in order]

    @classmethod
    def _words_in_lr_order(cls, text_list: List[str], order: List[int]) -> List[str]:
        ordered = cls._reorder_text_by_order(text_list, order)
        words = []
        for t in ordered:
            if t is None:
                continue
            parts = re.split(r"\s+", str(t).strip())
            words.extend([p for p in parts if p])
        return words

    @classmethod
    def _fetch_lists_for_image(cls,
                            image_path: Path,
                            external_frames: Sequence[pd.DataFrame],
                            filename_col: str,
                            bbox_col: str,
                            text_col: str) -> Tuple[List[str], List[List[int]], Tuple[int, int]]:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        
        # Use this instead of Path.stem to avoid "C.V-..." being misparse
        stem = image_path.name
        for ext in IMG_EXTS:
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
                break

        for df in (external_frames or []):
            if df is None or df.empty or (filename_col not in df.columns):
                continue

            # Strip known image extensions from df filenames too
            def safe_stem(x: str) -> str:
                for ext in IMG_EXTS:
                    if x.lower().endswith(ext):
                        return x[: -len(ext)]
                return x

            df_stems = df[filename_col].astype(str).apply(safe_stem)
            match_idx = df_stems[df_stems == stem].index

            if match_idx.empty:
                continue

            row = df.loc[match_idx[0]]
            texts = list(row[text_col])
            boxes = [list(map(float, b)) for b in row[bbox_col]]
            n = min(len(texts), len(boxes))
            texts = texts[:n]
            boxes = cls._clip_boxes_xyxy(boxes[:n], W, H)
            return texts, boxes, (W, H)

        return [], [], (W, H)

    def _predict_reading_order(self, norm_boxes: List[List[int]]) -> List[int]:
        inputs = boxes2inputs(norm_boxes)
        inputs = prepare_inputs(inputs, self.model)
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits.detach().cpu().squeeze(0)
        return parse_logits(logits, len(norm_boxes))