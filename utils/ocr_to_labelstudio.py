from PIL import Image
import glob
import os, re
import json
from typing import Dict, Any, List, Tuple, Iterator, Optional


class SuryaOCR:
    """
    this class converts Surya OCR output to LabelStudio format.
    """
    
    def __init__(self, labelstudio_base_path: str = "", LS_image_info: str = ""):
        self.labelstudio_base_path = labelstudio_base_path
        self.LS_image_info = LS_image_info
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.width, img.height

    def page_size_from_polygons(self, polygons: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
        if not polygons:
            return 1.0, 1.0
        max_x = max(p[0] for poly in polygons for p in poly)
        max_y = max(p[1] for poly in polygons for p in poly)
        return float(max_x), float(max_y)

    def polygon_to_ls_rect(self, polygon: List[Tuple[float, float]], 
                          base_w: float, base_h: float, ndigits: int = 2) -> Dict[str, float]:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)

        x_pct = 100.0 * xmin / base_w
        y_pct = 100.0 * ymin / base_h
        w_pct = 100.0 * w / base_w
        h_pct = 100.0 * h / base_h

        clip = lambda v: max(0.0, min(100.0, v))
        return {
            "x": round(clip(x_pct), ndigits),
            "y": round(clip(y_pct), ndigits),
            "width": round(clip(w_pct), ndigits),
            "height": round(clip(h_pct), ndigits),
            "rotation": 0
        }

    def _iter_documents(self, surya_data: Any) -> Iterator[Tuple[str, List[Dict]]]:
        if isinstance(surya_data, dict):
            if "text_lines" in surya_data and "page" in surya_data:
                yield "doc", [surya_data]
                return
            for k, v in surya_data.items():
                if isinstance(v, list):
                    yield k, v
                else:
                    if isinstance(v, dict) and "text_lines" in v:
                        yield k, [v]
        elif isinstance(surya_data, list):
            yield "doc", surya_data
        else:
            return

    def surya_page_to_labelstudio(self,
                                page: Dict[str, Any],
                                image_width: int,
                                image_height: int,
                                image_path: str,
                                id_prefix: str = "bb") -> List[Dict[str, Any]]:
        text_lines = page.get("text_lines", []) or []
        page_polys = [tl.get("polygon", []) for tl in text_lines if tl.get("polygon")]
        poly_w, poly_h = self.page_size_from_polygons(page_polys)

        results = []
        total_conf, n_conf = 0.0, 0

        for tl_idx, line_data in enumerate(text_lines):
            text = line_data.get("text", "")
            polygon = line_data.get("polygon", [])
            conf = float(line_data.get("confidence", 0.0))

            if not (text and polygon and len(polygon) >= 4):
                continue

            rect = self.polygon_to_ls_rect(polygon, poly_w, poly_h, ndigits=2)
            ann_id = f"{id_prefix}_{tl_idx}"

            bbox_result = {
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": rect,
                "id": ann_id,
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectangle"
            }
            label_result = {
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {**rect, "labels": ["Text"]},
                "id": ann_id,
                "from_name": "label",
                "to_name": "image",
                "type": "labels"
            }
            transcription_result = {
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {**rect, "text": [text]},
                "id": ann_id,
                "from_name": "transcription",
                "to_name": "image",
                "type": "textarea",
                "score": conf
            }

            results.extend([bbox_result, label_result, transcription_result])
            total_conf += conf
            n_conf += 1

        avg_score = (total_conf / n_conf) if n_conf else 0.0

        image_filename = os.path.basename(image_path)
        full_image_path = f"{self.labelstudio_base_path}{image_filename}" if self.labelstudio_base_path else image_path

        if self.LS_image_info:
            image_filename = os.path.basename(image_path)
            full_image_path = f"{self.LS_image_info.rstrip('/')}/{image_filename}"
        else:
            full_image_path = image_path

        return [{
            "data": {"ocr": full_image_path},
            "predictions": [{
            "model_version": "surya_ocr_model",
            "result": results,
            "score": round(avg_score, 4)
        }]
    }]

    def _find_image_for_page(self, image_dir: str, base_name: str, page_num: int) -> str:
        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"):
            candidate = os.path.join(image_dir, f"{base_name}_page_{page_num:03d}{ext}")
            if os.path.exists(candidate):
                return candidate

        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"):
            candidate = os.path.join(image_dir, f"{base_name}{ext}")
            if os.path.exists(candidate):
                return candidate

        pattern = re.compile(re.escape(base_name), re.IGNORECASE)
        for path in glob.glob(os.path.join(image_dir, "*")):
            if os.path.isfile(path) and pattern.search(os.path.basename(path)):
                return path

        return ""
    def process_directory(self, json_dir: str, image_dir: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        json_files = glob.glob(os.path.join(json_dir, "*.json"))

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                surya_data = json.load(f)

            base_name = os.path.splitext(os.path.basename(json_file))[0]

            for doc_key, pages in self._iter_documents(surya_data):
                for i, page in enumerate(pages):
                    page_num = int(page.get("page", i + 1))

                    image_path = self._find_image_for_page(image_dir, base_name, page_num)
                    if not image_path:
                        print(f"[WARN] No image for {base_name} page {page_num} in {image_dir}")
                        image_width, image_height = 0, 0
                    else:
                        image_width, image_height = self.get_image_dimensions(image_path)

                    id_prefix = f"bb_{base_name}_p{page_num}"
                    ls_page = self.surya_page_to_labelstudio(
                        page=page,
                        image_width=image_width,
                        image_height=image_height,
                        image_path=image_path if image_path else f"{base_name}_page_{page_num:03d}",
                        id_prefix=id_prefix
                    )

                    if re.search(r"_page_\d{3}$", base_name):
                        clean_base = base_name
                    else:
                        clean_base = f"{base_name}_page_{page_num:03d}"

                    out_path = os.path.join(output_dir, f"{clean_base}_labelstudio.json")
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        json.dump(ls_page, out_f, ensure_ascii=False, indent=2)

                    print(f"[OK] {base_name} page {page_num} -> {out_path}")

    def process_single_file(self, json_file: str, image_dir: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(json_file, "r", encoding="utf-8") as f:
            surya_data = json.load(f)

        base_name = os.path.splitext(os.path.basename(json_file))[0]

        for doc_key, pages in self._iter_documents(surya_data):
            for i, page in enumerate(pages):
                page_num = int(page.get("page", i + 1))

                image_path = self._find_image_for_page(image_dir, base_name, page_num)
                if not image_path:
                    print(f"[WARN] No image for {base_name} page {page_num} in {image_dir}")
                    image_width, image_height = 0, 0
                else:
                    image_width, image_height = self.get_image_dimensions(image_path)

                id_prefix = f"bb_{base_name}_p{page_num}"
                ls_page = self.surya_page_to_labelstudio(
                    page=page,
                    image_width=image_width,
                    image_height=image_height,
                    image_path=image_path if image_path else f"{base_name}_page_{page_num:03d}",
                    id_prefix=id_prefix
                )

                if re.search(r"_page_\d{3}$", base_name):
                    clean_base = base_name
                else:
                    clean_base = f"{base_name}_page_{page_num:03d}"

                out_path = os.path.join(output_dir, f"{clean_base}_labelstudio.json")

                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(ls_page, out_f, ensure_ascii=False, indent=2)

                print(f"[OK] {base_name} page {page_num} -> {out_path}")