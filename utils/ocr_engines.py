import easyocr
import pytesseract
from PIL import Image
import json
import pandas as pd
import os, re
from mistralai import Mistral
import mimetypes
import base64
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseOCREngine(ABC):
    def __init__(self, input_dir: str, output_dir: str, languages: List[str] = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.languages = languages or self.default_languages
    
    @property
    @abstractmethod
    def default_languages(self) -> List[str]:
        pass
    
    @abstractmethod
    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        pass
    
    def process_images(self, extensions=('.png', '.jpg', '.jpeg')):
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(extensions):
                image_path = os.path.join(self.input_dir, filename)
                print(f"Processing: {filename}")
                
                try:
                    results = self._process_single_image(image_path)
                    output_data = {
                        "image": image_path,
                        "results": results
                    }
                    output_file = os.path.join(
                        self.output_dir, f"{os.path.splitext(filename)[0]}.json"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Saved: {output_file} ({len(results)} text elements)")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


class EasyOCREngine(BaseOCREngine):
    @property
    def default_languages(self) -> List[str]:
        return ['en', 'es']
    
    def __init__(self, input_dir: str, output_dir: str, languages: List[str] = None):
        super().__init__(input_dir, output_dir, languages)
        self.reader = easyocr.Reader(self.languages)
    
    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        results = self.reader.readtext(image_path)
        output = []
        for (bbox, text, confidence) in results:
            output.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": [[float(x), float(y)] for x, y in bbox]
            })
        return output


class TesseractEngine(BaseOCREngine):
    @property
    def default_languages(self) -> List[str]:
        return ['eng', 'spa']
    
    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        image = Image.open(image_path)
        lang_config = "+".join(self.languages)
        
        data = pytesseract.image_to_data(image, lang=lang_config, 
                                       output_type=pytesseract.Output.DICT)
        
        output = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                output.append({
                    "text": data['text'][i],
                    "confidence": int(data['conf'][i]),
                    "bbox": {
                        "left": int(data['left'][i]),
                        "top": int(data['top'][i]),
                        "width": int(data['width'][i]),
                        "height": int(data['height'][i])
                    },
                    "block_num": int(data['block_num'][i]),
                    "line_num": int(data['line_num'][i]),
                    "word_num": int(data['word_num'][i])
                })
        return output



class MistralOCREngine:
    def __init__(
        self,
        input_dir: str,
        api_key: str,
        model: str = "mistral-ocr-latest",
        valid_exts: tuple = (".png", ".jpg", ".jpeg")
    ):
        self.input_dir = input_dir
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.valid_exts = tuple(ext.lower() for ext in valid_exts)

    # --------------------- static helpers ---------------------

    @staticmethod
    def encode_image_to_b64(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def guess_mimetype(image_path: Path) -> str:
        mt, _ = mimetypes.guess_type(str(image_path))
        return mt or "image/png"

    @staticmethod
    def clean_markdown_hashes(line: str) -> str:
        return re.sub(r'^\s*#+\s*', '', line)

    @staticmethod
    def flatten_words(lines: List[str]) -> List[str]:
        words: List[str] = []
        for ln in lines:
            parts = [w for w in ln.split(" ") if w.strip()]
            words.extend(parts)
        return words

    # --------------------- OCR logic ---------------------

    def _mistral_ocr_markdown_lines(self, image_path: Path) -> List[str]:
        b64 = self.encode_image_to_b64(image_path)
        mime = self.guess_mimetype(image_path)

        resp = self.client.ocr.process(
            model=self.model,
            document={
                "type": "image_url",
                "image_url": f"data:{mime};base64,{b64}"
            },
            include_image_base64=False
        )

        lines: List[str] = []
        pages = getattr(resp, "pages", []) or []
        pages = sorted(pages, key=lambda p: getattr(p, "index", 0))

        for page in pages:
            md = getattr(page, "markdown", "") or ""
            page_lines = [
                self.clean_markdown_hashes(ln.strip())
                for ln in md.splitlines() if ln.strip()
            ]
            lines.extend(page_lines)

        return lines

    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        img_path = Path(image_path)
        text_lines = self._mistral_ocr_markdown_lines(img_path)
        words_flat = self.flatten_words(text_lines)
        return [{"text": text_lines, "words": words_flat}]

    # --------------------- public API ---------------------

    def process_images(self, extensions=('.png', '.jpg', '.jpeg')) -> pd.DataFrame:
        exts = tuple(ext.lower() for ext in (extensions or self.valid_exts))

        path = Path(self.input_dir).expanduser().resolve()
        assert path.exists() and path.is_dir(), f"Invalid directory: {path}"

        records: List[Dict[str, Any]] = []
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])

        for img_path in files:
            try:
                result_list = self._process_single_image(str(img_path))
                text_lines = result_list[0]["text"]
                words_flat = result_list[0]["words"]

                records.append({
                    "filename": img_path.stem,
                    "text": text_lines,
                    "words": words_flat
                })
            except Exception as e:
                records.append({
                    "filename": img_path.stem,
                    "text": [],
                    "words": [],
                    "error": str(e)
                })

        cols = ["filename", "text", "words"]
        if any("error" in r for r in records):
            cols.append("error")

        df = pd.DataFrame(records, columns=cols)
        return df