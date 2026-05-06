import json
import pandas as pd
from typing import List, Dict, Any
import os 
import re
import glob
from abc import ABC, abstractmethod
from PIL import Image
from lxml import html
from pathlib import Path
            

class BaseJsonNormalization(ABC):
    def __init__(self, images_directory: str = None):
        self.images_directory = images_directory
    
    def process_directory(self, json_directory: str, pattern: str = "*.json") -> pd.DataFrame:
        json_files = glob.glob(os.path.join(json_directory, pattern))
        data_rows = []
        
        for json_file_path in json_files:
            try:
                #row_data = self.process_single_file(json_file_path)
                row_data = self.process_single_file_with_dimensions(json_file_path)
                if row_data:
                    data_rows.append(row_data)
            except Exception as e:
                print(f"Error processing file {json_file_path}: {e}")
                continue
        
        return pd.DataFrame(data_rows) if data_rows else pd.DataFrame()
    
    @abstractmethod
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        pass
    
    def extract_filename(self, data: Dict[str, Any], json_file_path: str) -> str:
        image_path = data.get("image", "")
        filename = os.path.basename(image_path) if image_path else os.path.basename(json_file_path)
        return os.path.splitext(filename)[0]
    
    def get_image_dimensions(self, filename: str) -> Dict[str, int]:
        if not self.images_directory:
            return {"width": 0, "height": 0}
            
        image_extensions = ['.png', '.jpg']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join(self.images_directory, f"{filename}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            for file in os.listdir(self.images_directory):
                if os.path.splitext(file)[0] == filename:
                    image_path = os.path.join(self.images_directory, file)
                    break
        
        if image_path:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    return {"width": width, "height": height}
            except Exception as e:
                print(f"Error al obtener dimensiones de {image_path}: {e}")
        
        return {"width": 0, "height": 0}

    def process_single_file_with_dimensions(self, json_file_path: str) -> Dict[str, Any]:
        result = self.process_single_file(json_file_path)
        if result and 'filename' in result:
            dimensions = self.get_image_dimensions(result['filename'])
            result.update(dimensions)
        return result

    def clean_b_tags(self, text: str) -> str:
        """
        Remove <b> and </b> tags from text while preserving the content inside them.
        """
        if not isinstance(text, str):
            return text
            
        # Remove both <b> and </b> tags
        cleaned_text = text.replace('<b>', '').replace('</b>', '')
        return cleaned_text

class JsonSuryaNormalization(BaseJsonNormalization):
    """
    this class takes the Surya OCR model and structures it in a dataframe
    """
    def __init__(self, images_directory: str = None):
        super().__init__(images_directory)
    
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        with open(json_file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and data:
            _, pages = next(iter(data.items()))
            filename = os.path.splitext(os.path.basename(json_file_path))[0]  # use disk filename as source of truth
        elif isinstance(data, list):
            filename = self.extract_filename({}, json_file_path)
            pages = data
        else:
            filename = self.extract_filename({}, json_file_path)
            pages = []

        def is_valid_bbox(b) -> bool:
            return isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b)

        def is_valid_char(ch: Dict[str, Any]) -> bool:
            if not isinstance(ch, dict):
                return False
            t = ch.get("text")
            if t is None or t == "":
                return False
            if not ch.get("bbox_valid", True):
                return False
            return is_valid_bbox(ch.get("bbox"))

        def merge_bboxes(bboxes: List[List[float]]):
            xs1 = [b[0] for b in bboxes]
            ys1 = [b[1] for b in bboxes]
            xs2 = [b[2] for b in bboxes]
            ys2 = [b[3] for b in bboxes]
            return [min(xs1), min(ys1), max(xs2), max(ys2)]

        text_lines: List[str] = []
        text_lines_bboxes: List[List[float]] = []
        words: List[str] = []
        word_bboxes: List[List[float]] = []
        chars: List[str] = []
        char_bboxes: List[List[float]] = []
        confidence = []

        for page in pages or []:
            if not isinstance(page, dict):
                continue
            # text-in-line
            for line in page.get("text_lines", []) or []:
                line_text = line.get("text", "")
                # Clean <b> tags from line text
                line_text = self.clean_b_tags(line_text)
                line_bbox = line.get("bbox")
                confidences = line.get("confidence")
                if not line_text or not is_valid_bbox(line_bbox):
                    continue
                text_lines.append(line_text)
                text_lines_bboxes.append(line_bbox)
                confidence.append(confidences)

                # characters
                for ch in line.get("chars", []) or []:
                    t = ch.get("text", "")
                    # Clean <b> tags from character text
                    t = self.clean_b_tags(t)
                    if t and is_valid_char(ch):
                        chars.append(t)
                        char_bboxes.append(ch["bbox"])

                # words (prefer provided, else derive from chars)
                provided_words = line.get("words", []) or []
                if provided_words:
                    for w in provided_words:
                        if isinstance(w, dict) and w.get("text") and is_valid_bbox(w.get("bbox")):
                            word_text = w.get("text", "")
                            # Clean <b> tags from word text
                            word_text = self.clean_b_tags(word_text)
                            if word_text:  # Only add if text remains after cleaning
                                words.append(word_text)
                                word_bboxes.append(w["bbox"])
                else:
                    current_word_chars: List[str] = []
                    current_word_bboxes: List[List[float]] = []

                    for ch in line.get("chars", []) or []:
                        t = ch.get("text", "")
                        # Clean <b> tags from character text
                        t = self.clean_b_tags(t)
                        if t and not t.isspace():
                            current_word_chars.append(t)
                            if ch.get("bbox_valid", True) and is_valid_bbox(ch.get("bbox")):
                                current_word_bboxes.append(ch["bbox"])
                        else:
                            if current_word_chars:
                                word_text = "".join(current_word_chars)
                                if word_text:  # Only add if word is not empty
                                    words.append(word_text)
                                    if current_word_bboxes:
                                        word_bboxes.append(merge_bboxes(current_word_bboxes))
                                    else:
                                        lb = line.get("bbox")
                                        word_bboxes.append(lb if is_valid_bbox(lb) else [0, 0, 0, 0])
                                current_word_chars, current_word_bboxes = [], []

                    if current_word_chars:
                        word_text = "".join(current_word_chars)
                        if word_text:  # Only add if word is not empty
                            words.append(word_text)
                            if current_word_bboxes:
                                word_bboxes.append(merge_bboxes(current_word_bboxes))
                            else:
                                lb = line.get("bbox")
                                word_bboxes.append(lb if is_valid_bbox(lb) else [0, 0, 0, 0])

        dimensions = self.get_image_dimensions(filename)
        
        return {
            "filename": filename,
            "text_line": text_lines,
            "bbox": text_lines_bboxes,
            "confidence": confidence,
            "words": words,
            "characters": chars,
            "width": dimensions["width"],
            "height": dimensions["height"]
        }
    
    def process_directory(self, json_directory: str, pattern: str = "*.json") -> pd.DataFrame:
        """
        process a complete directory of JSON output files from Surya OCR
        """
        json_files = glob.glob(os.path.join(json_directory, pattern))
        data_rows = []
        
        for json_file_path in json_files:
            try:
                row_data = self.process_single_file(json_file_path)
                if row_data:
                    data_rows.append(row_data)
            except Exception as e:
                print(f"Error processing file {json_file_path}: {e}")
                continue
        
        return pd.DataFrame(data_rows) if data_rows else pd.DataFrame()


class JsonEasyOCRNormalization(BaseJsonNormalization):
    """
    this class processes and normalizes EasyOCR JSON output files
    """
    
    @staticmethod
    def convert_bbox_8_to_4(bbox_8_points):
        if len(bbox_8_points) != 4:
            return [0, 0, 0, 0]
        
        x_coords = [point[0] for point in bbox_8_points]
        y_coords = [point[1] for point in bbox_8_points]
        
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        #filename = self.extract_filename(data, json_file_path)
        filename = os.path.splitext(os.path.basename(json_file_path))[0]
        
        text_elements = []
        bbox_elements_4pt = []  
        word_elements = []
        all_characters = []
        confidence = []
        
        for result in data.get("results", []):
            text = result.get("text", "").strip()
            bbox_8pt = result.get("bbox", [])
            confidences_scores = result.get("confidence", [])
            
            if text and len(bbox_8pt) == 4:  
                text_elements.append(text)
                
                bbox_4pt = self.convert_bbox_8_to_4(bbox_8pt)
                bbox_elements_4pt.append(bbox_4pt)
                                        
                words = text.split()
                word_elements.extend(words)

                characters = list(text)
                all_characters.extend(characters)

                confidence.append(confidences_scores)
        dimensions = self.get_image_dimensions(filename)
        
        return {
            "filename": filename,
            "text": text_elements,
            "bbox": bbox_elements_4pt,
            "confidence": confidence,
            "words": word_elements,
            "characters": all_characters,
            "width": dimensions["width"],
            "height": dimensions["height"]
        }


class JsonTesseractNormalization(BaseJsonNormalization):
    """
    this class processes and normalizes Tesseract OCR JSON output files
    """
    
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        #filename = self.extract_filename(data, json_file_path)
        filename = os.path.splitext(os.path.basename(json_file_path))[0]
        results = data.get("results", [])
        
        if not results:
            return None
        
        words = []
        bboxes = []
        block_nums = []
        line_nums = []
        word_nums = []
        all_characters = []
        confidence = []
        
        for result in results:
            word_text = result.get("text", "")
            words.append(word_text)
            
            characters = list(word_text)
            all_characters.extend(characters)
            
            bbox_data = result.get("bbox", {})
            bboxes.append([
                bbox_data.get("left", 0),
                bbox_data.get("top", 0),
                bbox_data.get("width", 0) + bbox_data.get("left", 0),  
                bbox_data.get("height", 0) + bbox_data.get("top", 0)   
            ])
            
            block_nums.append(result.get("block_num", 0))
            line_nums.append(result.get("line_num", 0))
            word_nums.append(result.get("word_num", 0))
            confidence.append(result.get("confidence", 0))
        dimensions = self.get_image_dimensions(filename)
        
        return {
            "filename": filename,
            "words": words,
            "bbox": bboxes,
            "confidence": confidence,
            "block_nums": block_nums,
            "line_nums": line_nums,
            "word_nums": word_nums,
            "characters": all_characters,
            "width": dimensions["width"],
            "height": dimensions["height"]
        }

class JsonDotsOCRNormalization(BaseJsonNormalization):
    """This class processes and normalizes DotsOCR JSON output files"""
    
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        filename = os.path.basename(json_file_path)
        filename = os.path.splitext(filename)[0]
        
        bbox_list = []
        text_list = []
        category_list = []
        words_list = []
        all_characters = []
        
        for element in data:
            bbox = element.get('bbox', [])
            text = element.get('text', '')
            category = element.get('category', '')
            
            if text.strip().startswith('<table>'):
                processed_text = self.extract_text_from_html_tables([text])
                clean_text_for_words = ' '.join(processed_text)
                words = self.extract_words(clean_text_for_words)
            else:
                words = self.extract_words(text)
            
            characters = list(text)
            
            all_characters.extend(characters)
            bbox_list.append(bbox)
            text_list.append(text)
            category_list.append(category)
            words_list.extend(words)
        
        dimensions = self.get_image_dimensions(filename)
        
        return {
            'filename': filename,
            'bbox': bbox_list,
            'text': text_list, 
            'category': category_list,
            'words': words_list, 
            'characters': all_characters,
            'width': dimensions["width"],
            'height': dimensions["height"]
        }
    
    def extract_text_from_html_tables(self, html_strings):
        """Extract clean text from HTML table content"""
        all_text = []
        for html_content in html_strings:
            tree = html.fromstring(html_content)
            text_elements = tree.xpath('//text()')
            clean_text = [
                text.strip() for text in text_elements 
                if text.strip() and (len(text.strip()) > 1 or text.strip().isdigit())
            ]
            
            all_text.extend(clean_text)
        return all_text
    
    def extract_words(self, text: str) -> List[str]:
        if not text:
            return []
        
        cleaned_text = re.sub(r'^#+\s*', '', text).strip()
        raw_tokens = cleaned_text.split()
        words = []
        for token in raw_tokens:
            token = re.sub(r'^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$', '', token)
            if token:
                words.append(token)
        
        return words

class JsonAWSTextractNormalization(BaseJsonNormalization):
    """This class processes and normalizes AWS Textract JSON output files"""

    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        json_path = Path(json_file_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        line_texts: List[str] = []
        word_texts: List[str] = []
        line_bboxes: List[Dict[str, Any]] = []
        word_bboxes: List[Dict[str, Any]] = []
        line_confidence: List[float] = []

        for block in data.get("Blocks", []):
            if "Text" not in block:
                continue

            text = block["Text"]
            block_type = block.get("BlockType")
            confidence_value = block.get("Confidence", 0.0)

            #   LINE blocks
            if block_type == "LINE":
                line_texts.append(text)

                bbox = (
                    block.get("Geometry", {})
                         .get("BoundingBox", None)
                )
                line_bboxes.append(bbox)
                line_confidence.append(confidence_value)

            #   WORD blocks
            elif block_type == "WORD":
                word_texts.append(text)

                bbox = (
                    block.get("Geometry", {})
                         .get("BoundingBox", None)
                )
                word_bboxes.append(bbox)
        filename = json_path.stem

        return {
            "filename": filename,
            "text": line_texts,
            "words": word_texts,
            "bbox": line_bboxes,
            "confidence": line_confidence,
            "word_bbox": word_bboxes,
            
        }

# class AWS_Textract_Normalization(BaseJsonNormalization):
#     def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
#         json_path = Path(json_file_path)

#         with open(json_path, "r") as f:
#             data = json.load(f)

#         # Get page dimensions from PAGE block
#         page_w, page_h = 1.0, 1.0  # fallback — will stay normalized if no page block
#         for block in data.get("Blocks", []):
#             if block.get("BlockType") == "PAGE":
#                 bb = block.get("Geometry", {}).get("BoundingBox", {})
#                 # Textract bboxes are fractional, we need actual pixel size
#                 # Use image dimensions if available, otherwise keep as 1.0
#                 break

#         line_texts: List[str] = []
#         word_texts: List[str] = []
#         line_bboxes: List[List[float]] = []
#         word_bboxes: List[List[float]] = []
#         line_confidence: List[float] = []

#         for block in data.get("Blocks", []):
#             if "Text" not in block:
#                 continue

#             text = block["Text"]
#             block_type = block.get("BlockType")
#             confidence_value = block.get("Confidence", 0.0)
#             bb = block.get("Geometry", {}).get("BoundingBox", {})

#             # Convert Textract fractional bbox dict → [l, t, r, b] in pixels
#             def bbox_to_xyxy(bb, W, H):
#                 l = bb.get("Left", 0.0) * W
#                 t = bb.get("Top", 0.0) * H
#                 r = (bb.get("Left", 0.0) + bb.get("Width", 0.0)) * W
#                 b = (bb.get("Top", 0.0) + bb.get("Height", 0.0)) * H
#                 return [l, t, r, b]

#             if block_type == "LINE":
#                 line_texts.append(text)
#                 line_bboxes.append(bbox_to_xyxy(bb, page_w, page_h))
#                 line_confidence.append(confidence_value)

#             elif block_type == "WORD":
#                 word_texts.append(text)
#                 word_bboxes.append(bbox_to_xyxy(bb, page_w, page_h))

#         filename = json_path.stem

#         return {
#             "filename": filename,
#             "text": line_texts,
#             "words": word_texts,
#             "bbox": line_bboxes,
#             "confidence": line_confidence,
#             "word_bbox": word_bboxes,
#         }