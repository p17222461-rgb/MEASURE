import warnings
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence '\\/'",
    category=SyntaxWarning,
)

import os
from pdf2image import convert_from_path
import ast
import unicodedata
import re
import ast, re, unicodedata, math
from typing import Any
import pandas as pd
import numpy as np
import os
import ast
from typing import Dict, Any
from utils.ocr_output_norma import BaseJsonNormalization

def pdf_to_images(input_dir, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            base_name = os.path.splitext(pdf_file)[0]
            
            for i, image in enumerate(images, 1):
                output_filename = f"{base_name}_page_{i:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                image.save(output_path, 'PNG')
                print(f"saved: {output_filename}")
                
            print(f"processed: {pdf_file} -> {len(images)} pages")
            
        except Exception as e:
            print(f"error processing: {pdf_file}: {e}")


# -------------------------------------------------------------------
def _is_null_scalar(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def _is_sequence_1d(x: Any) -> bool:
    if isinstance(x, (list, tuple)):
        return True
    if isinstance(x, (np.ndarray, pd.Series)):
        try:
            return getattr(x, "ndim", 1) == 1
        except Exception:
            return True
    return False

def _to_list_str(x: Any) -> list[str]:
    if isinstance(x, pd.Series):
        x = x.tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return ["" if it is None else str(it) for it in x]
    return ["" if x is None else str(x)]

class ImageDimFetcher(BaseJsonNormalization):
    def process_single_file(self, json_file_path: str) -> Dict[str, Any]:
        return {}

class OrganizeLSOutput:
    """
    this class is for structuring the LabelStudio annotation output CSV
    """
    def __init__(self, data, transcription_column='transcription', rename=None, images_directory=None):
        self.original_data = data.copy()
        self.data = data.copy()
        self.transcription_column = transcription_column
        self.rename = rename
        self.images_directory = images_directory
    
    def add_dimensions_column(self):
        if not self.images_directory:
            return self.data
        base_norm = ImageDimFetcher(images_directory=self.images_directory)
        dims_df = pd.DataFrame(
            self.data['filename'].apply(base_norm.get_image_dimensions).tolist(),
            index=self.data.index
        )
        self.data = pd.concat([self.data, dims_df], axis=1)
        return self.data

    def clean_labelstudio_text(self, text: Any) -> str:
        if _is_null_scalar(text):
            return ""

        if _is_sequence_1d(text):
            return " ".join(_to_list_str(text)).strip()

        text_str = str(text)
        text_str = text_str.replace('\\\\/', '/').replace('\\/', '/')

        if text_str.startswith('[') and text_str.endswith(']'):
            try:
                text_list = ast.literal_eval(text_str)
                return " ".join([str(item) for item in text_list])
            except Exception:
                text_str = text_str.replace('[', '').replace(']', '').replace('"', '')

        text_str = re.sub(r'^"|"$', '', text_str)
        return text_str

    def get_cleaned_transcription(self, text: Any) -> str:
        return self.clean_labelstudio_text(text)

    def split_into_words(self, text: Any) -> list[str]:
        if _is_null_scalar(text):
            return []
        cleaned_text = self.get_cleaned_transcription(text)
        return [w for w in cleaned_text.split() if w.strip()]

    def get_labelstudio_bbox(self, bbox_dict: dict) -> list[float]:
        return [bbox_dict['x'], bbox_dict['y'], bbox_dict['width'], bbox_dict['height']]

    def _parse_bbox_field(self, bbox_val: Any) -> list:
        if _is_null_scalar(bbox_val):
            return []

        if isinstance(bbox_val, str):
            try:
                bbox_val = ast.literal_eval(bbox_val)
            except Exception:
                return []

        if isinstance(bbox_val, (np.ndarray, pd.Series)):
            bbox_val = bbox_val.tolist()

        if isinstance(bbox_val, dict):
            bbox_val = [bbox_val]

        if not isinstance(bbox_val, (list, tuple)):
            return []

        return list(bbox_val)

    def process_word_bboxes(self, row: pd.Series) -> list[list[float]]:
        try:
            transcription_value = row.get(self.transcription_column, None)
            bbox_val = row.get('bbox', None)

            if _is_null_scalar(transcription_value) and not _is_sequence_1d(transcription_value):
                return []

            bbox_list = self._parse_bbox_field(bbox_val)
            if not bbox_list:
                return []

            labelstudio_bboxes = []
            for item in bbox_list:
                if isinstance(item, dict):
                    try:
                        ls_bbox = self.get_labelstudio_bbox(item)
                    except KeyError:
                        keys = {k.lower(): v for k, v in item.items()}
                        ls_bbox = [keys.get('x', 0), keys.get('y', 0),
                                   keys.get('width', keys.get('w', 0)),
                                   keys.get('height', keys.get('h', 0))]
                    labelstudio_bboxes.append([float(v) for v in ls_bbox])
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    labelstudio_bboxes.append([float(item[0]), float(item[1]),
                                               float(item[2]), float(item[3])])
                else:
                    continue
            return labelstudio_bboxes

        except Exception as e:
            print(f"error processing bbox: {e}")
            return []

    def extract_filename(self, ocr_path: Any) -> str:
        if _is_null_scalar(ocr_path):
            return ""
        ocr_str = str(ocr_path)
        if '=' in ocr_str:
            after_equals = ocr_str.split('=')[-1]
            filename = after_equals.split('/')[-1]
        else:
            filename = ocr_str.split('-')[-1]
        filename = re.sub(r'\.(png|jpg|jpeg)$', '', filename, flags=re.IGNORECASE)
        return filename

    def create_characters(self, transcription: Any) -> list[str]:
        if _is_null_scalar(transcription):
            return []

        text = self.get_cleaned_transcription(transcription)
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[\[\]"\",]', '', text)

        chars = []
        for ch in text:
            if not ch:
                continue
            if ch in ['\n', '\r', '\x0c', '\t']:
                continue
            if not ch.isprintable():
                continue
            if ch.isspace():
                if ch == ' ':
                    chars.append(ch)
                continue
            chars.append(ch)
        return chars

    def drop_duplicate_filenames(self):
        if "filename" in self.data.columns:
            self.data = self.data.drop_duplicates(subset=["filename"]).reset_index(drop=True)
        return self.data

    def clean_ro_label_column(self):
        if "ro_label" in self.data.columns:
            self.data["ro_label"] = self.data["ro_label"].apply(
                lambda x: [item["labels"][0] for item in x]
                if isinstance(x, list) else x
            )
        return self.data
    
    def add_words_column(self):
        self.data['words'] = self.data[self.transcription_column].apply(self.split_into_words)
        return self.data

    def add_bbox_column(self):
        self.data['bbox'] = self.data.apply(self.process_word_bboxes, axis=1)
        return self.data

    def add_filename_column(self):
        self.data['filename'] = self.data['ocr'].apply(self.extract_filename)
        return self.data

    def add_characters_column(self):
        self.data['characters'] = self.data[self.transcription_column].apply(self.create_characters)
        return self.data

    def process_all(self):
        self.add_words_column()
        self.add_bbox_column()
        self.add_filename_column()
        self.add_characters_column()
        self.add_dimensions_column()

        self.clean_ro_label_column()
        self.drop_duplicate_filenames()

        if self.rename:
            self.data = self.data.rename(columns=self.rename)

        return self.data

    # ---------- Getters ----------
    def get_processed_data(self):
        return self.data

    def get_original_data(self):
        return self.original_data

    def get_words(self):
        return self.data['words'] if 'words' in self.data.columns else None

    def get_word_bbox(self):
        return self.data['bbox'] if 'bbox' in self.data.columns else None

    def get_id_file(self):
        return self.data['filename'] if 'filename' in self.data.columns else None

    def get_characters(self):
        return self.data['characters'] if 'characters' in self.data.columns else None

# -------------------------------------------------------------------

class FormateLSOutputReadingOrder:
    def __init__(
        self,
        df,
        reading_order_col='reading_order',
        transcription_col='transcription',
        reading_order_number_col='reading_order_number',
        ro_bboxes_col='ro_bboxes',
        ro_transcription_col='ro_transcription',
        section_col='section',
        ro_section_col='ro_section',
        bio_col='bio',  # Add bio column
        ro_bio_col='ro_bio',  # Add reordered bio column
        label_col='label',  # Add label column  
        ro_label_col='ro_label',  # Add reordered label column
        filename_col='filename'   
    ):
        self.df = df
        self.reading_order_col = reading_order_col
        self.transcription_col = transcription_col
        self.reading_order_number_col = reading_order_number_col
        self.ro_bboxes_col = ro_bboxes_col
        self.ro_transcription_col = ro_transcription_col
        self.section_col = section_col
        self.ro_section_col = ro_section_col
        self.bio_col = bio_col  # Initialize bio column
        self.ro_bio_col = ro_bio_col  # Initialize reordered bio column
        self.label_col = label_col  # Initialize label column
        self.ro_label_col = ro_label_col  # Initialize reordered label column
        self.filename_col = filename_col  

    def process(self):
        # Parse reading_order column
        self.df[self.reading_order_col] = self.df[self.reading_order_col].apply(ast.literal_eval)
        self.df[self.reading_order_number_col] = [
            [item['number'] for item in lst]
            for lst in self.df[self.reading_order_col]
        ]
        self.df[self.ro_bboxes_col] = [
            [[item['x'], item['y'], item['width'], item['height']] for item in lst]
            for lst in self.df[self.reading_order_col]
        ]

        # Parse and clean transcription column
        self.df[self.transcription_col] = self.df[self.transcription_col].apply(ast.literal_eval)
        for i in range(len(self.df)):
            transcription_list = self.df.iloc[i][self.transcription_col]
            cleaned_list = []
            for item in transcription_list:
                if isinstance(item, str):
                    cleaned_list.append(item.replace('\\/', '/'))
                else:
                    cleaned_list.append(item)
            self.df.at[i, self.transcription_col] = cleaned_list

        # Parse section column
        self.df[self.section_col] = self.df[self.section_col].apply(ast.literal_eval)
        self.df[self.ro_section_col] = [
            self._extract_section_labels(section_list)
            for section_list in self.df[self.section_col]
        ]

        # Parse bio column if it exists
        if self.bio_col in self.df.columns:
            self.df[self.bio_col] = self.df[self.bio_col].apply(ast.literal_eval)
        else:
            print(f"Warning: Column '{self.bio_col}' not found in DataFrame")

        # Parse label column if it exists
        if self.label_col in self.df.columns:
            self.df[self.label_col] = self.df[self.label_col].apply(ast.literal_eval)
        else:
            print(f"Warning: Column '{self.label_col}' not found in DataFrame")

        # Reorder all columns
        self._reorder_all_columns()
        self._update_reading_order_numbers()

        return self.df

    def _extract_section_labels(self, section_list):
        section_labels = []
        for item in section_list:
            if isinstance(item, dict) and 'labels' in item:
                labels = item['labels']
                section_labels.append(labels[0] if labels else 'malo1')
            else:
                section_labels.append('malo2')
        return section_labels

    def _reorder_all_columns(self):
        def reorder_by_reading_order(row, data_list, list_name):
            reading_order_numbers = row[self.reading_order_number_col]
            file_name = None

            if self.filename_col in self.df.columns:
                file_name = row[self.filename_col]
            elif "ocr" in self.df.columns:
                file_name = os.path.basename(str(row["ocr"]))
            else:
                file_name = f"Index {row.name}"

            if len(data_list) != len(reading_order_numbers):
                print(
                    f"File: {os.path.basename(str(file_name))} → {list_name} has {len(data_list)} elements, "
                    f"reading_order has {len(reading_order_numbers)}"
                )
                min_len = min(len(data_list), len(reading_order_numbers))
                data_list = data_list[:min_len]
                reading_order_numbers = reading_order_numbers[:min_len]

            paired = list(zip(reading_order_numbers, data_list))
            paired_sorted = sorted(paired, key=lambda x: x[0])
            return [item[1] for item in paired_sorted]

        # Reorder transcription
        self.df[self.ro_transcription_col] = self.df.apply(
            lambda row: reorder_by_reading_order(row, row[self.transcription_col], "transcription"), axis=1
        )
        
        # Reorder bboxes
        self.df[self.ro_bboxes_col] = self.df.apply(
            lambda row: reorder_by_reading_order(row, row[self.ro_bboxes_col], "bboxes"), axis=1
        )
        
        # Reorder sections
        self.df[self.ro_section_col] = self.df.apply(
            lambda row: reorder_by_reading_order(row, row[self.ro_section_col], "sections"), axis=1
        )
        
        # Reorder bio if column exists
        if self.bio_col in self.df.columns:
            self.df[self.ro_bio_col] = self.df.apply(
                lambda row: reorder_by_reading_order(row, row[self.bio_col], "bio"), axis=1
            )
        
        # Reorder label if column exists
        if self.label_col in self.df.columns:
            self.df[self.ro_label_col] = self.df.apply(
                lambda row: reorder_by_reading_order(row, row[self.label_col], "label"), axis=1
            )

    def _update_reading_order_numbers(self):
        def create_sequential_order(row):
            length = len(row[self.ro_bboxes_col])
            return list(range(1, length + 1))
        self.df[self.reading_order_number_col] = self.df.apply(create_sequential_order, axis=1)