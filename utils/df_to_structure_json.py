import pandas as pd
import json
from pathlib import Path

def export_gt_to_structured_jsons(df: pd.DataFrame, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    def row_to_json(row) -> dict:
        output = {"id": row["filename"], "section": {}}
        for section, label, bio, bbox, transcription, ro_number in zip(
            row["ro_section"], row["ro_label"], row["ro_bio"],
            row["ro_bboxes"], row["ro_transcription"], row["reading_order_number"],
        ):
            output["section"].setdefault(section, []).append({
                "item": label,
                "bio": bio,
                "reading_order_number": ro_number,
                "bbox": bbox,
                "transcription": transcription
            })
        return output

    for obj in df.apply(row_to_json, axis=1):
        out_path = out_dir / f"{obj['id']}.json"
        out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")