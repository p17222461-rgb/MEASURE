import numpy as np
import pandas as pd

class BuildLines:
    @staticmethod
    def gt_percent_to_px_xywh(gt_percent_xywh, W, H):
        x, y, w, h = gt_percent_xywh
        return np.array([x/100*W, y/100*H, w/100*W, h/100*H], dtype=float)
    
    @staticmethod
    def xywh_to_xyxy(b):
        x, y, w, h = b
        return np.array([x, y, x+w, y+h], dtype=float)
    
    @staticmethod
    def _xyxy_to_xywh_row(bboxes_xyxy):
        out = []
        for x1, y1, x2, y2 in bboxes_xyxy:
            out.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        return out
    
    @staticmethod
    def merge_fragments_into_lines(ro_bboxes, ro_texts, y_tol=0.6, drop_bars=True):
        boxes = np.array(ro_bboxes, dtype=float)   
        texts = [str(t) for t in ro_texts if (not drop_bars) or (str(t).strip() != "|")]

        n = min(len(boxes), len(texts))
        if n == 0:
            return [], []
        boxes, texts = boxes[:n], texts[:n]

        y_centers = boxes[:,1] + boxes[:,3]/2.0
        order = np.argsort(y_centers)
        boxes = boxes[order]
        y_centers = y_centers[order]
        texts = list(np.array(texts, dtype=object)[order])

        lines = []
        for i in range(len(texts)):
            b = boxes[i]; yc = y_centers[i]; h = b[3]
            if lines:
                last = lines[-1]
                cond = abs(yc - last['y_mean']) <= max(h, last['h_mean']) * y_tol
                if cond:
                    x1 = min(last['bbox'][0], b[0]); y1 = min(last['bbox'][1], b[1])
                    x2 = max(last['bbox'][0]+last['bbox'][2], b[0]+b[2])
                    y2 = max(last['bbox'][1]+last['bbox'][3], b[1]+b[3])
                    last['bbox'] = np.array([x1, y1, x2-x1, y2-y1], float)
                    last['idxs'].append(i)
                    ys = [boxes[j,1] + boxes[j,3]/2.0 for j in last['idxs']]
                    hs = [boxes[j,3] for j in last['idxs']]
                    last['y_mean'], last['h_mean'] = float(np.mean(ys)), float(np.mean(hs))
                    continue
            lines.append({'idxs':[i], 'bbox': b.copy(), 'y_mean': float(yc), 'h_mean': float(h)})

        line_texts, line_bboxes = [], []
        for ln in lines:
            idxs = sorted(ln['idxs'], key=lambda j: boxes[j,0])  # izquierda→derecha
            line_texts.append(" ".join(texts[j].strip() for j in idxs if texts[j] is not None))
            line_bboxes.append(ln['bbox'].tolist())
        return line_texts, line_bboxes
    
    @classmethod
    def build_line_columns(cls, df, text_col, bbox_col, input_mode="xywh", y_tol=0.6):
        out = df.copy()
        if input_mode.lower() == "xyxy":
            out[bbox_col] = out[bbox_col].apply(cls._xyxy_to_xywh_row)
        res = out[[text_col, bbox_col]].apply(
            lambda r: cls.merge_fragments_into_lines(r[bbox_col], r[text_col], y_tol=y_tol),
            axis=1
        )
        out['text_in_line'] = res.apply(lambda x: x[0])
        out['bbox_in_line'] = res.apply(lambda x: x[1]) 
        return out


class ComputeMAP: 
    @staticmethod
    def gt_percent_to_px_xywh_list(gt_pct_list, W, H):
        gt_pct = np.array(gt_pct_list, dtype=float)       
        if gt_pct.ndim == 1: 
            gt_pct = gt_pct[None, :]
        gt_px = gt_pct.copy()
        gt_px[:, 0] = gt_pct[:, 0] * W / 100.0
        gt_px[:, 1] = gt_pct[:, 1] * H / 100.0
        gt_px[:, 2] = gt_pct[:, 2] * W / 100.0
        gt_px[:, 3] = gt_pct[:, 3] * H / 100.0
        return gt_px 
    
    @staticmethod
    def xywh_to_xyxy(arr):
        arr = np.array(arr, dtype=float)
        if arr.ndim == 1: 
            arr = arr[None, :]
        out = arr.copy()
        out[:, 2] = arr[:, 0] + arr[:, 2]
        out[:, 3] = arr[:, 1] + arr[:, 3]
        return out
    
    @staticmethod
    def iou_matrix_xywh(gt_xywh, pr_xywh):
        if len(gt_xywh) == 0 or len(pr_xywh) == 0:
            return np.zeros((len(gt_xywh), len(pr_xywh)), dtype=float)
        A = ComputeMAP.xywh_to_xyxy(gt_xywh)
        B = ComputeMAP.xywh_to_xyxy(pr_xywh)
        ious = np.zeros((A.shape[0], B.shape[0]), dtype=float)
        for i in range(A.shape[0]):
            ax1, ay1, ax2, ay2 = A[i]
            aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
            for j in range(B.shape[0]):
                bx1, by1, bx2, by2 = B[j]
                bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)
                inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = inter_w * inter_h
                union = aa + bb - inter
                ious[i, j] = (inter / union) if union > 0 else 0.0
        return ious
    
    @staticmethod
    def greedy_match(ious, thr):
        pairs = []
        if ious.size == 0:
            return pairs
        flat = [(ious[i,j], i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
        flat.sort(reverse=True, key=lambda x: x[0])
        used_gt = set()
        used_pr = set()
        for v, i, j in flat:
            if v < thr: 
                break
            if i in used_gt or j in used_pr: 
                continue
            pairs.append((i, j))
            used_gt.add(i)
            used_pr.add(j)
        return pairs
    
    @staticmethod
    def calculate_map_with_scores(gt_boxes, pred_boxes_with_scores, iou_threshold=0.5):
        if len(gt_boxes) == 0 or len(pred_boxes_with_scores) == 0:
            return 0.0
        
        pred_boxes = [box[:4] for box in pred_boxes_with_scores]
        pred_scores = [box[4] for box in pred_boxes_with_scores]
        
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes_sorted = [pred_boxes[i] for i in sorted_indices]
        pred_scores_sorted = [pred_scores[i] for i in sorted_indices]
        
        ious = ComputeMAP.iou_matrix_xywh(gt_boxes, pred_boxes_sorted)
        
        gt_matched = [False] * len(gt_boxes)
        tp = np.zeros(len(pred_boxes_sorted))
        fp = np.zeros(len(pred_boxes_sorted))
        
        for j in range(len(pred_boxes_sorted)):
            best_iou = 0.0
            best_gt_idx = -1
            
            for i in range(len(gt_boxes)):
                if not gt_matched[i] and ious[i, j] >= iou_threshold and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt_idx = i
            
            if best_gt_idx != -1:
                tp[j] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[j] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calcular AP usando método Pascal VOC 11-point
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])
        ap /= 11
        
        return ap
    
    @staticmethod
    def _is_empty_boxes(v):
        if v is None:
            return True
        if isinstance(v, float) and np.isnan(v):
            return True
        if isinstance(v, (list, tuple)):
            return len(v) == 0
        if isinstance(v, np.ndarray):
            return v.size == 0
        return False
    
    @classmethod
    def prepare_aws_textract_df(cls, predictions_data_aws_textract):
        df = predictions_data_aws_textract.copy()

        def convert_bbox(bbox, W, H):
            if bbox is None:
                return None

            # Case 1: AWS Textract raw dict
            if isinstance(bbox, dict):
                return [
                    bbox.get("Left", 0.0) * W,
                    bbox.get("Top", 0.0) * H,
                    bbox.get("Width", 0.0) * W,
                    bbox.get("Height", 0.0) * H
                ]

            # Case 2: already a list/tuple/array: [x, y, w, h]
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
                x, y, w, h = map(float, bbox)

                # If values look normalized: 0–1
                if max(x, y, w, h) <= 1.0:
                    return [x * W, y * H, w * W, h * H]

                # If values look like percentages: 0–100
                if max(x, y, w, h) <= 100.0:
                    return [x * W / 100.0, y * H / 100.0, w * W / 100.0, h * H / 100.0]

                # Otherwise assume already pixels
                return [x, y, w, h]

            return None

        converted_boxes = []

        for _, row in df.iterrows():
            W = float(row["width"])
            H = float(row["height"])

            boxes = [convert_bbox(b, W, H) for b in row["bbox"]]
            converted_boxes.append(boxes)

        df["bbox"] = converted_boxes

        df["filtered"] = df.apply(
            lambda row: [
                (b, t, c)
                for b, t, c in zip(row["bbox"], row["text"], row["confidence"])
                if b is not None
            ],
            axis=1
        )

        df["bbox"] = df["filtered"].apply(lambda x: [item[0] for item in x])
        df["text"] = df["filtered"].apply(lambda x: [item[1] for item in x])
        df["confidence"] = df["filtered"].apply(lambda x: [item[2] for item in x])

        df.drop("filtered", axis=1, inplace=True)

        return df
    
    @classmethod
    def metrics_for_thresholds(cls, gt_pct_lines, pred_px_lines_with_scores, W, H, thresholds=(0.5, 0.75)):
        gt_px = cls.gt_percent_to_px_xywh_list(gt_pct_lines, W, H)
        
        pr_data = np.array(pred_px_lines_with_scores, dtype=float)
        if pr_data.ndim == 1 and pr_data.size == 5: 
            pr_data = pr_data[None, :]
        
        pr_boxes_only = pr_data[:, :4] if pr_data.size > 0 else pr_data
        ious = cls.iou_matrix_xywh(gt_px, pr_boxes_only)

        out = {}
        for t in thresholds:
            pairs = cls.greedy_match(ious, t)
            TP = len(pairs)
            FP = pr_boxes_only.shape[0] - TP
            FN = gt_px.shape[0] - TP
            P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            
            map_score = cls.calculate_map_with_scores(gt_px, pr_data, t)
            
            out[f"TP@{t}"] = TP
            out[f"FP@{t}"] = FP
            out[f"FN@{t}"] = FN
            out[f"Precision@{t}"] = P
            out[f"Recall@{t}"] = R
            out[f"F1@{t}"] = F1
            out[f"mAP@{t}"] = map_score 
        
        return out
    
    @classmethod
    def make_gt_lookup(cls, gt_df, key_col="image_id", gt_bbox_col="bbox_in_line", w_col="width", h_col="height"):
        return {row[key_col]: (row[gt_bbox_col], float(row[w_col]), float(row[h_col])) 
                for _, row in gt_df[[key_col, gt_bbox_col, w_col, h_col]].iterrows()}
    
    @classmethod
    def add_metrics_to_pred_df(cls, pred_df, gt_lookup, key_col="image_id",
                               pred_bbox_col="bbox_in_line", confidence_col="confidence", 
                               thresholds=(0.5, 0.75), prefix="tess", bbox_format="default"):

        out = pred_df.copy()

        def _row_metrics(row):
            k = row[key_col]
            if k not in gt_lookup:
                return pd.Series({f"{prefix}_{m}@{t}": np.nan
                                  for t in thresholds
                                  for m in ["TP", "FP", "FN", "Precision", "Recall", "F1", "mAP"]})
            
            gt_boxes, W, H = gt_lookup[k]
            
            if cls._is_empty_boxes(gt_boxes) or cls._is_empty_boxes(row[pred_bbox_col]):
                return pd.Series({f"{prefix}_{m}@{t}": np.nan
                                  for t in thresholds
                                  for m in ["TP", "FP", "FN", "Precision", "Recall", "F1", "mAP"]})
            
            pred_boxes = row[pred_bbox_col]
            confidence_scores = row[confidence_col] if confidence_col in row else [1.0] * len(pred_boxes)
            
            if len(confidence_scores) != len(pred_boxes):
                confidence_scores = [1.0] * len(pred_boxes)
            
            pred_with_scores = []
            for i, bbox in enumerate(pred_boxes):
                if i < len(confidence_scores):
                    pred_with_scores.append(bbox + [confidence_scores[i]])
                else:
                    pred_with_scores.append(bbox + [1.0])
            
            res = cls.metrics_for_thresholds(gt_boxes, pred_with_scores, W, H, thresholds)
            flat = {}
            for t in thresholds:
                for m in ["TP", "FP", "FN", "Precision", "Recall", "F1", "mAP"]:
                    flat[f"{prefix}_{m}@{t}"] = res[f"{m}@{t}"]
            return pd.Series(flat)

        metrics_cols = out.apply(_row_metrics, axis=1)
        out = pd.concat([out, metrics_cols], axis=1)
        return out
    
    @staticmethod
    def summarize_pred_df(pred_eval_df, prefix="tess", thresholds=(0.5, 0.75)):
        cols = [f"{prefix}_{m}@{t}" for t in thresholds for m in ["TP", "FP", "FN", "Precision", "Recall", "F1", "mAP"]]
        present = [c for c in cols if c in pred_eval_df.columns]
        if not present:
            return pd.DataFrame()
        means = pred_eval_df[present].mean(numeric_only=True)
        row = {"engine": prefix}
        row.update({k: float(v) for k, v in means.items()})
        return pd.DataFrame([row])