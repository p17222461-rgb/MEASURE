from typing import List, Sequence, Any, Tuple, Dict
from utils.ocr_metrics import normalize_text
import pandas as pd
import collections
import sacrebleu
import math


## ------ NDLD METRIC--------------
def ensure_list_tokens(x: Any) -> List[str]:
    return [str(t) for t in x]

def seq_from_text(x: Any, level="word", normalize_fn=normalize_text, **norm_kwargs) -> List[str]:
    if level not in ("word", "char"):
        raise ValueError("level must be 'word' or 'char'")
    
    if isinstance(x, str) and x.startswith('[') and x.endswith(']') and level == "word":
        try:
            import ast
            parsed_list = ast.literal_eval(x)
            if isinstance(parsed_list, list):
                tokens = ensure_list_tokens(parsed_list)
                tokens = [normalize_fn(tok, **norm_kwargs) for tok in tokens if normalize_fn(tok, **norm_kwargs)]
                return tokens
        except:
            pass 
    
    if isinstance(x, list) and level == "word":
        tokens = ensure_list_tokens(x)
        tokens = [normalize_fn(tok, **norm_kwargs) for tok in tokens if normalize_fn(tok, **norm_kwargs)]
        return tokens
    
    s = x if isinstance(x, str) else ("" if x is None else str(x))
    s = normalize_fn(s, **norm_kwargs)
    return list(s) if level == "char" else s.split()

def damerau_levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0: return len2
    if len2 == 0: return len1
    da = collections.defaultdict(int)
    maxdist = len1 + len2
    d = [[0]*(len2+2) for _ in range(len1+2)]
    d[0][0] = maxdist
    for i in range(len1+1):
        d[i+1][0] = maxdist
        d[i+1][1] = i
    for j in range(len2+1):
        d[0][j+1] = maxdist
        d[1][j+1] = j
    for i in range(1, len1+1):
        db = 0
        s_i = seq1[i-1]
        for j in range(1, len2+1):
            t_j = seq2[j-1]
            i1 = da[t_j]
            j1 = db
            cost = 0 if s_i == t_j else 1
            if cost == 0:
                db = j
            d[i+1][j+1] = min(
                d[i][j] + cost,          # substitution
                d[i+1][j] + 1,           # insertion
                d[i][j+1] + 1,           # deletion
                d[i1][j1] + (i-i1-1) + 1 + (j-j1-1)  # transposition
            )
        da[s_i] = i
    return d[len1+1][len2+1]

def compute_ndld_dataframe(gt_df, pred_df, gt_col="ro_transcription", pred_col=None, key=None, level="word", normalize_fn=normalize_text, align="inner"):
    
    if key:
        merged = gt_df[[key, gt_col]].merge(
            pred_df[[key, pred_col]], on=key, how=align
        )
    else:
        merged = pd.DataFrame({
            gt_col: gt_df[gt_col].values,
            pred_col: pred_df[pred_col].values
        })

    def _seq(x):
        return seq_from_text(
            x,
            level=("char" if level == "character" else "word"),
            normalize_fn=normalize_fn,   # pass through
        )

    records = []
    for idx, row in merged.iterrows():
        gt = row.get(gt_col, "")
        pr = row.get(pred_col, "")

        gt_seq = _seq(gt)
        pr_seq = _seq(pr)

        dist  = damerau_levenshtein_distance(gt_seq, pr_seq)
        denom = max(len(gt_seq), len(pr_seq))
        ndld_val = (dist / denom) if denom else 0.0

        records.append({
            (key or "row_index"): (row[key] if key else idx),
            "gt_len": len(gt_seq),
            "pred_len": len(pr_seq),
            "distance": dist,
            "denom": denom,
            "ndld": ndld_val,
            "similarity": 1.0 - ndld_val,
        })
    return pd.DataFrame(records)

# -------- BLEU METRIC -------------------

def to_plain_string(v: Any) -> str:
    if isinstance(v, list):
        # Extract text
        parts: List[str] = []
        for it in v:
            if isinstance(it, dict) and "text" in it:
                parts.append(str(it["text"]))
            else:
                parts.append(str(it))
        if parts and all(len(p) == 1 for p in parts):
            return "".join(parts)
        return " ".join(parts)
    return "" if v is None else str(v)


def _normalize_words(s: str) -> str:
    return normalize_text(
        s,
        lowercase=True,
        remove_punct=True,
        remove_diacritics=True,
        dehyphenate=True,
    )

def prepare_refs_hyps(ensayo_ro: pd.DataFrame, pred_df: pd.DataFrame, pred_col: str, gt_filename_col: str = "filename",
    gt_text_col: str = "ro_transcription") -> Tuple[List[str], List[str]]:
    
    gt_tmp = ensayo_ro[[gt_filename_col, gt_text_col]].copy()
    pred_tmp = pred_df[[gt_filename_col, pred_col]].copy()

    gt_tmp["ref_str"] = gt_tmp[gt_text_col].apply(_normalize_words)
    pred_tmp["hyp_str"] = pred_tmp[pred_col].apply(to_plain_string).apply(_normalize_words)

    merged = gt_tmp.merge(pred_tmp[[gt_filename_col, "hyp_str"]], on=gt_filename_col, how="inner")
    merged = merged[(merged["ref_str"].str.len() > 0) | (merged["hyp_str"].str.len() > 0)].copy()

    hyps = merged["hyp_str"].tolist()
    refs = merged["ref_str"].tolist()
    return hyps, refs


def compute_bleu(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    page_scores: List[float] = []
    for hyp, ref in zip(hyps, refs):
        sc = sacrebleu.sentence_bleu(hyp, [ref]).score  
        page_scores.append(float(sc))
    avg_page_bleu = float(sum(page_scores) / len(page_scores)) if page_scores else 0.0

    bleu_obj = sacrebleu.BLEU(tokenize="none", effective_order=True)
    corp = bleu_obj.corpus_score(hyps, [refs])

    out = {
        "pages_scored": float(len(page_scores)),
        "avg_page_bleu": float(avg_page_bleu),
        "corpus_bleu": float(corp.score),
        "p1": float(corp.precisions[0]),
        "p2": float(corp.precisions[1]),
        "p3": float(corp.precisions[2]),
        "p4": float(corp.precisions[3]),
        "brevity_penalty": float(corp.bp),
    }
    return out


# just for debugging:

def add_brevity_penalty_column(
    ensayo_ro: pd.DataFrame,
    pred_df: pd.DataFrame,
    pred_col: str,
    gt_filename_col: str = "filename",
    gt_text_col: str = "ro_transcription"
) -> pd.DataFrame:
    gt_tmp = ensayo_ro[[gt_filename_col, gt_text_col]].copy()
    pred_tmp = pred_df[[gt_filename_col, pred_col]].copy()

    gt_tmp["ref_str"] = gt_tmp[gt_text_col].apply(_normalize_words)
    pred_tmp["hyp_str"] = pred_tmp[pred_col].apply(to_plain_string).apply(_normalize_words)

    merged = gt_tmp.merge(pred_tmp[[gt_filename_col, "hyp_str"]], on=gt_filename_col, how="inner")
    merged = merged[(merged["ref_str"].str.len() > 0) | (merged["hyp_str"].str.len() > 0)].copy()

    def _row_bp_p4(row) -> tuple[float, float]:
        s = sacrebleu.sentence_bleu(
            row["hyp_str"],
            [row["ref_str"]],
            tokenize="none",
            use_effective_order=True
        )
        p4 = float(s.precisions[3]) if len(s.precisions) >= 4 else math.nan
        return float(s.bp), p4

    merged[["brevity_penalty", "p4"]] = merged.apply(_row_bp_p4, axis=1, result_type="expand")
    return merged