import re, unicodedata, string
import jiwer

def normalize_text(text,
                   lowercase=True,
                   remove_punct=True,
                   remove_diacritics=False,
                   dehyphenate=True):
    import re, string, unicodedata
    PUNCT = string.punctuation + "¿¡“”«»—–·…"
    CID_RE = re.compile(r"\(cid:\d+\)")

    if isinstance(text, (list, tuple)):
        s = " ".join(str(t) for t in text if t)
    else:
        s = "" if text is None else str(text)

    s = unicodedata.normalize("NFKC", s)
    s = CID_RE.sub("", s)

    if dehyphenate:
        s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)

    s = s.replace("\n", " ")
    if lowercase:
        s = s.lower()

    if remove_diacritics:
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')

    if remove_punct:
        s = s.translate(str.maketrans({c: " " for c in PUNCT}))

    s = re.sub(r"\s+", " ", s).strip()
    return s



def compute_wer_simple(ground_truth, hypothesis, **norm_opts):
    gt = normalize_text(ground_truth, **norm_opts)
    hyp = normalize_text(hypothesis, **norm_opts)
    return jiwer.wer(gt, hyp)


def compute_cer_simple(ground_truth, hypothesis, keep_spaces: bool = False, **norm_opts):

    gt = normalize_text(ground_truth, **norm_opts)
    hyp = normalize_text(hypothesis, **norm_opts)

    if not keep_spaces:
        gt = gt.replace(" ", "")
        hyp = hyp.replace(" ", "")

    N = len(gt)
    if N == 0:
        # if ref is empty and pred too, then -> error 0; if ref is empty and pred no, then -> 1.0
        return 0.0 if len(hyp) == 0 else 1.0

    return jiwer.cer(gt, hyp)

def compute_metrics_wer_cer(predictions_df, ground_truth_df, 
                     predictions_id_col='filename', 
                     predictions_text_col='words',
                     ground_truth_id_col='filename',
                     ground_truth_words_col='words',
                     ground_truth_chars_col='characters'):
 
    ground_truth_words_dict = {}
    ground_truth_chars_dict = {}

    for idx, row in ground_truth_df.iterrows():
        id_file = row[ground_truth_id_col]
        words = row[ground_truth_words_col] 
        characters = row[ground_truth_chars_col] 
        ground_truth_words_dict[id_file] = words
        ground_truth_chars_dict[id_file] = characters
   
    wer_scores = []; cer_scores = []

    for idx, row in predictions_df.iterrows():
        filename = row[predictions_id_col]
        prediction_text = row[predictions_text_col]
        
        ground_truth_words = ground_truth_words_dict.get(filename)
        ground_truth_chars = ground_truth_chars_dict.get(filename)
        
        if ground_truth_words is not None and ground_truth_chars is not None:
            wer_score = compute_wer_simple(ground_truth_words, prediction_text)
            wer_scores.append(wer_score)
            
            cer_score = compute_cer_simple(ground_truth_chars, prediction_text,
                                         remove_punct=False,
                                         remove_diacritics=False,
                                         keep_spaces=False)
            cer_scores.append(cer_score)
        else:
            print(f"ground-truth was not found: {filename}")
            wer_scores.append(None)
            cer_scores.append(None)
    result_df = predictions_df.copy()
    result_df['wer_score'] = wer_scores
    result_df['cer_score'] = cer_scores
    
    return result_df