import tempfile
import os
from pathlib import Path

import pandas as pd


def compute_radcliq(refs, hyps, use_idf=False):
    """Compute RadCliQ metrics for reference and hypothesis reports.

    Args:
        refs: List of ground truth report strings.
        hyps: List of predicted report strings.
        use_idf: Whether to use IDF weighting for BERTScore.

    Returns:
        Dictionary with keys: radgraph_combined, bertscore, semb_score,
        bleu_score, RadCliQ-v0, RadCliQ-v1. Each value is a list of floats.
    """
    from CXRMetric.run_eval import calc_metric

    if len(refs) != len(hyps):
        raise ValueError(f"Length mismatch: {len(refs)} refs vs {len(hyps)} hyps")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_csv = tmp / "gt.csv"
        pred_csv = tmp / "pred.csv"
        out_csv = tmp / "out.csv"

        study_ids = list(range(len(refs)))
        pd.DataFrame({"study_id": study_ids, "report": refs}).to_csv(gt_csv, index=False)
        pd.DataFrame({"study_id": study_ids, "report": hyps}).to_csv(pred_csv, index=False)

        calc_metric(str(gt_csv), str(pred_csv), str(out_csv), use_idf)

        df = pd.read_csv(out_csv).sort_values("study_id")
        return {
            "radgraph_combined": df["radgraph_combined"].astype(float).tolist(),
            "bertscore": df["bertscore"].astype(float).tolist(),
            "semb_score": df["semb_score"].astype(float).tolist(),
            "bleu_score": df["bleu_score"].astype(float).tolist(),
            "RadCliQ-v0": df["RadCliQ-v0"].astype(float).tolist(),
            "RadCliQ-v1": df["RadCliQ-v1"].astype(float).tolist(),
        }
