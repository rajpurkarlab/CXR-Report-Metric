import pandas as pd
import numpy as np
import os

from fast_bleu import BLEU
from bert_score import BERTScorer
import torch

REPORT_COL_NAME ="report"
weights = {'bigram': (1/2., 1/2.)}

def prep_reports(reports):
    return [list(filter(lambda val: val !=  "", str(elem).lower().replace(".", " .").split(" "))) for elem in reports]

# adds a column with the bleu score for each report
def add_bleu_col(gt_df, pred_df):
    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row['report']])[0]
        pred_row = pred_df[pred_df['study_id'] == row['study_id']]
        if len(pred_row) == 0:
            print("problem")
            continue
        predicted_report = prep_reports([pred_row['report'].values[0]])[0]
        if len(pred_row) == 1:
            bleu = BLEU([gt_report], weights)
            score = bleu.get_score([predicted_report])['bigram']
            assert len(score) == 1
            _index = pred_df.index[pred_df['study_id']==row['study_id']].tolist()[0]
            pred_df.at[_index, "bleu_score"] = score[0]
    return pred_df

def add_bertscore_col(gt_df, pred_df):
    test_reports = gt_df["report"].tolist()
    test_reports = [test.lstrip() for test in test_reports]
    method_reports = pred_df["report"].tolist()
    method_reports = [report.lstrip() for report in method_reports]

    scorer = BERTScorer(model_type="distilroberta-base", batch_size=256)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    np_label_embeds = torch.stack([*label_embeds.values()], dim=0).numpy()
    np_pred_embeds = torch.stack([*pred_embeds.values()], dim=0).numpy()
    print(len(pred_df), len(np_pred_embeds), len(np_label_embeds))
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum()/(np.linalg.norm(label)*np.linalg.norm(pred))
        scores.append(sim_scores)
    pred_df["semb_score"] = scores
    return pred_df

"""
cd /deep/group/report-clip/CheXbert/src
python encode.py -c ../models/chexbert.pth -d /deep/u/rayank/CXR-RePaiR/final/radgraph_copy.csv -o /deep/u/rayank/CXR-RePaiR/final/radgraph_copy_imp.pt
python encode.py -c ../models/chexbert.pth -d /deep/u/rayank/CXR-RePaiR/final/bertscore_copy.csv -o /deep/u/rayank/CXR-RePaiR/final/bertscore_copy_imp.pt
"""

def calc_metric(gt_csv, pred_csv, out_csv):
    print(gt_csv, pred_csv)
    # take a csv to the eval an gt reports
    gt, pred = pd.read_csv(gt_csv), pd.read_csv(pred_csv)

    # check that the length is the same, assume that the order is the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)

    # add blue column to the eval df
    #pred = add_bleu_col(gt, pred)

    # add bertscore column to the eval df
    #pred = add_bertscore_col(gt, pred)

    # run encode.py to make the semb column
    print(os.getcwd())
    pred_embed_path = "cache/pred_embeddings.pt"
    gt_embed_path = "cache/gt_embeddings.pt"
    os.system(f"python CXRMetric/CheXbert/src/encode.py -c CXRMetric/CheXbert/models/chexbert.pth -d {pred_csv} -o {pred_embed_path}")
    os.system(f"python CXRMetric/CheXbert/src/encode.py -c CXRMetric/CheXbert/models/chexbert.pth -d {gt_csv} -o {gt_embed_path}")
    pred = add_semb_col(pred, pred_embed_path, gt_embed_path)

    # run radgraph to create that column

    # save results in the out folder
    pred.to_csv(out_csv)
