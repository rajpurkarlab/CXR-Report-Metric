
import pandas as pd
import numpy as np
from fast_bleu import BLEU
from sklearn.metrics import f1_score
import scipy.stats
from bert_score import BERTScorer
import torch
import json

# PATHS
#gt_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/reports.csv"
#gt_path = "final/gt_copy.csv"
gt_path = "final/test_impressions.csv"
#gt_path = "/deep/u/fyu9/dygiepp/metric-oracles/gt_oracle_reports.csv"
bleu_hypo = "final/bleu_copy.csv"
#radgraph_hypo = "/deep/u/fyu9/CXR-RePaiR/mimic_entity_study_level_retrieval.csv"
radgraph_hypo = "final/radgraph_copy.csv"
#semb_hypo = "/deep/u/rayank/CXR-RePaiR/semb_predicted_reports.csv"
#semb_hypo = "/deep/u/rayank/CXR-RePaiR/semb_hypothetical_eval.csv"
semb_hypo = "final/semb_copy.csv"
#bertscore_hypo = "/deep/u/markendo/CXR-RePaiR/bertscore_best_matches/bertscore_mimic_study_level_retrieval.csv"
bertscore_hypo = "final/bertscore_copy.csv"
# existing models PATHS
m2trans_path = "final/m2trans_copy.csv"
r2gen_path = "final/r2gen_copy.csv"
cxr_repair_path = "final/cxr_repair_copy.csv"
cxr_repair_2_path = "final/cxr_repair_2_copy.csv"
wcl_path = "final/wcl_copy.csv"
warm_starting_path = "final/warm_starting_copy.csv"
random_path = "final/random_copy.csv"
"""
m2trans_path = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/M2-Trans/generated_reports.csv"
r2gen_path = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/R2Gen/generated_reports.csv"
cxr_repair_path = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/CXR-RePaiR-New/CXR-RePaiR-Select/generated_reports.csv"
cxr_repair_2_path = "/deep/u/fyu9/dygiepp/CXR-RePaiR-RadGraph/clip_2_generated_reports.csv"
wcl_path = "/deep/u/fyu9/WCL/wcl_generated_reports.csv"
warm_starting_path = "/deep/u/fyu9/warm-start/generated_reports.csv"
"""

# F1
gt_path_labels = "/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/labels.csv"
useful_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']
short_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','EC','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']

weights = {'bigram': (1/2., 1/2.)}

def add_radgraph_col(pred_df, method):
    paths = {}
    paths["m2trans_entity"] = "/deep/u/fyu9/dygiepp/m2trans_radgraph_entity_f1.json"
    paths["m2trans_relation"] = "/deep/u/fyu9/dygiepp/m2trans_radgraph_relation_f1.json"
    paths["r2gen_entity"] = "/deep/u/fyu9/dygiepp/r2gen_radgraph_entity_f1.json"
    paths["r2gen_relation"] = "/deep/u/fyu9/dygiepp/r2gen_radgraph_relation_f1.json"
    paths["cxr_repair_2_entity"] = "/deep/u/fyu9/dygiepp/CXR-RePaiR-RadGraph/clip_2_radgraph_entity_f1.json"
    paths["cxr_repair_2_relation"] = "/deep/u/fyu9/dygiepp/CXR-RePaiR-RadGraph/clip_2_radgraph_relation_f1.json"
    paths["cxr_repair_entity"] = "/deep/u/fyu9/dygiepp/CXR-RePaiR-RadGraph/clip_select_radgraph_entity_f1.json"
    paths["cxr_repair_relation"] = "/deep/u/fyu9/dygiepp/CXR-RePaiR-RadGraph/clip_select_radgraph_relation_f1.json"
    paths["wcl_entity"] = "/deep/u/fyu9/dygiepp/wcl_radgraph_entity_f1.json"
    paths["wcl_relation"] = "/deep/u/fyu9/dygiepp/wcl_radgraph_relation_f1.json"
    paths["warm_starting_entity"] = "/deep/u/fyu9/dygiepp/warmstart_radgraph_entity_f1.json"
    paths["warm_starting_relation"] = "/deep/u/fyu9/dygiepp/warmstart_radgraph_relation_f1.json"
    paths["random_entity"] = "/deep/u/fyu9/dygiepp/random_radgraph_entity_f1.json"
    paths["random_relation"] = "/deep/u/fyu9/dygiepp/random_radgraph_relation_f1.json"
    study_id_to_radgraph = {}
    with open(paths[method + "_entity"], "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            study_id_to_radgraph[int(study_id)] = float(f1)
    with open(paths[method + "_relation"], "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            study_id_to_radgraph[int(study_id)] += float(f1)
            study_id_to_radgraph[int(study_id)] /= float(2)
    radgraph_scores = []
    count = 0
    for i, row in pred_df.iterrows():
        #if int(row['study_id']) not in study_id_to_radgraph:
        #    radgraph_scores.append(row['study_id'])
        radgraph_scores.append(study_id_to_radgraph[int(row['study_id'])])
    pred_df["radgraph_combined"] = radgraph_scores
    #print(sorted(list(study_id_to_radgraph.keys()))[:5])
    #print(sorted(radgraph_scores)[:5])
    return pred_df

def eval_semb(pred_df, semb_path):
    #gt_path = "cache/mimic_test_impressions_preds_rep.pt"
    gt_path = "final/test_impressions.pt"
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    np_label_embeds = torch.stack([*label_embeds.values()], dim=0).numpy()
    np_pred_embeds = torch.stack([*pred_embeds.values()], dim=0).numpy()
    print(len(pred_df), len(np_pred_embeds), len(np_label_embeds))
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum()/(np.linalg.norm(label)*np.linalg.norm(pred))
        scores.append(sim_scores)
    bootstrap = np.random.choice(scores, size=5000, replace=True)
    mean, ste = np.mean(bootstrap), scipy.stats.sem(bootstrap)
    ci  = ste * scipy.stats.t.ppf((1 + 0.95) / 2., len(bootstrap)-1)
    print(round(mean-ci, 3), round(mean, 3), round(mean+ci, 3))


def add_semb_col(pred_df, semb_path):
    #gt_path = "cache/mimic_test_impressions_preds_rep.pt"
    gt_path = "final/test_impressions.pt"
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

def add_bertscore_col(gt_df, pred_df):
    test_reports = gt_df["report"].tolist()
    test_reports = [test.lstrip() for test in test_reports]
    method_reports = pred_df["report"].tolist()
    method_reports = [report.lstrip() for report in method_reports]
    """
    new_method_reports = []
    count = 0
    for r in method_reports:
        if type(r) == float:
            new_method_reports.append("")
            count += 1
        else:
            new_method_reports.append(r.lstrip())
    method_reports = new_method_reports
    print(" ", count)
    """

    scorer = BERTScorer(model_type="distilroberta-base", batch_size=256)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

"""
gt_df, pred_df: pandas data frame with column "study_id" and column "report"
returns: single float f1 score
"""
def eval_f1(gt_df, pred_df):
    _gt_df = gt_df.replace(-1, 0)
    _pred_df = pred_df.replace(-1, 0)
    new_gt = []
    new_pred = []
    for i, row in _gt_df.iterrows():
        pred_row = _pred_df[pred_df['study_id'] == row['study_id']]
        if len(pred_row) == 0:
            scores.append(0)
            continue
        pred_row = pred_row[useful_labels]
        if 1 not in row[useful_labels].values:
            #print("BIG PROBLEM")
            #print(row['study_id'])
            #print(row[useful_labels].values)
            row["No Finding"] = 1
        new_gt.append(row[useful_labels].values)
        new_pred.append(pred_row[useful_labels].values[0])
            
    new_gt_np = np.array(new_gt)
    new_pred_np = np.array(new_pred)
    print(new_gt_np.shape, new_pred_np.shape)
    accs = []
    for i in range(new_gt_np.shape[1]):
        accs.append(f1_score(new_gt_np[:, i], new_pred_np[:, i]))
    accs.append(np.array(accs).mean())
    accs_df = pd.DataFrame([accs], columns=[*short_labels, "Ave"])
    return accs_df


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



def eval_bleu(gt_df, pred_df):
    scores = []
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row['report']])[0]
        pred_row = pred_df[pred_df['study_id'] == row['study_id']]
        if len(pred_row) == 0:
            scores.append(0)
            continue
        predicted_report = prep_reports([pred_row['report'].values[0]])[0]
        if len(pred_row) == 1:
            bleu = BLEU([gt_report], weights)
            score = bleu.get_score([predicted_report])['bigram']
            assert len(score) == 1
            scores.append(score[0])
    #print(np.array(scores).mean())
    scores = np.array(scores)
    bootstrap = np.random.choice(scores, size=5000, replace=True)
    mean, ste = np.mean(bootstrap), scipy.stats.sem(bootstrap)
    ci  = ste * scipy.stats.t.ppf((1 + 0.95) / 2., len(bootstrap)-1)
    print(round(mean-ci, 3), round(mean, 3), round(mean+ci, 3))

def check_df(df):
    assert "report" in df.columns
    assert "study_id" in df.columns
    df = df.drop_duplicates(subset=["study_id"])
    df = df.dropna()
    return df

"""
take a radgraph predicted report and fill missing study ids
"""
def preprocess_radgraph(gt_df, rad_df):
    for i, row in gt_df.iterrows():
        rad_row = rad_df[rad_df['study_id'] == row['study_id']]
        if len(rad_row) != 1:
            print("added", row['study_id'])
            rad_df = rad_df.append({"study_id": row['study_id'], "report": "", "gt":row["report"]}, ignore_index=True)
    return rad_df

"""
the gt_df needs to be UNPROCESSED
take an ordered bleu prediction file and add study id column
"""
def preprocess_bleu(gt_df, pred_df):
    print(len(gt_df), len(pred_df))
    assert len(gt_df) == len(pred_df)
    pred_df['study_id'] = gt_df['study_id']
    return pred_df

def get_f1_df(name, path):
    if name in ["semb", "bleu", "bertscore", "radgraph"]:
        pred_labels = pd.read_csv(("/deep/u/rayank/CXR-RePaiR/cache/"+name+"/labeled_reports.csv"), index_col=False).fillna(0)[useful_labels]
    else:
        pred_labels = pd.read_csv(gt_path_labels, index_col=False).fillna(0)[useful_labels]
    original_df = pd.read_csv(path)
    if 'study_id' not in original_df.columns:
        original_df = preprocess_bleu(pd.read_csv(gt_path), original_df)
    assert len(original_df) == len(pred_labels)
    pred_labels['study_id'] = original_df['study_id']
    return pred_labels

if __name__ == "__main__":
    # sort everything by study id
    """
    options = [m2trans_path, r2gen_path, cxr_repair_path, cxr_repair_2_path, wcl_path, warm_starting_path]
    options = [semb_hypo]
    gt_df = pd.read_csv(gt_path)
    for opt in options:
        _df = pd.read_csv(opt)
        #_df = _df[_df.study_id.isin(gt_df['study_id'])]
        _df = check_df(_df)
        print(opt, _df.shape)
        _df = _df.sort_values('study_id')
        _df.to_csv(opt)
    """

    # preprocess two
    """
    gt_df = pd.read_csv(gt_path)
    hypo_df = pd.read_csv(wcl_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = hypo_df[hypo_df.study_id.isin(gt_df['study_id'])]
    print(hypo_df.shape)
    hypo_df.to_csv(wcl_path)

    gt_df = pd.read_csv(gt_path)
    hypo_df = pd.read_csv(warm_starting_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = hypo_df[hypo_df.study_id.isin(gt_df['study_id'])]
    print(hypo_df.shape)
    hypo_df.to_csv(warm_starting_path)
    """

    # get new semb df
    """
    gt_df = pd.read_csv(gt_path)
    hypo_df = pd.read_csv(semb_hypo)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = add_semb_col(hypo_df, semb_hypo.replace(".csv", "_imp.pt"))
    #hypo_df = add_bleu_col(gt_df, hypo_df)
    print(hypo_df.head())
    hypo_df.to_csv("final/semb_copy.csv")
    print("done")
    """

    # add the columns
    """
    print("start")
    hypo_df = pd.read_csv("final/new_col_bertscore_copy.csv")
    gt_df = pd.read_csv(gt_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = add_bertscore_col(gt_df, hypo_df)
    hypo_df.to_csv("final/new_col_bertscore_copy.csv") 
    print("end")
    print("start")
    hypo_df = pd.read_csv("final/new_col_radgraph_copy.csv")
    gt_df = pd.read_csv(gt_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = add_bertscore_col(gt_df, hypo_df)
    hypo_df.to_csv("final/new_col_radgraph_copy.csv") 
    print("end")
    print("start")
    hypo_df = pd.read_csv("final/new_col_semb_copy.csv")
    gt_df = pd.read_csv(gt_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = add_bertscore_col(gt_df, hypo_df)
    hypo_df.to_csv("final/new_col_semb_copy.csv") 
    print("end")
    print("start")
    hypo_df = pd.read_csv("final/new_col_bleu_copy.csv")
    gt_df = pd.read_csv(gt_path)
    hypo_df = check_df(hypo_df)
    gt_df = check_df(gt_df)
    hypo_df = add_bertscore_col(gt_df, hypo_df)
    hypo_df.to_csv("final/new_col_bleu_copy.csv") 
    print("end")
    exit()
    """

    # get new bleu df
    options = [random_path, r2gen_path, cxr_repair_path, cxr_repair_2_path, wcl_path, warm_starting_path, m2trans_path]
    methods = ["random", "r2gen", "cxr_repair", "cxr_repair_2", "wcl", "warm_starting", "m2trans"]
    gt_df = pd.read_csv(gt_path)
    for opt, method in zip(options, methods):
        print(opt)
        hypo_df = pd.read_csv(opt)
        #hypo_df = preprocess_bleu(gt_df, hypo_df)
        hypo_df = check_df(hypo_df)
        gt_df = check_df(gt_df)
        hypo_df = add_radgraph_col(hypo_df, method)
        hypo_df = add_bleu_col(gt_df, hypo_df)
        hypo_df = add_semb_col(hypo_df, opt.replace(".csv", "_imp.pt"))
        hypo_df = add_bertscore_col(gt_df, hypo_df)
        hypo_df.to_csv(opt.replace("final/", "final/new_col_")) 


    # calc blue or semb
    """
    gt_df = pd.read_csv(gt_path)
    r2gen_df = pd.read_csv(r2gen_path)
    m2trans_df = pd.read_csv(m2trans_path)
    cxr_repair_df = pd.read_csv(cxr_repair_path)
    cxr_repair_2_df = pd.read_csv(cxr_repair_2_path)
    semb_hypo_df = pd.read_csv(semb_hypo)
    random_df = pd.read_csv(random_path)
    wcl_df = pd.read_csv(wcl_path)
    warm_starting_df = pd.read_csv(warm_starting_path)

    #r2gen_df = preprocess_bleu(gt_df, r2gen_df)
    #m2trans_df = preprocess_bleu(gt_df, m2trans_df)
    #cxr_repair_df = preprocess_bleu(gt_df, cxr_repair_df)
    r2gen_df = check_df(r2gen_df)
    m2trans_df = check_df(m2trans_df)
    cxr_repair_df = check_df(cxr_repair_df)
    semb_hypo_df = check_df(semb_hypo_df)
    random_df = check_df(random_df)
    wcl_df = check_df(wcl_df)
    warm_starting_df = check_df(warm_starting_df)
    cxr_repair_2_df = check_df(cxr_repair_2_df)
    gt_df = check_df(gt_df)

    methods = [("r2gen", r2gen_df), ("m2trans", m2trans_df), ("cxr_repair", cxr_repair_df)]
    methods.append(("random", random_df))
    methods.append(("wcl", wcl_df))
    methods.append(("warm_starting", warm_starting_df))
    methods.append(("cxr_repair_2", cxr_repair_2_df))
    methods.append(("semb", semb_hypo_df))
    for name, df in methods:
        print(name)
        eval_semb(df, "final/" + name + "_copy_imp.pt")
        #eval_bleu(gt_df, df)
    """

    """
    gt_df = pd.read_csv(gt_path)
    print("gt", gt_df.shape)

    bleu_hypo_df = pd.read_csv(bleu_hypo)
    bleu_hypo_df = preprocess_bleu(gt_df, bleu_hypo_df)
    bleu_hypo_df = check_df(bleu_hypo_df)
    print("bleu", bleu_hypo_df.shape)

    semb_hypo_df = pd.read_csv(semb_hypo)
    semb_hypo_df = preprocess_bleu(gt_df, semb_hypo_df)
    semb_hypo_df = check_df(semb_hypo_df)
    print("semb", semb_hypo_df.shape)

    gt_df = check_df(gt_df)
    print("gt", gt_df.shape)

    radgraph_hypo_df = preprocess_radgraph(gt_df, pd.read_csv(radgraph_hypo))
    radgraph_hypo_df = check_df(radgraph_hypo_df)
    print("radgraph", bleu_hypo_df.shape)

    bertscore_hypo_df = check_df(pd.read_csv(bertscore_hypo))
    print("bertscore", bertscore_hypo_df.shape)

    methods = [("bleu", bleu_hypo_df), ("semb", semb_hypo_df), ("radgraph", radgraph_hypo_df), ("bertscore", bertscore_hypo_df)]
    for name, df in methods:
        break
        print(name)
        eval_bleu(gt_df, df)

    gt_f1_df = get_f1_df("gt", gt_path)
    radgraph_f1_df = get_f1_df("radgraph", radgraph_hypo)
    bertscore_f1_df = get_f1_df("bertscore", bertscore_hypo)
    bleu_f1_df = get_f1_df("bleu", bleu_hypo)
    semb_f1_df = get_f1_df("semb", semb_hypo)
    print(gt_f1_df.shape, bleu_f1_df.shape)
    print(eval_f1(gt_f1_df, bleu_f1_df))
    """

    







