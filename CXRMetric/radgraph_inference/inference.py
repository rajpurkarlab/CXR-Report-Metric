import math
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse

"""Code adapted from https://physionet.org/content/radgraph/1.0.0: models/inference.py."""

# def get_file_list(path):
    
#     """Gets path to all the reports (.txt format files) in the specified folder, and
#     saves it in a temporary json file
    
#         Args:
#             path: Path to the folder containing the reports
#     """
    
#     file_list = [item for item in glob.glob(f"{path}/*.txt")]
    
#     # Number of files for inference at once depends on the memory available.
#     ## Recemmended to use no more than batches of 25,000 files
    
#     with open('./temp_file_list.json', 'w') as f:
#         json.dump(file_list, f)

# def preprocess_reports():
    
#     """ Load up the files mentioned in the temporary json file, and
#     processes them in format that the dygie model can take as input.
#     Also save the processed file in a temporary file.
#     """
    
#     file_list = json.load(open("./temp_file_list.json"))
#     final_list = []
#     for idx, file in enumerate(file_list):

#         temp_file = open(file).read()
#         sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',temp_file).split()
#         temp_dict = {}

#         temp_dict["doc_key"] = file
        
#         ## Current way of inference takes in the whole report as 1 sentence
#         temp_dict["sentences"] = [sen]

#         final_list.append(temp_dict)

#         if(idx % 1000 == 0):
#             print(f"{idx+1} reports done")
    
#     print(f"{idx+1} reports done")
    
#     with open("./temp_dygie_input.json",'w') as outfile:
#         for item in final_list:
#             json.dump(item, outfile)
#             outfile.write("\n")

def preprocess_reports(data_path, start, end, sentence=False, image=False):
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    impressions = pd.read_csv(data_path)
    if start != None and end != None:
        impressions = impressions.iloc[start:end]
    final_list = []
    for idx, row in impressions.iterrows():
        if (isinstance(row["report"], float) and math.isnan(row["report"])): continue
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row["report"]).split()
        temp_dict = {}

        if not sentence:  # Report-level
            if image:  # Different image can have different reports
                temp_dict["doc_key"] = f"{row['dicom_id']}_{row['study_id']}"
            else:
                temp_dict["doc_key"] = str(row["study_id"])
        else:  # Sentence-level
            temp_dict["doc_key"] = f"{row['study_id']}_{row['sentence_id']}"
        
        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

        if(idx % 1000 == 0):
            print(f"{idx+1} reports done")
    
    print(f"{idx+1} reports done")
    
    with open("./temp_dygie_input.json",'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")

def run_inference(model_path, cuda):
    
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path = "./temp_dygie_output.json"
    data_path = "./temp_dygie_input.json"
    
    os.system(f"allennlp predict {model_path} {data_path} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --silent")

def postprocess_reports(data_source, data_split):
    
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    file_name = f"./temp_dygie_output.json"
    data = []

    with open(file_name,'r') as f:
        for line in f:
            data.append(json.loads(line))

    for file in data:
        postprocess_individual_report(file, final_dict, data_source=data_source, data_split=data_split)
    
    return final_dict

def postprocess_individual_report(file, final_dict, data_source=None, data_split="inference"):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = data_split

        if file['doc_key'] in final_dict:  # Handle duplicate study IDs.
            final_dict[file['doc_key'] + '+'] = temp_dict
        else:
            final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
def get_entity(n,r,s):
    
    """Gets the entities for individual reports
    
    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence
        
    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json 
    
    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
        for i,val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab,str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx+1)] = temp_dict
    
    return dict_entity

def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    # os.system("rm temp_file_list.json")
    os.system("rm temp_dygie_input.json")
    os.system("rm temp_dygie_output.json")

def _json_to_csv(path, csv_path):
    with open(path, "r") as f:
        match_results = json.load(f)
    reconstructed_reports = []
    for _, (_, train, match) in match_results.items():
        test_report_id = match[0][0][:8]
        reconstructed_reports.append((test_report_id, train))
    pd.DataFrame(reconstructed_reports, columns=["study_id", "report"]).to_csv(csv_path)

def _add_ids_column(
            csv_path, study_id_csv_path, output_path):
    with open(csv_path, "r") as f:
        generated_reports = pd.read_csv(f)
    with open(study_id_csv_path, "r") as f:
        ids_csv = pd.read_csv(f)
        study_ids = ids_csv["study_id"]
        dicom_ids = ids_csv["dicom_id"]
        subject_ids = ids_csv["subject_id"]
    generated_reports["study_id"] = study_ids
    generated_reports["dicom_id"] = dicom_ids
    generated_reports["subject_id"] = subject_ids
    #generated_reports.drop_duplicates(subset=["study_id"], keep="first")
    generated_reports.to_csv(output_path)

def run(model_path, data_path, out_path, cuda):
    
    # print("Getting paths to all the reports...")
    # get_file_list(data_path)
    # print(f"Got all the paths.")

    ground_truth_reports_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/reports.csv"
    clip_dir = "CXR-RePaiR-RadGraph/"
    clip_raw_dir = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/CXR-RePaiR-New/"
    clip_generated_reports_files = [
            "clip_1_generated_reports.csv",
            "clip_2_generated_reports.csv",
            "clip_3_generated_reports.csv",
            "clip_4_generated_reports.csv",
            "clip_5_generated_reports.csv",
            "clip_6_generated_reports.csv",
            "clip_select_generated_reports.csv",
    ]
    clip_raw_generated_reports_files = [
            "CXR-RePaiR-1/generated_reports.csv",
            "CXR-RePaiR-2/generated_reports.csv",
            "CXR-RePaiR-3/generated_reports.csv",
            "CXR-RePaiR-4/generated_reports.csv",
            "CXR-RePaiR-5/generated_reports.csv",
            "CXR-RePaiR-6/generated_reports.csv",
            "CXR-RePaiR-Select/generated_reports.csv",
    ]
    clip_output_prefixes = [
            "clip_1_generated_",
            "clip_2_generated_",
            "clip_3_generated_",
            "clip_4_generated_",
            "clip_5_generated_",
            "clip_6_generated_",
            "clip_select_generated_",
    ]

    for csv, raw_csv, output_prefix in zip(clip_generated_reports_files,
                            clip_raw_generated_reports_files,
                            clip_output_prefixes):
        data_path = os.path.join(clip_dir, csv)
        raw_data_path = os.path.join(clip_raw_dir, raw_csv)
        _add_ids_column(raw_data_path,
                        ground_truth_reports_path,
                        data_path)
        data_source = "MIMIC-CXR"
        start, end = None, None
        data_split = "CXR-RePaiR"

        print("Preprocessing all the reports...")
        preprocess_reports(data_path, start, end, sentence=False, image=True)
        print("Done with preprocessing.")

        print("Running the inference now... This can take a bit of time")
        run_inference(model_path, cuda)
        print("Inference completed.")

        print("Postprocessing output file...")
        final_dict = postprocess_reports(data_source, data_split)
        print("Done postprocessing.")

        print("Saving results and performing final cleanup...")
        cleanup()

        with open(os.path.join(clip_dir, output_prefix + out_path), 'w') as outfile:
            json.dump(final_dict, outfile)


#     # M2-Trans ground-truth and generated reports
#     ground_truth_reports_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/bootstrap_test/reports.csv"
#     m2trans_generated_reports_path = "m2trans_generated_reports.csv"
#     m2trans_raw_generated_reports_path = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/M2-Trans/generated_reports.csv"  # Without study IDs.
#     r2gen_generated_reports_path = "r2gen_generated_reports.csv"
#     r2gen_raw_generated_reports_path = "/deep/u/markendo/CXR-RePaiR/results/MIMIC-CXR/R2Gen/generated_reports.csv"  # Without study IDs.

#     _add_ids_column(m2trans_raw_generated_reports_path,
#                     ground_truth_reports_path,
#                     m2trans_generated_reports_path)
#     _add_ids_column(r2gen_raw_generated_reports_path,
#                     ground_truth_reports_path,
#                     r2gen_generated_reports_path)

#     data_source = "MIMIC-CXR"
#     start, end = None, None

#     for data_path, output_prefix, data_split in [
#             (m2trans_generated_reports_path, "m2trans_generated_", "train"),
#             (r2gen_generated_reports_path, "r2gen_generated_", "train"),
#             (ground_truth_reports_path, "gt_", "test"),
#     ]:
#         print("Preprocessing all the reports...")
#         preprocess_reports(data_path, start, end, sentence=False, image=True)
#         print("Done with preprocessing.")

#         print("Running the inference now... This can take a bit of time")
#         run_inference(model_path, cuda)
#         print("Inference completed.")

#         print("Postprocessing output file...")
#         final_dict = postprocess_reports(data_source, data_split)
#         print("Done postprocessing.")

#         print("Saving results and performing final cleanup...")
#         cleanup()

#         with open(output_prefix + out_path, 'w') as outfile:
#             json.dump(final_dict, outfile)


#     # MIMIC-CXR sentence-level retrieval results (combined reports)
#     mimic_entity_sent_report_output_path = "../CXR-RePaiR/mimic_sentence_report_entity_retrieval.json"
#     mimic_relation_sent_report_output_path = "../CXR-RePaiR/mimic_sentence_report_relation_retrieval.json"
#     mimic_combined_sent_report_output_path = "../CXR-RePaiR/mimic_sentence_report_combined_retrieval.json"

#     entity_data_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_entity_sentence_retrieval_results.csv"
#     relation_data_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_relation_sentence_retrieval_results.csv"
#     combined_data_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_combined_sentence_retrieval_results.csv"

#     _json_to_csv(mimic_entity_sent_report_output_path, entity_data_path)
#     _json_to_csv(mimic_relation_sent_report_output_path, relation_data_path)
#     _json_to_csv(mimic_combined_sent_report_output_path, combined_data_path)

#     for data_path, output_prefix in \
#             [(entity_data_path, "mimic_entity_sentence_retrieval_results_"),
#              (relation_data_path, "mimic_relation_sentence_retrieval_results_"),
#              (combined_data_path, "mimic_combined_sentence_retrieval_results_")]:
#         data_source = "MIMIC-CXR"
#         data_split = "test"
#         start, end = None, None

#         print("Preprocessing all the reports...")
#         preprocess_reports(data_path, start, end, sentence=False)
#         print("Done with preprocessing.")

#         print("Running the inference now... This can take a bit of time")
#         run_inference(model_path, cuda)
#         print("Inference completed.")

#         print("Postprocessing output file...")
#         final_dict = postprocess_reports(data_source, data_split)
#         print("Done postprocessing.")

#         print("Saving results and performing final cleanup...")
#         cleanup()

#         with open(output_prefix + out_path, 'w') as outfile:
#             json.dump(final_dict, outfile)


#     # MIMIC-CXR train impressions
#     data_source = "MIMIC-CXR"
#     data_split = "train"
#     data_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_train_sentence_impressions.csv"
#     batch_size = 10000

#     for i in tqdm(range(782032 // batch_size + 1)):
#         print("Preprocessing all the reports...")
#         preprocess_reports(data_path, i * batch_size, (i + 1) * batch_size, sentence=True)
#         print("Done with preprocessing.")

#         print("Running the inference now... This can take a bit of time")
#         run_inference(model_path, cuda)
#         print("Inference completed.")

#         print("Postprocessing output file...")
#         final_dict = postprocess_reports(data_source, data_split)
#         print("Done postprocessing.")

#         print("Saving results and performing final cleanup...")
#         cleanup()

#         with open(f"outputs/sent_batch_{i}_" + out_path, 'w') as outfile:
#             json.dump(final_dict, outfile)


#     # MIMIC-CXR test impressions
#     data_source = "MIMIC-CXR"
#     data_split = "test"
#     data_path = "/deep/group/data/med-data/mimic-cxr-jpg-split/mimic_test_sentence_impressions.csv"
#     start, end = None, None

#     print("Preprocessing all the reports...")
#     preprocess_reports(data_path, start, end, sentence=True)
#     print("Done with preprocessing.")

#     print("Running the inference now... This can take a bit of time")
#     run_inference(model_path, cuda)
#     print("Inference completed.")

#     print("Postprocessing output file...")
#     final_dict = postprocess_reports(data_source, data_split)
#     print("Done postprocessing.")

#     print("Saving results and performing final cleanup...")
#     cleanup()

#     with open("sent_test_" + out_path, 'w') as outfile:
#         json.dump(final_dict, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=False,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')

    
    args = parser.parse_args()
    
    run(args.model_path, args.data_path, args.out_path, args.cuda_device)
