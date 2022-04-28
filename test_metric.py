from CXRMetric.run_eval import calc_metric

gt_reports = "reports/test_impressions.csv"
predicted_reports = "reports/generated_reports.csv"
out_file = "out/report_scores.csv"
calc_metric(gt_reports, predicted_reports, out_file)



