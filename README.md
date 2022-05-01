
## Table of Contents
* Prerequisites
* Requirements
* Usage
* License
* Citing

# Prerequisites:
(yml file, requirement file)

# Requirements:
model checkpoint files

Ground Truth and Predicted reports files must be in the
same order, have a column "reports" which have the reports and
a study id column

# Usage:
```
from CXRMetric.run_eval import calc_metric
calc_metric(gt_reports, predicted_reports, out_file)
```



