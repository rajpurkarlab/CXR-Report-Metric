<img src="figures/metric-radiologist-alignment.jpg" width="700"/>

Repository referenced in the paper "Measuring Progress in Automatic Chest 
X-Ray Radiology Report Generation". This repository provides code for computing
metric scores for radiology report evaluation. The metrics are:
* BLEU
* BERTscore
* CheXbert labeler vector similarity
* RadGraph entity and relation F1
* Composite metric of the above metrics


## Table of Contents
* [Prerequisites](#prerequisites)
* [Requirements](#requirements)
* [Usage](#usage)
* [License](#license)
* [Citing](#citing)


<a name="prerequisites"></a>

# Prerequisites
(yml file, requirement file)
TODO: add dependencies for metrics beyond RadGraph

To install the dependencies, run:
```
pip install -r requirements.txt
```

<a name="requirements"></a>

# Requirements
Ground Truth and Generated reports must be arranged in the same order in a
column named "reports" in two CSV files. The CSVs should also contain a
corresponding "study_id" column that contains unique identifies for the reports.

## CheXbert
TODO

## RadGraph
To compute the RadGraph metric score, download the RadGraph model checkpoint
from PhysioNet [here](https://physionet.org/content/radgraph/1.0.0/). The
checkpoint file can be found under the "Files" section at path
`models/model_checkpoint/`.
The code for computing the RadGraph metric score is adapted from
[dwadden/dygiepp](https://github.com/dwadden/dygiepp).
Note: You need to apply for credentialed access to RadGraph on PhysioNet.


<a name="usage"></a>

# Usage
```
from CXRMetric.run_eval import calc_metric
calc_metric(gt_reports, predicted_reports, out_file)
```


<a name="license"></a>

# License
This repository is made publicly available under the MIT License.


<a name="citing"></a>

# Citing
If you are using this repo, please cite this paper:
```
TODO: bibliography
```
