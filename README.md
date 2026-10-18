# C2F-CSLRNet

Research components and saved experimental results for **Component-Conditioned
Coarse-to-Fine Retrieval and Dynamic Residual Scoring for Multivariate
Time-Series Anomaly Detection**.

Manuscript authors: Siyu Zhu, Jiahao Zhang, Jie Liu, Yonghao Yang, Yanshan Qian,
Yuli Chen, Lin Miao and Xiulei Liu.

The method separates two decisions:

- A forecaster selects historical context through component-specific source,
  patch and local-endpoint retrieval.
- A residual scorer combines two calibrated tail-memory streams using a
  history-conditioned bounded gate. Gate fitting uses official Tuning labels;
  Evaluation and cross-predictor transfer use the frozen gate.

## Current research data

The versioned [public research-data release](https://github.com/yinnierzaidemeng/C2F-CSLRNet/releases/tag/data-v1.0.0)
provides the saved evaluation tables, per-file records, scientific settings,
provenance and safe array files supporting the current manuscript. Download
`C2F_public_research_data_20261018.zip` and start with `PUBLIC_DATA_INDEX.json`.
Its source bindings distinguish original saved bytes from path-normalized
public copies. Data values retain the saved results, including unfavorable
comparisons. See [data access and licensing](data/README.md).

The numerical component source remains version 0.1.0. Release preparation
checks source syntax, file hashes and saved-result provenance. Full training
execution uses the benchmark, the corresponding fitted artifacts and numerical
environment described in [implementation details](docs/IMPLEMENTATION.md).

## Contents

| Directory | Content |
|---|---|
| `src/` | Reusable predictor, normalization and dynamic-scoring components |
| `configs/` | Scientific settings without private machine paths |
| `results/` | Preserved 3 October saved-result snapshot; current data are in the versioned release |
| `data/` | Current public-data access, citation and licensing information |
| `checkpoints/` | Small frozen scoring-gate and constant-mixture checkpoints |
| `scripts/` | Standard-library file-integrity checks |
| `docs/` | Implementation, data sources and scope notes |

See [implementation details](docs/IMPLEMENTATION.md),
[data sources](docs/DATASETS.md), and [release scope](docs/REPRODUCIBILITY.md).
The [result index](results/index.json) and verifier identify the actual supplied files; they do not
require the original experimental directory layout.

## Inspect the saved results

The saved-artifact check requires Python 3.10 or newer and no scientific
packages:

```bash
python scripts/verify_release.py
python scripts/verify_artifacts.py
python scripts/verify_public_data.py /path/to/C2F_public_research_data_20261018.zip
```

These commands read files and compare hashes. They do not load PyTorch
checkpoints, decode numerical arrays, recalculate metrics, train a model or use
network services. A passing report confirms the delivered files, not an
independent reproduction of the experiments.

The two main studies use different predictor fits:

| Study | Example corrected VUS-PR means | Interpretation |
|---|---|---|
| Retained-predictor scoring | C2F dynamic 0.4245; fixed mixture 0.4149; learned constant 0.4146 | Compare scoring rules on the same retained residuals |
| Unified-training structures | Full C2F 0.4153; shared relations 0.4176; radius 0 0.4165 | Compare predictor structures under the common training protocol |

The current manuscript uses equal-file evaluation over the official Evaluation
files. Retained-predictor scoring and unified-training structures have separate
fits and result entries. The current data release preserves complete saved
precision, per-file outcomes and paired estimates. Predictor-seed and gate-seed
variability remain separately identified in those records.

The separate `results/retrieval/` folders contain saved synthetic relation
recovery and fixed-candidate local-weight intervention records. They retain
per-system outcomes and unfavorable comparisons; they are not additional
real-data detection cohorts.

## Use the research components

Read [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) before using the source modules
and supplied scientific configurations. Numerical execution requires a
compatible scientific Python environment and externally obtained data. The
minimal dependency declaration is not a tested environment lockfile.

No raw TSB-AD data, full predictor-checkpoint collection, complete residual
trajectory archive or historical cluster launcher is bundled. Frozen scorer
checkpoints alone do not replace a fitted predictor or its calibration
residuals. This release preparation performed static source checks and file
checks, not new model execution.

## Data and method sources

Obtain the benchmark and its official split definitions from
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD). Refer to the original sources
and dataset-specific conditions linked there. See
[THIRD_PARTY.md](docs/THIRD_PARTY.md) for method references and external
dependencies; external baseline implementations and benchmark metric code are
not relabeled as this project's software.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) for the software and manuscript citation, and
[the dataset citation](data/CITATION_DATA.cff) for research-data version 1.0.0.

Original software and associated documentation use the [MIT License](LICENSE).
Original result tables, scientific configurations and validation records use
[CC BY 4.0](LICENSE_DATA.txt). Scorer checkpoints and third-party material
retain their existing rights, as detailed in [LICENSING.md](docs/LICENSING.md).

中文说明见 [README_zh.md](README_zh.md).
