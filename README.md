# C2F-CSLRNet

Research components and saved experimental results for **Component-Conditioned
Coarse-to-Fine Retrieval and Dynamic Residual Scoring for Multivariate
Time-Series Anomaly Detection**, by Jiahao Zhang.

The method separates two decisions:

- A forecaster selects historical context through component-specific source,
  patch and local-endpoint retrieval.
- A residual scorer combines two calibrated tail-memory streams using a
  history-conditioned bounded gate. Gate fitting uses official Tuning labels;
  Evaluation and cross-predictor transfer use the frozen gate.

This is a source-and-results release prepared from the retained research
records. It is not a newly validated, one-command reproduction of every
training run. The paper is an unpublished manuscript; no acceptance, DOI or
public repository URL is asserted here.

## Contents

| Directory | Content |
|---|---|
| `src/` | Reusable predictor, normalization and dynamic-scoring components |
| `configs/` | Scientific settings without private machine paths |
| `results/` | Complete corrected summaries and saved per-file metric tables |
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

Do not pool the studies or calculate gains across them. Complete precision,
all six metrics, F/C/X sensitivities and paired uncertainty are in `results/`.
F gives equal weight to the official 180 Evaluation files. Predictor-seed and
gate-seed variability are separate. Dynamic scoring improves the mean over
constant mixtures in the retained cohort, but the paired intervals include
zero; the full structure does not establish a general detection advantage over
the simpler variants.

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

Use [CITATION.cff](CITATION.cff) for the software and unpublished-manuscript
citation. Add the actual repository URL and published-paper information only
when they exist.

The author's original software and associated documentation use the
[MIT License](LICENSE). Dataset, result and model-file scope is explained in
[LICENSING.md](docs/LICENSING.md); no new license is granted over third-party
datasets or external dependencies.

中文说明见 [README_zh.md](README_zh.md).
