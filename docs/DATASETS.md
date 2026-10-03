# Data sources

The experiments use the multivariate portion of the official TSB-AD benchmark:
20 Tuning files and 180 Evaluation files. Obtain the benchmark, split lists and
source-specific license information from the
[official TSB-AD repository](https://github.com/TheDatumOrg/TSB-AD).

TSB-AD itself aggregates datasets with different original sources. Its code or
curation license is not a blanket replacement for each underlying dataset's
terms. This repository does not redistribute raw benchmark files.

The corrected saved tables use post-training targets and no point adjustment.
The main F view weights each of the 180 official files equally. C and X retain
the recorded exact-content and Tuning-overlap sensitivity analyses. No files
are removed from the primary result to improve the reported values.

Full-content duplicates, target support and calibration choices are part of
the frozen experimental protocol. A new training run must reproduce those
choices rather than infer them from rounded table values.

Benchmark reference:

Qinghua Liu and John Paparrizos. The Elephant in the Room: Towards a Reliable
Time-Series Anomaly Detection Benchmark. NeurIPS 2024.

Metric reference:

John Paparrizos et al. Volume Under the Surface: A New Accuracy Evaluation
Measure for Time-Series Anomaly Detection. Proceedings of the VLDB Endowment,
15(11), 2774-2787, 2022. https://doi.org/10.14778/3551793.3551830
