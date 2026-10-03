# Third-party software and method references

The user selected MIT for the project's original research code. The top-level
license applies to that original code; it does not relicense third-party
datasets, software, article text or illustrations. No external baseline
repository, benchmark metric implementation or official dataset is vendored
in `src/`.

## Runtime dependencies (installed separately)

| Dependency | Purpose | Upstream licensing source |
|---|---|---|
| PyTorch | Tensor operations and neural modules | <https://github.com/pytorch/pytorch/blob/main/LICENSE> |
| NumPy | Numerical arrays and calibration | <https://github.com/numpy/numpy/blob/main/LICENSE.txt> |
| SciPy | The frozen rolling-statistics implementation uses `scipy.signal.lfilter` | <https://github.com/scipy/scipy/blob/main/LICENSE.txt> |
| pandas | CSV feature selection and prefix preprocessing for the spectral prior | <https://github.com/pandas-dev/pandas/blob/main/LICENSE> |

Their source distributions and licenses remain with their respective projects;
the source of these libraries is not copied into this upload. Version pins
record the research environment and are not a new cross-platform validation.

## Method attribution

The project's ProbSparse-style attention and causal TCN classes implement
ideas described by Informer and temporal convolutional networks. Citing those
ideas does not imply that the upstream implementation was copied or that its
license applies to all project code. Static review found no embedded upstream
copyright notice or vendored package in the selected project component bodies.

- Informer: Zhou et al., *Informer: Beyond Efficient Transformer for Long
  Sequence Time-Series Forecasting*, AAAI 2021.
  <https://doi.org/10.1609/aaai.v35i12.17325>
- TCN: Bai, Kolter and Koltun, *An Empirical Evaluation of Generic Convolutional
  and Recurrent Networks for Sequence Modeling*, 2018.
  <https://arxiv.org/abs/1803.01271>

## Excluded external implementations

The benchmark baselines and scoring metrics must be obtained from their own
sources under their own current licenses. This release does not infer a grant
of redistribution rights from a citation or a locally available checkout.

- TSB-AD data lists, datasets and metric framework:
  <https://github.com/TheDatumOrg/TSB-AD>
- PatchTST: <https://github.com/yuqinie98/PatchTST>
- GDN: <https://github.com/d-ailin/GDN>
- DLinear: <https://github.com/cure-lab/LTSF-Linear>
- PaAno paper: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/5991785875cf5eee9c29e2ed7d26d1c0-Abstract-Conference.html>

These links identify upstream work, not included source files or a verified
installation. Publication PDFs, publisher class files, and copies of third-
party repositories are not required runtime components of this package.
