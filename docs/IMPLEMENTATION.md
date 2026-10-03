# Research component implementation

This folder exposes readable components of the paper's final predictor and
bounded scorer. Version `0.1.0` identifies this prepared local source snapshot;
it does not assert that a package or repository has been publicly released.

The original component/function bodies were selected from hash-pinned research
sources. File orchestration, machine restrictions, private paths, metric
implementations, external baselines and the retired global scoring router are
excluded. The extraction was checked with Python AST parsing and source hashes.
**The assembled package has not been imported, installed, trained or evaluated
as part of release preparation, and no numerical equivalence test of the new
adapters has been run.** Existing paper results belong to the archived studies;
they are not new results produced by this extracted package.

## Files and API

| Module | Public entry point | Input and purpose |
|---|---|---|
| `c2f_cslrnet.model` | `C2FCSLRNet`, `build_predictor` | Assemble full, shared, or radius0 predictor; input `[batch, 64, channels]` |
| `c2f_cslrnet.components` | Frozen component classes | ProbSparse encoders, causal TCN, decomposition and retrieval bodies |
| `c2f_cslrnet.normalization` | `ExactConstantStandardizer.fit`, `.transform` | Fit a nonempty finite FP32 matrix of fitting observations |
| `c2f_cslrnet.spectral_prior` | `estimate_channel_spectral_reliability(csv_path, train_end)` | Estimate a per-channel spectral prior from the explicitly permitted prefix |
| `c2f_cslrnet.tail_memory` | `branch_arrays(reference, target)` | Return `reference_a, reference_b, a, b` from absolute standardized residuals |
| `c2f_cslrnet.history` | `history_features(a, b)`, `moments(units)` | Past-only 12-feature rows and equal-file fitting moments |
| `c2f_cslrnet.gate` | `BoundedHistoryGate`, `predict_gate` | Bounded 12-to-8-to-1 gate and frozen FP32 block inference |
| `c2f_cslrnet.scoring` | `score_residuals` | Thin composition returning branch scores, gate weights and mixed scores |

Import these names from their individual modules. The package initializer does
not import a numerical runtime. `requirements.txt` records the numerical
dependency versions from the frozen Python 3.11 environment. The archived PyTorch build
used CUDA 11.8; selecting an appropriate PyTorch distribution for another
machine is separate from the scientific configuration. Dependency installation
and compatibility have not been tested for this extracted package.

## Predictor and normalization

The copied component bodies preserve the original mathematics. The new
`C2FCSLRNet` adapter follows the active branch order of the frozen predictor,
using the original module names `decomposer`, `trend`, `periodic`, `residual`,
`component_gate`, `cross_channel`, `relation_head` and `head`. This preserves
the intended parameter-name layout; checkpoint compatibility of the assembled
predictor is not claimed as a tested result. Spectral reliability is a
nonpersistent buffer, so it must be supplied separately when constructing a
predictor. The adapter requires exactly one prior value per channel instead of
silently falling back when the supplied prior has the wrong length.

The window is 64 and feature width is 64. Trend and periodic branches each use
three ProbSparse layers, four heads and sampling factor five. Their feed-forward
width is 256. The residual TCN uses three blocks with two kernel-four causal
convolutions per block and dilations 1, 2 and 4 along the token axis. Patch
length/stride/token counts are 32/16/3, 24/12/5 and 16/8/7. Retrieval retains up
to eight non-self sources and two joint source-patch pairs, then uses four-head
fine attention over eight-coordinate histories with radius eight.

Shared relations retains the three content encoders and separate component
value/output encoders, but uses one selector on the sum of the component
signals and the 16/8 grid. All components use its undropped, head-averaged
probability map. This also changes selector inputs and grid, rather than only
tying parameters. Radius0 keeps the component-specific selectors and permits
only the zero refinement offset around each selected coarse endpoint. It still
has learned weights across endpoints and eight-coordinate value histories.

`ExactConstantStandardizer` preserves the historical FP32 reductions and
`1e-8` scale threshold for nonconstant channels. Exactly constant fitting
channels receive their common input value as mean and scale one. Supply only
the permitted fitting rows. The same statistics transform later observations;
the module does not select a split or clean labels automatically.

The spectral-prior function is copied with its original CSV interface. It drops
the `Label` column, keeps numeric features, forward-fills missing values and
uses the supplied prefix endpoint. It has no knowledge of a split: callers
must supply the cohort's permitted endpoint. Unified-training structural runs
use the fitting segment; the historical retained predictor cohort has its own
designated-prefix convention. These cohorts must not be conflated.

## Configuration examples and training scope

The three `configs/strict_*_smd060_seed42.json` files are corrected examples for
one named benchmark file, not universally reusable fitted statistics. The
source spectral-prior vector, seed, fit and validation target ranges, and
training settings are retained. The shared/radius0 switches are materialized
as in the frozen variant builder; its saved base configuration alone did not
contain these overrides. Private source paths and execution bindings are
omitted. A new dataset needs its own permitted-prefix prior and normalization.

The unified structural study used seeds 42-44, fit-only normalization and
spectral priors, validation main MSE selection, effective batch 512, physical
and validation batch 64 for fewer than 128 channels and 16 otherwise, Adam
learning rate 0.001, gradient clipping at norm one, at most 20 epochs and
patience five. Its objective is main prediction MSE plus 0.2 auxiliary
relation MSE. These are protocol descriptions, not an implemented trainer.
The original expanded-convolution training backend, paired initialization and
random streams, distributed trainers and checkpoint-selection machinery are
not part of this release. Calling `build_predictor` alone does not recreate
the original paired initialization protocol.

## Tail-memory branches and chronological gate

Residual inputs are absolute errors of the main prediction head in standardized
units, shaped `[time, channels]`. The caller supplies the designated-prefix
reference and the target residuals. The reference is not label-cleaned.
Archived function docstrings that say "normal residuals" describe the nominal
reference assumption; they do not instruct label-based filtering. Tail
calibration combines smoothed empirical survival with exponential excess
continuation; evidence is capped at 300. Both branches use the mean of the
largest `min(2, channels)` channel evidences. Branch A uses quantile 0.98 and
memory 65; B uses 0.90 and 129. Both receive scalar recalibration at 0.98.
The copied score-bank methods include their local score-control operations,
but its legacy `scores` dispatcher and global router are absent.

`history_features` applies `log1p`, computes preceding scores plus mean,
population standard deviation and disagreement histories over 17 and 65
points, and shifts every feature row by one timestamp. Concatenate reference
and target branch scores before forming features, then retain the target rows;
resetting history at the test boundary changes the documented method.
`score_residuals` implements this concatenation. Arrays must be nonempty,
finite and in chronological order. No label is an inference input.

`moments` accepts the original list-of-units format, each containing
`unit['arrays']['features']`. It uses at most 4096 evenly spaced rows per
fitting file, equal-file first/second moments, and a 0.001 standard-deviation
floor. The original inference function standardizes these raw features,
clips them to [-8, 8], and uses blocks of 8192 rows.

The bounded gate uses `sigmoid(anchor_logit + tanh(net(features)))`, with
`net` equal to Linear(12, 8), Tanh, Linear(8, 1). The output layer starts at
zero. The anchor is a buffer; the network has 113 trainable parameters. Saved
gate moments, network weights and anchor remain fixed at evaluation. The
published `bounded_gate_protocol.json` retains fitting settings and seeds.
The original fitting stage first learned the constant mixture initialized at
0.1, then trained the bounded correction with the anchor frozen, using
same-file anomalous/normal pairs and softplus ranking loss on the log1p mixed
scores (temperature 0.25). Its sampling/training loop is not exposed as a CLI.

## Saved gate checkpoint contract

The packaged bounded `gate.pt` files expose a `state_dict` containing
`anchor_logit`, `net.0.weight`, `net.0.bias`, `net.2.weight`, `net.2.bias`, plus
top-level `feature_mean`, `feature_std`, `seed` and `settings`. Static pickle
inspection confirms a scalar anchor, weights of shapes (8, 12) and (1, 8),
and biases of shapes (8,) and (1,). The constant checkpoint contains `bias`
and `seed`. Do not assume a top-level `anchor_logit` field; the anchor is in
the gate's state dictionary.

To use an already loaded trusted bounded checkpoint, construct
`BoundedHistoryGate()`, load its `state_dict` strictly, set evaluation mode,
and pass its `feature_mean` and `feature_std` to `predict_gate` or
`score_residuals`. This is an API contract, not a claim that loading or
inference was executed during this release preparation. No automatic
checkpoint loader is included. Historical `.pt` files contain Python/NumPy
serialization as well as tensors; the person loading them must explicitly
decide whether to trust the source. Unbounded-control files share tensor keys
but require the different unbounded forward rule and must not be loaded into
the bounded gate to represent that control.

## Verification boundary

Static checks cover source SHA-256 pins, extracted symbol AST identity,
Python syntax, JSON/TOML structure, dependency names, and absence of private
paths in the public files. The small predictor/scoring adapters are new glue;
their runtime behavior and exact equivalence need a separate authorized
numerical validation. There is no one-command raw-data-to-paper reproduction
claim, and no experiment was rerun to prepare this folder.
