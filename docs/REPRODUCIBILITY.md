# Reproduction scope

Current manuscript data are distributed in the public `data-v1.0.0` release.
Its `PUBLIC_DATA_INDEX.json` separates the adopted studies and
`SOURCE_BINDINGS.json` records the original saved members and path-only
normalization. The repository's `results/` tree preserves the 3 October
snapshot; current manuscript values are read from the versioned data release.

This release provides research components and already computed results.
File-integrity checks can run without scientific packages. They establish that
the delivered files match the release manifest; they do not establish an
independent reproduction of the scientific results.

## Included

- Component-conditioned full, shared-selector and radius-zero predictor code.
- Corrected constant-channel normalization and dynamic-scoring components.
- Scientific configurations and implementation notes.
- Complete corrected summary statistics and per-file metric records for the
  retained-scoring and unified-training studies, with separate historical
  control and prediction-error records.
- Saved synthetic relation-recovery and fixed-candidate weight-intervention
  summaries, protocols and per-system/seed outcomes, separate from real-data
  detection results.
- Small frozen bounded, constant-mixture and unbounded scoring-model files,
  where listed in the artifact manifest.

## Not included

- Original benchmark data and the full collection of fitted predictors.
- Complete residual, prediction and anomaly-score trajectories.
- A frozen installation image or a portable reproduction of the historical
  distributed scheduling environment.
- A newly executed one-command pipeline from raw data to manuscript tables.

The numerical components were curated by static source inspection. No models
were loaded, no arrays were recomputed and no training or inference was run
during this release preparation. Installation and numerical behavior in a new
environment remain unverified. Use the supplied APIs and configurations with
that distinction in mind.

## Scientific distinctions retained

The proposed scoring gate uses the history of both calibrated residual
streams, official Tuning supervision, a learned constant anchor and bounded
corrections. Its parameters remain frozen for Evaluation and cross-predictor
transfer. No global router is part of the proposed method.

Shared relations is a common-selector design, not only parameter tying. It
changes the selector input and patch grid and broadcasts a shared probability
map while retaining component-specific value aggregation. Radius zero retains
fine weighting at selected coarse endpoints but removes neighboring offsets.

The retained and strict predictor cohorts remain separate. Nine corrected
historical distributed fits follow saved two-GPU configurations; incomplete
earlier execution logs prevent a claim of bitwise reproduction of their
original execution. The corrected experiment is a validation on a previously
accessed benchmark, not an untouched-source confirmation.

Privacy edits, if any, apply to path or execution-account strings in public
copies, not to results or scientific parameters. Source/output hashes and
redaction notes describe the derived files. Original archives are not changed.
