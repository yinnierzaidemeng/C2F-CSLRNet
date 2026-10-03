# Corrected saved results

Start with `index.json`. It points to five complete corrected real-data summaries and their saved per-file tables, plus the synthetic recovery and fixed-candidate weight-intervention studies. These are existing completed results, not new calculations and not a collection of only the repaired files.

## Study folders

| Folder | Saved study |
|---|---|
| `retained/scoring` | Scoring rules and frozen cross-backbone transfer on retained predictor fits |
| `retained/branches_native` | Exact A/B branches, mixture controls and aligned native PaAno comparison |
| `historical_control` | Earlier controlled predictor comparison |
| `strict/structure` | Unified-training Full, Shared and Radius 0 comparisons |
| `strict/prediction` | Prediction-error summaries and per-file records from unified training |
| `retrieval/recovery` | Component-source-lag recovery in 15 generated systems, including all per-run and per-system records |
| `retrieval/intervention` | Fixed-candidate local-weight permutation using 45 fitted models from those 15 systems |

Retained scoring and unified-training structure results use different predictor fits. Do not pool them or compute gains across the two cohorts. PaAno retains native offline inference and its distinct information and supervision conditions.

The complete summaries preserve the six metrics, F/C/X views, separate predictor/gate seed summaries, paired intervals and unfavorable comparisons. The CSVs retain their original bytes. The data domain is post-training targets without point adjustment. F is the official 180-file equal-weight result; C uses 164 distinct contents; X excludes exact Tuning overlaps, leaving 170 files and 162 contents. No evaluation data were used to select new parameters for this release.

`correction_effects.csv` records changes due to the approved repair; it is not an additional benchmark. Source-index JSON files bind saved artifacts to their original hashes. They are provenance records, not a list of raw data distributed here.

The retrieval studies are separate synthetic mechanism tests, not another real-data detection cohort. Their saved summaries retain every scenario, alternative structure, relation tolerance and adverse contrast. The intervention summary contains 45 per-seed records and 15 system records. The recovery summary contains its complete `per_run`, `per_instance`, `family_summaries` and `paired_contrasts` fields. Protocol and existing verification records accompany both. No metrics or confidence intervals were recomputed while preparing this repository.

## Verify files

From the repository root, run:

```text
python scripts/verify_artifacts.py
```

The standard-library checker compares physical file sizes and SHA256 hashes listed in `manifest.json`. It does not load checkpoints or arrays, train, infer, score, recompute metrics or bootstrap intervals, or access a network. Passing establishes byte integrity, not independent scientific reproduction. The manifest is the index root and is not self-hashed.

## Privacy and provenance

Private execution identities and absolute filesystem locations were replaced by symbolic paths in JSON string metadata. All JSON fields, list entries, numeric values and booleans were retained. `redaction_policy.json` lists the scope, and `manifest.json` records both source and public hashes. Public provenance excludes the authors' original private paths. Some source hashes bind arrays or detailed records that are not distributed in this compact release; they should not be interpreted as included files.

The release omits historical evidence ZIPs, raw benchmark data, full prediction/residual trajectories and metric/bootstrap NPZ arrays. Complete results remain available in the supplied JSON summaries and CSV tables. No separate open-data license is selected here; the repository's code license does not automatically relicense results or third-party data. See the repository licensing notes before redistribution.
