# Frozen scoring checkpoints

This directory contains nine small, original scoring checkpoints. It does not contain all trained predictor weights.

| Paths | Contents |
|---|---|
| `bounded/{42,43,44}/gate.pt` | Frozen history-conditioned bounded gates |
| `bounded/{42,43,44}/constant.pt` | Their learned constant anchors |
| `unbounded/{42,43,44}/gate.pt` | Matched unbounded control gates |

The seeds here are gate seeds, not predictor seeds. The bounded gates and anchors were fitted using the original 20-file C2F Tuning residuals and retained unchanged for Evaluation and transfer. Each matched unbounded control uses the same frozen anchor and the approved control protocol. See `results/index.json` for the study definitions and `complete.json` beside each seed for its saved training record.

Static serialization inspection identifies `state_dict`, `feature_mean`, `feature_std`, `seed` and settings in the gate files. `anchor_logit` is inside `state_dict`, not a separate top-level checkpoint field. The state-dictionary layers are `net.0` and `net.2`; the saved linear weight shapes are `(8, 12)` and `(1, 8)`. The constant files contain `bias` and `seed`. These facts come from pickle opcode metadata; the release preparation did not unpickle a model, decode feature arrays or execute inference.

All nine `.pt` files are byte-identical to their saved sources. ZIP members, raw marker bytes and pickle string operands were inspected for private paths and execution identities. JSON completion/freeze records retain their scientific fields and numerical values, with private filesystem strings replaced by symbolic paths. The source and release hashes are recorded in `results/manifest.json`.

Run `python scripts/verify_artifacts.py` from the repository root for a size-and-hash check without loading a checkpoint. Serialized model files should only be loaded in a trusted environment using the matching research implementation. Static metadata checks do not establish numerical equivalence or make a pickle intrinsically safe.

No separate open-model license is selected here. The license for original repository code is not a blanket license for checkpoints or third-party assets; consult the repository licensing notes and the author before redistribution.
