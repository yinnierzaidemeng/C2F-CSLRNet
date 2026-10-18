# Public research data

Version 1.0.0 supplies saved research results, scientific configurations and
provenance for *Component-Conditioned Coarse-to-Fine Retrieval and Dynamic
Residual Scoring for Multivariate Time-Series Anomaly Detection*.

Download the data asset from the fixed
[data-v1.0.0 release](https://github.com/yinnierzaidemeng/C2F-CSLRNet/releases/tag/data-v1.0.0).
The asset is `C2F_public_research_data_20261018.zip`. Start with
`PUBLIC_DATA_INDEX.json` inside the extracted data directory. `DATA_MANIFEST.json`
checks supplied bytes; `SOURCE_BINDINGS.json` connects public files to their
original saved members. Execution-directory strings are normalized, while
saved numerical text and copied array bytes retain their original values.

The dataset covers retained-predictor scoring, unified-training structures,
synthetic recovery, fixed-candidate weight intervention and the five completed
native-baseline evaluations. Outcome files retain favorable and unfavorable
comparisons. Long unit-record paths use stable identifiers, with the original
relative names recorded in the source bindings.

Research data use [CC BY 4.0](../LICENSE_DATA.txt). The ZIP excludes raw TSB-AD
benchmark data, software repositories and model checkpoint collections.
Obtain TSB-AD from its [official source](https://github.com/TheDatumOrg/TSB-AD).
Original software is available in this repository under MIT. Separately supplied
scorer files retain their existing rights.

Use [CITATION_DATA.cff](CITATION_DATA.cff) to cite this data version. The
`DATA_RELEASE.json` record pins the asset name, byte size and SHA256.

```text
python scripts/verify_public_data.py /path/to/C2F_public_research_data_20261018.zip
```

The verifier uses the Python standard library and reads saved bytes; it runs
no model, scoring procedure or statistical recomputation.
