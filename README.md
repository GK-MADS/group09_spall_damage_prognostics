# group09_spall_damage_prognostics
Our goal is to estimate the severity of gear tooth wear (spall damage) using only externally mounted vibration sensor signals

## Layout

| Path | What it is |
|------|-----------|
| `scripts/phm_data.py` | Downloads the PHM 2026 Data Challenge shares and un-nests the zip-in-zip archives into flat per-split folders with a `manifest.csv` |
| `scripts/flatten_hdf5.py` | Reads every low-frequency HDF5 file and writes `lf_1s.parquet` (one row per second) and `particles.parquet` (one row per file) per split |
| `docs/phm_data_pipeline_explained.pdf` | Plain-English walkthrough of the source data structure and the flattening pipeline |
| `data/channels.csv` | Units and processing attributes for every column in the parquet tables |
| `data/<split>/manifest.csv` | Provenance for every extracted file (kept in git; the files themselves are not) |

## Getting the data

The raw archives, extracted HDF5 files, photos, and parquet tables are several GB and are not stored in git.
Recreate them locally:

```bash
python -m venv .venv && .venv/bin/pip install h5py numpy pandas pyarrow
export PHM_TEST_SHARE_PASSWORD=<password from the challenge>   # only needed for the test split
.venv/bin/python scripts/phm_data.py small        # ~1.7 GB: LF, CI, photos, submission templates
.venv/bin/python scripts/flatten_hdf5.py          # writes data/<split>/lf_1s.parquet and particles.parquet
```

## Unsupervised damage model

`models/spall_hi.py` builds a health index (HI) without damage labels and writes challenge submissions to `models/out/`.

```bash
uv pip install --python .venv/bin/python scikit-learn scipy pytest shap lime matplotlib nbformat
.venv/bin/python models/spall_hi.py          # LOGO evaluation + models/out/{validation,test}_submission.csv
.venv/bin/python -m pytest tests -q
```

Each gear is referenced to its own healthy first hours (0.5–6 h) after regressing out temperature and torque.
Healthy-state models are fit on pooled training baselines only; the submitted HI is the upward departure of the
tooth-damage CIs (FM4, M6A, M8A, NA4), taken as the median of each run's last hour and made non-decreasing per
gear with isotonic regression. The by-eye photo labels are used only to compare candidates under an approximation
of the challenge metric (MSE after the best monotone rescaling).

### Using the Exp-A / Exp-B spall labels

`data/labels/spall_scores.csv` holds the per-tooth spall scores (1.0 = healthy) at each run end for Exp-A and Exp-B.
`models/spall_supervised.py` uses them in two ways and writes `models/out/labelled_*`:

```bash
PYTHONPATH=models .venv/bin/python models/spall_supervised.py
```

1. A non-negative ridge fit on the labels (compared A↔B leave-one-gear-out; written as the alternate `labelled_ridge_*` files).
2. Label-driven tuning of the unsupervised HI across 96 settings, scored on the A/B labels and the Exp-C/F photo proxy.
   The winner (FM4 upward departure, last 60 min of each run, running max) is the primary `labelled_*_submission.csv`.

## Evaluation and failure analysis

`models/eval/` evaluates every approach (health-index candidates, Isolation Forest, ridge, label-tuned HI nested and
not, elapsed hours, debris) with one leave-one-gear-out protocol and writes all figures, tables and quoted numbers
for the report to `models/eval/out/`. Design: `docs/superpowers/specs/2026-10-09-model-evaluation-design.md`.

```bash
.venv/bin/python -m models.eval.run_all          # regenerate models/eval/out/ (about 30 s)
.venv/bin/python -m models.eval.build_notebook   # rebuild models/eval/spall_eval.ipynb
.venv/bin/python -m pytest tests -m "not slow"   # fast tests
```
