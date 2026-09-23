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
