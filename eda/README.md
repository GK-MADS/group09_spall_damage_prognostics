# EDA for the spall-damage models

Scripts that produce `out/data.json`, the data behind the EDA report page. Run them from the repo root, in this order:

```bash
uv pip install pandas pyarrow numpy scipy scikit-learn pillow   # in addition to the base requirements
python eda/scripts/photo_proxy.py            # dark-area score per tooth photo -> eda/out/photo_proxy.csv
python eda/scripts/photo_proxy2.py eda/out   # adds the texture score (tex) used in the report
python eda/scripts/build_min.py              # all_1s / all_1min / particles_all parquet (Exp-A Run-2 nulled: byte copy of Run-1)
python eda/scripts/compute.py                # summary stats, indicator metrics, novelty prototype -> eda/out/data.json
```

`tex` is an automated texture score: the share of high-gradient pixels on the tooth flank below the contact band.
It finds large spalls on gears A, B and F. It misses early flakes, and on Exp-C photos (two camera setups) it produces false positives.
Spall and initiation labels in `compute.py` (`EYE`) were checked by eye on the final two inspections only. They are not a full labelling pass.

## Colab notebook

`eda/spall_figures.ipynb` has one cell per figure and table in the report, all drawn with matplotlib from `out/data.json`.
Run the Setup cell first. It asks you to upload `data.json` (and `thumbs.json` for the photo cell) if it can't find them in `DATA_DIR`.
Rebuild the notebook with `python eda/scripts/build_notebook.py` (needs `nbformat`).

`eda/spall_figures_plotly.ipynb` is the Plotly version (hover tooltips, zoom). It does not need `data.json`: cells 1–4 read the parquet tables from `gs://siads696-phm2026/data` (reusing `lf`, `particles`, `channels` if they are already loaded) and rebuild every number in the report.
For the photo labels, copy `eda/out/photo_proxy.csv` to `gs://siads696-phm2026/eda/photo_proxy.csv`; without it the notebook scores the photos under `data/train/photos` (about 1 GB).
The cell sources live in `eda/scripts/colab_cells.py`; rebuild with `python eda/scripts/build_plotly_notebook.py`.
