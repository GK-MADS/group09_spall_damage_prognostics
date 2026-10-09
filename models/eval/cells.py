"""Notebook cell sources for spall_eval.ipynb: one section per section of the PDF report. Reads out/ only."""

SETUP = r"""# Setup: read the evaluation outputs (regenerate with: python -m models.eval.run_all)
import json
from pathlib import Path
import pandas as pd
try:
    from IPython.display import Image, display
except ImportError:  # plain python (used by the headless check)
    def Image(filename, width=None): return filename
    display = print

OUT = Path("models/eval/out") if Path("models/eval/out").exists() else Path("out")
N = json.loads((OUT / "report_numbers.json").read_text())
pd.set_option("display.width", 200)
def fig(name): display(Image(filename=str(OUT / name), width=900))
def table(name, **kw): return pd.read_csv(OUT / name, **kw)
"""

CELLS = [
    (
        "md",
        "# Spall models: evaluation and failure analysis\n\nEvery number and figure here is read from "
        "`models/eval/out/`, the same files the PDF report uses. Run **Setup** first.",
    ),
    ("code", SETUP),
    (
        "md",
        "## 1. Evaluation method\n\nLeave-one-gear-out over Exp-A, B, C, F. T1 = photo damage (all 4 gears); "
        "T2 = tooth labels (`mean_ex`, Exp-A/B folds only). Gear error = squared residual under one pooled "
        "isotonic map, averaged within gear. p-values are exact sign-flip tests over gears; with 4 gears the "
        "smallest possible p is 1/16.",
    ),
    ("code", 'print("p floor, 4 gears:", N["p_floor_4"], "| 2 gears:", N["p_floor_2"])'),
    ("md", "## 2. Step 1.1 — Ablation"),
    (
        "code",
        'fig("fig_ablation_heatmap.png")\nprint(N["ablation_largest_t1_gain"]); print(N["ablation_largest_t1_loss"])\n'
        'table("ablation.csv").sort_values("d_t1")',
    ),
    ("md", "## 3. Step 1.2 — Hyperparameter sensitivity"),
    (
        "code",
        'fig("fig_sensitivity.png")\nprint("ridge alpha flat over:", N["ridge_alpha_flat_range"])\n'
        'print("Isolation Forest seed SD (T1):", N["iforest_seed_sd_t1"])\n'
        'print("distinct nested winners across folds:", N["tuned_distinct_winners"])\n'
        'table("sensitivity_tuned_winners.csv")',
    ),
    ("md", "## 4. Step 1.3 — Learning curves"),
    (
        "code",
        'fig("fig_learning_curves.png")\n'
        'print(N["curve_headline"], "error drop from 2 to 3 gears:", N["curve_gain_2_to_3_gears"])\n'
        'print("hi_tooth_maha:", N["curve_gain_2_to_3_gears_hi_tooth_maha"], "(hi_tooth fits no parameters: flat)")',
    ),
    ("md", "## 5. Step 2 — Comparison"),
    ("code", 'cmp = table("comparison.csv", index_col=0)\ncmp.round(4)'),
    (
        "code",
        'fig("fig_per_gear_dots.png"); fig("fig_trajectories.png"); fig("fig_lead_time.png")\n'
        'print("winner:", N["winner"], "| beats hours on", N["winner_k_of_4"], "of 4 gears, p =", '
        'N["winner_p_vs_hours"], "(floor", N["p_floor_4"], ")")\n'
        'print("selection bias, nested - non-nested T1 MSE:", N["selection_bias_gap_t1"])\n'
        'print("ranking changes on checked-only inspections:", N["ranking_changes_on_checked_only"])',
    ),
    ("md", "## 6. Failure examples"),
    (
        "code",
        'ex = table("failure_examples.csv")\nex[["approach","experiment","run_num","category","rank","confirmed",'
        '"why","top_exact_or_shap","top_lime","lime_agreement","seed_residual_sd"]]',
    ),
    (
        "code",
        "for i in range(1, len(ex) + 1):\n"
        '    if (OUT / f"fig_failure_{i}.png").exists():\n'
        '        fig(f"fig_failure_{i}.png")',
    ),
    ("md", "## 7. Failure candidates (full ranked list, for auditing the selection)"),
    (
        "code",
        'c = table("failure_candidates.csv").sort_values(["approach", "target", "rank"])\n'
        'c.groupby(["approach", "target"]).head(5)',
    ),
    ("md", "## 8. Regenerate everything (optional, needs the parquet tables; several minutes)"),
    ("code", "# import models.eval.run_all as r; r.main()"),
]
