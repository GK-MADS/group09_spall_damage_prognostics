import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))  # noqa: E402
from spall_hi import TOOTH, challenge_score, run_end_scores  # noqa: E402


def test_challenge_score_is_invariant_to_monotone_rescaling():
    truth = np.array([0, 0, 1, 3, 0, 2, 5.0])
    pred = np.array([0.1, 0.2, 0.5, 0.9, 0.0, 0.6, 1.0])
    a = challenge_score(pred, truth)
    b = challenge_score(np.exp(10 * pred), truth)
    assert a["mse_pooled"] == b["mse_pooled"] == 0.0


def test_challenge_score_pooled_penalises_cross_gear_scale_but_per_gear_does_not():
    truth = np.array([0, 1, 0, 1.0])
    pred = np.array([0, 1, 10, 11.0])  # gear 2's healthy value is above gear 1's damaged value
    s = challenge_score(pred, truth, np.array(["a", "a", "b", "b"]))
    assert s["mse_per_gear"] == 0.0 and s["mse_pooled"] > 0


def test_run_end_scores_tail_median_and_monotone_modes():
    rows = []
    for run, level in [(1, 0.0), (2, 5.0), (3, 1.0), (4, 6.0)]:
        for k in range(90):
            rows.append(dict(experiment="Exp-X", run_num=run, hours=run * 2 + k / 60, hi=level if k >= 30 else 99.0))
    s = pd.DataFrame(rows)
    ends = pd.DataFrame(dict(experiment="Exp-X", run_num=[1, 2, 3, 4, 5], end_h=[3.5, 5.5, 7.5, 9.5, 11.5]))
    none = run_end_scores(s, ends, ["hi"], "none").hi.tolist()
    assert none == [0.0, 5.0, 1.0, 6.0, 6.0]  # last 60 min only; run 5 has no data and carries forward
    assert run_end_scores(s, ends, ["hi"], "cummax").hi.tolist() == [0.0, 5.0, 5.0, 6.0, 6.0]
    iso = run_end_scores(s, ends, ["hi"], "iso").hi.to_numpy()
    assert np.all(np.diff(iso) >= 0) and iso[1] == iso[2] == 3.0  # spike at run 2 pooled with run 3


def test_tooth_features_are_a_subset_of_the_modelled_features():
    from spall_hi import FEATS

    assert set(TOOTH) <= set(FEATS)


def test_label_damage_is_mean_tooth_excess_per_run(tmp_path):
    from spall_hi import label_damage

    pd.DataFrame(
        dict(experiment="Exp-X", tooth=[1, 2, 1, 2], run_num=[1, 1, 2, 2], score=[1.0, 1.0, 1.0, 3.0])
    ).to_csv(tmp_path / "l.csv", index=False)
    ends = pd.DataFrame(dict(experiment="Exp-X", run_num=[2, 1, 3]))
    out = label_damage(ends, tmp_path / "l.csv")
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])  # follows ends' order; unlabelled run -> NaN


def test_load_labels_aggregates_teeth(tmp_path):
    from spall_supervised import load_labels

    rows = [dict(experiment="Exp-X", tooth=t, run_num=1, score=1.0) for t in range(1, 29)]
    rows += [dict(experiment="Exp-X", tooth=t, run_num=2, score=3.8 if t == 1 else 1.0) for t in range(1, 29)]
    pd.DataFrame(rows).to_csv(tmp_path / "l.csv", index=False)
    lab = load_labels(tmp_path / "l.csv").set_index("run_num")
    assert lab.loc[1, "mean_ex"] == 0 and lab.loc[1, "max"] == 1.0
    assert np.isclose(lab.loc[2, "mean_ex"], 0.1) and np.isclose(lab.loc[2, "top3"], 2.8 / 3)


def test_minute_targets_interpolate_between_inspections():
    from spall_supervised import minute_targets

    z = pd.DataFrame(dict(experiment="Exp-X", hours=[1.0, 6.0, 9.0, 12.0]))
    ends = pd.DataFrame(dict(experiment="Exp-X", run_num=[1, 2], end_h=[6.0, 12.0]))
    lab = pd.DataFrame(dict(experiment="Exp-X", run_num=[1, 2], mean_ex=[0.0, 0.4]))
    assert np.allclose(minute_targets(z, ends, lab, "mean_ex"), [0.0, 0.0, 0.2, 0.4])


def _gears(levels: dict[str, list[float]]) -> pd.DataFrame:
    """Per-minute HI rows: one constant level per run, 60 minutes each."""
    rows = []
    for gear, lv in levels.items():
        for run, v in enumerate(lv, start=1):
            rows += [dict(experiment=gear, run_num=run, hours=run * 6 - 1 + k / 60, hi=v) for k in range(60)]
    return pd.DataFrame(rows)


def test_label_free_scores_rank_a_clean_degradation_above_a_noisy_one():
    from spall_hi import label_free_table

    clean = _gears({"Exp-A": [0, 0, 1, 5], "Exp-B": [0, 1, 2, 5], "Exp-C": [0, 0, 2, 5]})
    noisy = _gears({"Exp-A": [3, 0, 4, 1], "Exp-B": [0, 4, 0, 9], "Exp-C": [2, 0, 3, 0]})
    s = clean.rename(columns={"hi": "clean"}).assign(noisy=noisy.hi.to_numpy())
    ends = s.groupby(["experiment", "run_num"]).hours.max().rename("end_h").reset_index()
    thr = pd.DataFrame({"clean": 0.5, "noisy": 0.5}, index=["Exp-A", "Exp-B", "Exp-C"])
    t = label_free_table(s, thr, ends, ["clean", "noisy"])
    assert t.loc["clean", "monotonicity"] == 1.0  # never falls in any gear
    assert t.loc["clean", "prognosability"] == 1.0  # every gear ends at 5
    assert t.loc["clean", "false_alarm"] == 0.0 and np.isclose(t.loc["noisy", "false_alarm"], 2 / 3, atol=1e-4)
    assert t.loc["clean", "trendability"] > t.loc["noisy", "trendability"]
    assert t.fitness.idxmax() == "clean"
