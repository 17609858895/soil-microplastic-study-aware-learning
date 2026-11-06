# -*- coding: utf-8 -*-
"""
12_conformal_by_region.py — are PRISM prediction intervals for unseen studies calibrated outside the data-rich region?

Reproduces the leave-study-out folds of 05_prism_model.py (GroupKFold(5), same inner calibration split), keeps the
record-level predictions, and stratifies 90% interval coverage and width by region:
  China (china_box = 1), Asia outside China, outside Asia.
k = 0 : study-level split-conformal and Gaussian intervals (refit here);
k >= 1: Gaussian intervals after empirical-Bayes anchoring (read from anchored_predictions.csv).
Coverage is reported per record and per study (each study weighted equally).
Outputs -> results/prism/interval_coverage_by_region.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pm", ROOT / "scripts" / "05_prism_model.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)
_lgbm_orig = pm.lgbm


def _lgbm_single(n=400, leaves=15, mcs=30):
    m = _lgbm_orig(n, leaves, mcs)
    m.set_params(n_jobs=1)
    return m


pm.lgbm = _lgbm_single
OUT = ROOT / "results" / "prism"
Z90 = 1.645


def region_of(d):
    r = np.where(d["continent"] != "Asia", "Outside Asia", np.where(d["china_box"] == 1, "China", "Asia outside China"))
    return pd.Series(r, index=d.index)


def summarise(frame, k, kind):
    rows = []
    for reg, g in list(frame.groupby("region")) + [("All", frame)]:
        per_study = g.groupby("study")["covered"].mean()
        rows.append(dict(k=k, interval=kind, region=reg, n_records=len(g), n_studies=g["study"].nunique(),
                         coverage_records=g["covered"].mean(), coverage_studies=per_study.mean(),
                         coverage_studies_q10=per_study.quantile(0.10), mean_width=g["width"].mean()))
    return rows


def main():
    d = pm.load()
    y, sid = d["y"].values, d["study_id"].values
    reg = region_of(d)
    folds = list(GroupKFold(5).split(d, y, sid))
    recs = []
    for fi, (tr, te) in enumerate(folds):
        dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
        pr = pm.PRISM().fit(dtr)
        mu, var = pr.predict(dte)
        inner = list(GroupKFold(4).split(dtr, dtr["y"], dtr["study_id"]))[0]
        pin = pm.PRISM().fit(dtr.iloc[inner[0]].reset_index(drop=True))
        dcal = dtr.iloc[inner[1]].reset_index(drop=True)
        mcal, vcal = pin.predict(dcal)
        q90 = np.quantile(np.abs(dcal["y"].values - mcal) / np.sqrt(vcal), 0.9)
        sd = np.sqrt(var)
        recs.append(pd.DataFrame({"fold": fi, "study": dte["study_id"].values, "region": reg.iloc[te].values,
                                  "y": dte["y"].values, "mu": mu, "sd": sd, "q90": q90}))
        print(f"fold {fi}: q90 = {q90:.3f}", flush=True)
    R = pd.concat(recs, ignore_index=True)
    R.to_csv(OUT / "prism_oof_k0_records.csv", index=False)
    # k = 0 skill on the SAME studies that enter the k-anchor evaluation (studies with >= k + 2 records)
    from sklearn.metrics import r2_score
    n_rec = R.groupby("study")["y"].transform("size")
    same = [dict(k=k, n_studies=int(R.loc[n_rec >= k + 2, "study"].nunique()),
                 R2_k0_same_studies=r2_score(R.loc[n_rec >= k + 2, "y"], R.loc[n_rec >= k + 2, "mu"])) for k in [1, 3, 5]]
    pd.DataFrame(same).to_csv(OUT / "prism_k0_on_anchor_subsets.csv", index=False)
    print(pd.DataFrame(same).round(3).to_string())
    rows = []
    for kind, mult in [("conformal", R["q90"]), ("gaussian", Z90)]:
        F = R.assign(covered=np.abs(R["y"] - R["mu"]) <= mult * R["sd"], width=2 * mult * R["sd"])
        rows += summarise(F, 0, kind)
    A = pd.read_csv(OUT / "anchored_predictions.csv")
    st_reg = pd.Series(reg.values, index=d["study_id"].values).groupby(level=0).first()
    for k in [1, 3, 5]:
        a = A[A.k == k].copy()
        sd = np.sqrt(a["PRISM_var"])
        a = a.assign(region=a["study"].map(st_reg), covered=np.abs(a["y"] - a["PRISM"]) <= Z90 * sd, width=2 * Z90 * sd)
        rows += summarise(a, k, "gaussian_anchored")
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "interval_coverage_by_region.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
