# -*- coding: utf-8 -*-
"""
11_shap_fingerprint.py — does SHAP "driver importance" measure effects or study identity?

Model A : pooled LightGBM on the real features (the usual random-split workflow).
Model B : the same features PLUS 8 random numbers that are constant within each study (a meaningless study code).
If protocol variables are important only because they fingerprint studies, their SHAP share should collapse in
Model B, with the importance moving to the random code. Repeated over 5 random codes.
Also reports each feature's between-study variance share (study-identifying power) vs its SHAP share.
Outputs -> results/benchmark/shap_fingerprint_*.csv
"""
import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bm", ROOT / "scripts" / "02_benchmark_20models.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)
OUT = ROOT / "results" / "benchmark"
G = json.loads((ROOT / "data" / "processed" / "feature_groups.json").read_text(encoding="utf-8"))


def group_of(c):
    if c.startswith("code"):
        return "Random study code"
    if c in G["M_protocol"]:
        return "Measurement protocol"
    if c in ("lat", "lon", "abs_lat", "china_box", "samp_year_filled") or c.startswith("CONT_"):
        return "Location & period"
    if c in G["E_soil_sparse"]:
        return "Soil properties"
    return "Land use & depth"


def fit_shap(X, y, seed):
    m = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=8, random_state=seed, verbose=-1).fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    return pd.Series(np.abs(sv).mean(0), index=X.columns)


def main():
    d, X, y, sid, site = bm.load()
    imp_a = fit_shap(X, y, 0)
    rows = []
    for rep in range(5):
        rng = np.random.default_rng(100 + rep)
        codes = rng.normal(size=(sid.max() + 1, 8))
        XB = X.join(pd.DataFrame(codes[sid], columns=[f"code{i}" for i in range(8)]))
        imp_b = fit_shap(XB, y, rep)
        for model, imp in [("A: real features", imp_a), ("B: real features + random study code", imp_b)]:
            share = (imp / imp.sum()).groupby(imp.index.map(group_of)).sum()
            for g, v in share.items():
                rows.append(dict(rep=rep, model=model, group=g, shap_share=v))
    S = pd.DataFrame(rows).groupby(["model", "group"])["shap_share"].agg(["mean", "std"]).reset_index()
    S.to_csv(OUT / "shap_fingerprint_group_shares.csv", index=False)
    # feature-level: SHAP share in model A vs between-study variance share of the feature
    dm = pd.DataFrame(X.values, columns=X.columns).assign(sid=sid)
    icc = {}
    for c in X.columns:
        v = dm[[c, "sid"]].dropna()
        tot = v[c].var()
        icc[c] = v.groupby("sid")[c].transform("mean").var() / tot if tot > 0 else np.nan
    F = pd.DataFrame({"shap_share_A": imp_a / imp_a.sum(), "between_study_share": pd.Series(icc)})
    F["group"] = F.index.map(group_of)
    F.sort_values("shap_share_A", ascending=False).to_csv(OUT / "shap_fingerprint_features.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(S.round(3).to_string())
    print(F.sort_values("shap_share_A", ascending=False).head(15).round(3).to_string())
    top = F.sort_values("shap_share_A", ascending=False)
    w = top["shap_share_A"]
    print("SHAP-weighted mean between-study share of features:", round(float(np.nansum(w * top["between_study_share"]) / w.sum()), 3))


if __name__ == "__main__":
    main()
