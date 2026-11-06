# -*- coding: utf-8 -*-
"""
04_semisynthetic_validation.py
Semi-synthetic benchmark with KNOWN ground truth, built on the real covariates, real study structure and
real protocol assignment of the global soil-microplastic dataset.

Data-generating process (log10 items kg-1):
  y_ij = f_env(land use, subtype, depth, geography)_ij          # environmental signal (known)
       + theta_spec * spectro_i + theta_dens * (density_i-1.2) + (1-alpha) * log10(size_min_i)   # protocol
       + u_i                                                     # laboratory offset, CORRELATED with land-use mix and protocol
       + e_ij
u_i is generated to mimic the observed confounding: studies of forests/grasslands and spectroscopic studies
carry systematically different offsets (as the ESSD preprint reports for LDIR-dominated natural-forest records).

Estimators compared
  Land-use contrast (Forest vs Cropland, Settlements vs Cropland) and depth slope:
    pooled OLS | pooled LightGBM partial dependence | random-intercept mixed model | within-study FE (proposed)
  Protocol effects (spectroscopic counting, size-limit slope):
    pooled LightGBM PDP | record-level DML with iid SE and random folds (as in published environmental DML)
    | record-level DML with group cross-fitting + cluster-robust SE
    | study-level (Mundlak) R-learner and AIPW with the study as the unit (proposed)
Scenarios: S1 'observed' and S2 'unobserved' confounding, 20 replicates each (fixed seeds).
Pass 1 (one_rep) runs the pooled, mixed, fixed-effect and record-level estimators; pass 2 (one_rep_study)
runs the study-level estimators on the same simulated data (identical seed and first random draw).
Metrics: bias, RMSE across replicates, 95% CI coverage.
Outputs -> results/semisynthetic/
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_level as sl  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "semisynthetic"
OUT.mkdir(parents=True, exist_ok=True)

TRUE = dict(LU_Forest=-0.30, LU_Grass=-0.20, LU_Settle=0.25, LU_Wet=-0.10, LU_Other=-0.45,
            greenhouse=0.35, mulch=0.25, depth=-0.60, spec=0.55, dens=0.40, alpha=1.80)
ENV = ["lat", "lon", "abs_lat", "depth_mid", "samp_year_filled", "LU_greenhouse", "LU_mulch", "LU_traffic", "LU_urban"]
PROT = ["spectro_quant", "max_density", "log_size_min", "log_size_max", "P_Fenton", "P_alkali", "Q_LDIR", "Q_Raman"]


def load():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna()].reset_index(drop=True)
    d["depth_mid"] = d["depth_mid"].fillna(d["depth_mid"].median())
    d["log_depth"] = np.log10(d["depth_mid"] + 1)
    d["max_density"] = d["max_density"].fillna(d.groupby("study_id")["max_density"].transform("median")).fillna(1.4)
    d["log_size_min"] = d["log_size_min"].fillna(np.log10(50))
    d["log_size_max"] = d["log_size_max"].fillna(np.log10(5000))
    d["LU"] = pd.Categorical(d["land_use"], categories=["Cropland", "Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"])
    return d


def simulate(d, rng, confound=1.0, scenario="observed"):
    lu = d["land_use"]
    f = (TRUE["LU_Forest"] * (lu == "Forest Land") + TRUE["LU_Grass"] * (lu == "Grassland") +
         TRUE["LU_Settle"] * (lu == "Settlements") + TRUE["LU_Wet"] * (lu == "Wetlands") +
         TRUE["LU_Other"] * (lu == "Other Land") + TRUE["greenhouse"] * d["LU_greenhouse"] +
         TRUE["mulch"] * d["LU_mulch"] + TRUE["depth"] * d["log_depth"] +
         0.25 * np.sin(np.deg2rad(d["lat"]) * 3) + 0.15 * np.cos(np.deg2rad(d["lon"])))
    prot = (TRUE["spec"] * d["spectro_quant"] + TRUE["dens"] * (d["max_density"] - 1.2) +
            (1 - TRUE["alpha"]) * d["log_size_min"])
    st = d.groupby("study_id").agg(nat=("land_use", lambda s: np.mean(s.isin(["Forest Land", "Grassland"]))),
                                   spec=("spectro_quant", "mean"), lat=("lat", "mean"),
                                   year=("samp_year_filled", "mean"))
    # S1 "observed": lab offsets depend on OBSERVED study-level covariates (land-use mix, latitude, year) that
    #    also predict protocol choice -> confounding is identifiable by adjustment.
    # S2 "unobserved": additionally, unmeasured lab practice co-varies with the counting instrument itself
    #    (+0.5 for spectroscopic labs) -> no estimator can separate it from the instrument effect (stress test).
    u_obs = 0.9 * st["nat"] + 0.3 * np.tanh((st["lat"] - 32) / 10) + 0.25 * (st["year"] - 2022) / 2
    u = confound * u_obs + (0.5 * st["spec"] if scenario == "unobserved" else 0.0) + rng.normal(0, 0.55, len(st))
    u = u - u.mean()
    y = f + prot + d["study_id"].map(u).values + rng.normal(0, 0.35, len(d))
    return y.values


def lgbm():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.07, num_leaves=15, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)


def design(d, cols):
    X = d[cols].astype(float).join(pd.get_dummies(d["land_use"], prefix="LUc").astype(float))
    return X


def pdp_contrast(model, X, col_on, col_off_list):
    Xa, Xb = X.copy(), X.copy()
    for c in col_off_list:
        Xa[c] = 0.0
        Xb[c] = 0.0
    Xa[col_on] = 1.0
    return float(np.mean(model.predict(Xa) - model.predict(Xb)))


def pdp_slope(model, X, col, delta):
    Xa, Xb = X.copy(), X.copy()
    Xa[col] = Xa[col] + delta
    return float(np.mean(model.predict(Xa) - model.predict(Xb)) / delta)


def dml(y, T, X, groups, clustered=True, k=5, seed=0):
    rng = np.random.default_rng(seed)
    if clustered:
        u = rng.permutation(np.unique(groups))
        fmap = {s: i % k for i, s in enumerate(u)}
        fold = np.array([fmap[s] for s in groups])
    else:
        fold = rng.integers(0, k, len(y))
    yr, tr_ = np.zeros(len(y)), np.zeros(len(y))
    for f in range(k):
        a, b = fold != f, fold == f
        yr[b] = y[b] - lgbm().fit(X[a], y[a]).predict(X[b])
        tr_[b] = T[b] - lgbm().fit(X[a], T[a]).predict(X[b])
    th = np.sum(tr_ * yr) / np.sum(tr_ ** 2)
    psi = tr_ * (yr - th * tr_)
    if clustered:
        cl = pd.Series(psi).groupby(groups).sum().values
        se = np.sqrt(np.sum(cl ** 2)) / np.sum(tr_ ** 2)
    else:
        se = np.sqrt(np.sum(psi ** 2)) / np.sum(tr_ ** 2)
    return th, se


def one_rep(d, rep, confound, scenario="observed"):
    rng = np.random.default_rng(1000 + rep)
    dd = d.copy()
    dd["ys"] = simulate(d, rng, confound, scenario)
    sid = dd["study_id"].values
    rows = []
    # ---------- land-use / depth estimators ----------
    m = smf.ols("ys ~ C(LU) + log_depth + LU_greenhouse + LU_mulch", data=dd).fit(cov_type="cluster", cov_kwds={"groups": sid})
    rows += [dict(est="Pooled OLS", target="Forest vs Cropland", val=m.params["C(LU)[T.Forest Land]"], se=m.bse["C(LU)[T.Forest Land]"]),
             dict(est="Pooled OLS", target="Settlements vs Cropland", val=m.params["C(LU)[T.Settlements]"], se=m.bse["C(LU)[T.Settlements]"]),
             dict(est="Pooled OLS", target="Depth slope", val=m.params["log_depth"], se=m.bse["log_depth"])]
    fe = smf.ols("ys ~ C(LU) + log_depth + LU_greenhouse + LU_mulch + C(study_id)", data=dd).fit(cov_type="cluster", cov_kwds={"groups": sid})
    rows += [dict(est="Within-study FE (proposed)", target="Forest vs Cropland", val=fe.params["C(LU)[T.Forest Land]"], se=fe.bse["C(LU)[T.Forest Land]"]),
             dict(est="Within-study FE (proposed)", target="Settlements vs Cropland", val=fe.params["C(LU)[T.Settlements]"], se=fe.bse["C(LU)[T.Settlements]"]),
             dict(est="Within-study FE (proposed)", target="Depth slope", val=fe.params["log_depth"], se=fe.bse["log_depth"])]
    try:
        me = smf.mixedlm("ys ~ C(LU) + log_depth + LU_greenhouse + LU_mulch + spectro_quant + max_density + log_size_min",
                         data=dd, groups=dd["study_id"]).fit(reml=True, method="lbfgs")
        rows += [dict(est="Random-intercept mixed model", target="Forest vs Cropland", val=me.params["C(LU)[T.Forest Land]"], se=me.bse["C(LU)[T.Forest Land]"]),
                 dict(est="Random-intercept mixed model", target="Settlements vs Cropland", val=me.params["C(LU)[T.Settlements]"], se=me.bse["C(LU)[T.Settlements]"]),
                 dict(est="Random-intercept mixed model", target="Depth slope", val=me.params["log_depth"], se=me.bse["log_depth"]),
                 dict(est="Random-intercept mixed model", target="Spectroscopic counting", val=me.params["spectro_quant"], se=me.bse["spectro_quant"]),
                 dict(est="Random-intercept mixed model", target="Size-limit slope (1-alpha)", val=me.params["log_size_min"], se=me.bse["log_size_min"])]
    except Exception:
        pass
    X = design(dd, ENV + PROT + ["log_depth"])
    gb = lgbm().fit(X, dd["ys"])
    lucols = [c for c in X.columns if c.startswith("LUc_")]
    rows += [dict(est="Pooled LightGBM (PDP)", target="Forest vs Cropland", val=pdp_contrast(gb, X, "LUc_Forest Land", lucols) - pdp_contrast(gb, X, "LUc_Cropland", lucols), se=np.nan),
             dict(est="Pooled LightGBM (PDP)", target="Settlements vs Cropland", val=pdp_contrast(gb, X, "LUc_Settlements", lucols) - pdp_contrast(gb, X, "LUc_Cropland", lucols), se=np.nan),
             dict(est="Pooled LightGBM (PDP)", target="Depth slope", val=pdp_slope(gb, X, "log_depth", 0.3), se=np.nan),
             dict(est="Pooled LightGBM (PDP)", target="Spectroscopic counting", val=pdp_contrast(gb, X, "spectro_quant", []), se=np.nan),
             dict(est="Pooled LightGBM (PDP)", target="Size-limit slope (1-alpha)", val=pdp_slope(gb, X, "log_size_min", 0.3), se=np.nan)]
    # ---------- protocol DML ----------
    for T, tname in [("spectro_quant", "Spectroscopic counting"), ("log_size_min", "Size-limit slope (1-alpha)")]:
        ctrl = [c for c in ENV + PROT + ["log_depth"] if c != T]
        if T == "spectro_quant":
            ctrl = [c for c in ctrl if c not in ("Q_LDIR", "Q_Raman")]
        Xc = design(dd, ctrl).values
        yv, tv = dd["ys"].values, dd[T].values.astype(float)
        th, se = dml(yv, tv, Xc, sid, clustered=False, seed=rep)
        rows.append(dict(est="Record-level DML (iid SE)", target=tname, val=th, se=se))
        th, se = dml(yv, tv, Xc, sid, clustered=True, seed=rep)
        rows.append(dict(est="Record-level DML (cluster SE)", target=tname, val=th, se=se))
    for r in rows:
        r["rep"] = rep
        r["confound"] = confound
        r["scenario"] = scenario
    return rows


def one_rep_study(d, rep, scenario):
    """Only the study-level estimators, on the SAME simulated data as one_rep (identical seed and first rng draw)."""
    rng = np.random.default_rng(1000 + rep)
    dd = d.copy()
    dd["ys"] = simulate(d, rng, 1.0, scenario)
    S = sl.study_table(dd.assign(y=dd["ys"]), "y")
    prot_ctrl = ["max_density", "P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_H2O2", "P_Fenton", "P_alkali", "P_acid", "P_oil"]
    c_spec = sl.ENV_S + prot_ctrl + ["log_size_min", "log_size_max"]
    c_size = sl.ENV_S + ["Q_LDIR", "Q_Raman", "Q_muFTIR_img", "Q_fluor", "spectro_quant", "ID_none"] + prot_ctrl + ["log_size_max"]
    rows = []
    th, se = sl.rlearner(S, "spectro_quant", c_spec, R=3, seed=rep)
    rows.append(dict(est="Study-level R-learner (proposed)", target="Spectroscopic counting", val=th, se=se))
    th, se = sl.aipw(S, "spectro_quant", c_spec, R=3, seed=rep)
    rows.append(dict(est="Study-level AIPW (proposed)", target="Spectroscopic counting", val=th, se=se))
    th, se = sl.rlearner(S, "log_size_min", c_size, R=3, seed=rep)
    rows.append(dict(est="Study-level R-learner (proposed)", target="Size-limit slope (1-alpha)", val=th, se=se))
    for r in rows:
        r.update(rep=rep, confound=1.0, scenario=scenario)
    return rows


def summarise(df):
    truth = {"Forest vs Cropland": TRUE["LU_Forest"], "Settlements vs Cropland": TRUE["LU_Settle"],
             "Depth slope": TRUE["depth"], "Spectroscopic counting": TRUE["spec"],
             "Size-limit slope (1-alpha)": 1 - TRUE["alpha"]}
    df["truth"] = df["target"].map(truth)
    df["err"] = df["val"] - df["truth"]
    df["covered"] = np.where(df["se"].notna(), (np.abs(df["err"]) <= 1.96 * df["se"]).astype(float), np.nan)
    df.to_csv(OUT / "semisynthetic_replicates.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby(["scenario", "target", "est"]).agg(truth=("truth", "first"), mean_est=("val", "mean"), bias=("err", "mean"),
                                                         rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                                         coverage95=("covered", "mean"), n=("val", "size")).reset_index()
    summ.to_csv(OUT / "semisynthetic_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string())


def main(n_rep=20):
    d = load()
    scen = ["observed", "unobserved"]
    core = Parallel(n_jobs=12, verbose=5)(delayed(one_rep)(d, r, 1.0, sc) for sc in scen for r in range(n_rep))
    study = Parallel(n_jobs=10, verbose=5)(delayed(one_rep_study)(d, r, sc) for sc in scen for r in range(n_rep))
    summarise(pd.DataFrame([x for rr in core + study for x in rr]))


if __name__ == "__main__":
    main()
