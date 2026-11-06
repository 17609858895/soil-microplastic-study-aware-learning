# -*- coding: utf-8 -*-
"""
study_level.py — shared helpers for STUDY-LEVEL (Mundlak-style) estimation of protocol effects.

Why: a measurement protocol is chosen once per study, so its confounders are study characteristics
(land-use mix, depth design, location, period). A record-level design never sees the land-use MIX of a study,
so record-level DML cannot adjust for it (semi-synthetic v1: bias -0.15, coverage 0.55 for spectroscopy).

Pipeline
  1. within-study fixed-effect regression of y on land use, subtype and depth -> study offset alpha_i
     (the part of the study level not explained by its own environmental composition)
  2. study-level covariates: location, period, land-use/subtype shares, depth design, other protocol steps
  3. cross-fitted R-learner (partially linear) and AIPW (binary treatments), K folds x R repetitions,
     ensemble nuisances (random forest + ridge/logistic), optional precision weights 1/(tau2 + sigma2/n_i),
     influence-function standard errors (studies are the independent units).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LU_LEVELS = ["Cropland", "Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]
SUBTYPES = ["LU_greenhouse", "LU_mulch", "LU_paddy", "LU_orchard", "LU_vegetable", "LU_industrial",
            "LU_traffic", "LU_urban", "LU_irrig_sludge", "LU_natural"]


def within_design(d):
    X = pd.get_dummies(d["land_use"]).reindex(columns=LU_LEVELS[1:], fill_value=0).astype(float)
    X["log_depth"] = np.log10(d["depth_mid"].fillna(d["depth_mid"].median()) + 1)
    for c in SUBTYPES:
        X[c] = d[c].astype(float)
    return X


def study_offsets(d, ycol="y"):
    """Within-study FE regression -> (alpha_i Series, beta_W, sigma2_within)."""
    X = within_design(d)
    y = d[ycol].values.astype(float)
    sid = d["study_id"].values
    Xd = X - X.groupby(sid).transform("mean")
    yd = y - pd.Series(y).groupby(sid).transform("mean").values
    keep = Xd.std() > 0
    Xd = Xd.loc[:, keep]
    beta, *_ = np.linalg.lstsq(Xd.values, yd, rcond=None)
    bW = pd.Series(beta, index=Xd.columns)
    adj = y - X.loc[:, keep].values @ beta
    alpha = pd.Series(adj).groupby(sid).mean()
    resid = yd - Xd.values @ beta
    n_s = pd.Series(sid).value_counts()
    sigma2 = float(np.sum(resid ** 2) / (len(y) - len(n_s) - Xd.shape[1]))
    return alpha, bW, sigma2


PROT_COLS = ["spectro_quant", "Q_LDIR", "Q_Raman", "Q_muFTIR_img", "Q_fluor", "ID_none", "ID_ATR", "ID_muFTIR",
             "ID_Raman", "max_density", "P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_H2O2", "P_Fenton",
             "P_alkali", "P_acid", "P_oil", "P_nilered", "log_size_min", "log_size_max"]


def study_table(d, ycol="y"):
    """One row per study: offset, n, Mundlak aggregates, protocol (study-constant by design)."""
    alpha, bW, sigma2 = study_offsets(d, ycol)
    g = d.groupby("study_id")
    S = pd.DataFrame({"alpha": alpha, "n": g.size(), "ybar": g[ycol].mean()})
    lat, lon = np.deg2rad(g["lat"].mean()), np.deg2rad(g["lon"].mean())
    S["sx"], S["sy"], S["sz"] = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    S["abs_lat"] = g["lat"].mean().abs()
    S["year"] = g["samp_year_filled"].mean()
    S["china"] = g["china_box"].mean()
    S["log_n"] = np.log(S["n"])
    for lu in LU_LEVELS:
        S["share_" + lu.replace(" ", "")] = g["land_use"].apply(lambda s, lu=lu: np.mean(s == lu))
    for c in SUBTYPES:
        S["share_" + c] = g[c].mean()
    dep = np.log10(d["depth_mid"].fillna(d["depth_mid"].median()) + 1)
    S["depth_mean"] = dep.groupby(d["study_id"]).mean()
    S["depth_max"] = dep.groupby(d["study_id"]).max()
    S["multi_layer"] = (d.groupby("study_id")["depth_mid"].nunique() > 1).astype(float)
    for c in PROT_COLS:
        S[c] = g[c].mean()
    S["max_density"] = S["max_density"].fillna(S["max_density"].median())
    S["log_size_min"] = S["log_size_min"].fillna(np.log10(50))
    S["log_size_max"] = S["log_size_max"].fillna(np.log10(5000))
    S.attrs["sigma2"] = sigma2
    return S


ENV_S = ["sx", "sy", "sz", "abs_lat", "year", "china", "log_n", "depth_mean", "depth_max", "multi_layer"] + \
        ["share_" + lu.replace(" ", "") for lu in LU_LEVELS[1:]] + ["share_" + c for c in SUBTYPES]


def _reg():
    return [RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0),
            make_pipeline(StandardScaler(), Ridge(alpha=10.0))]


def _clf():
    return [RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0),
            make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))]


def _fit_pred_reg(Xa, ya, Xb, w=None):
    return np.mean([m.fit(Xa, ya, **({} if w is None else {m.steps[-1][0] + "__sample_weight": w} if hasattr(m, "steps") else {"sample_weight": w})).predict(Xb) for m in _reg()], axis=0)


def _fit_pred_clf(Xa, ta, Xb):
    return np.mean([m.fit(Xa, ta).predict_proba(Xb)[:, 1] for m in _clf()], axis=0)


def precision_weights(S, tau2):
    return 1.0 / (tau2 + S.attrs.get("sigma2", 0.12) / S["n"].values)


def rlearner(S, T, controls, outcome="alpha", weights=None, K=5, R=10, seed=0):
    """Study-level partially linear DML (R-learner); IF-based SE; median over R repetitions."""
    X = S[controls].values.astype(float)
    y = S[outcome].values.astype(float)
    t = S[T].values.astype(float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    ths, ses = [], []
    for r in range(R):
        yr, tr = np.zeros(len(y)), np.zeros(len(y))
        for a, b in KFold(K, shuffle=True, random_state=seed + r).split(X):
            yr[b] = y[b] - _fit_pred_reg(X[a], y[a], X[b])
            tr[b] = t[b] - _fit_pred_reg(X[a], t[a], X[b])
        J = np.mean(w * tr ** 2)
        th = np.mean(w * tr * yr) / J
        psi = w * tr * (yr - th * tr) / J
        ths.append(th)
        ses.append(np.sqrt(np.mean(psi ** 2) / len(y)))
    ths, ses = np.array(ths), np.array(ses)
    th = float(np.median(ths))
    se = float(np.sqrt(np.median(ses ** 2 + (ths - th) ** 2)))
    return th, se


def aipw(S, T, controls, outcome="alpha", K=5, R=10, clip=0.05, seed=0):
    """Study-level AIPW for a binary treatment (ATE over studies); IF-based SE."""
    X = S[controls].values.astype(float)
    y = S[outcome].values.astype(float)
    t = (S[T].values > 0.5).astype(int)
    ths, ses = [], []
    for r in range(R):
        mu1, mu0, e = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for a, b in KFold(K, shuffle=True, random_state=seed + r).split(X):
            Xa1 = np.column_stack([X[a], t[a]])
            mu1[b] = _fit_pred_reg(Xa1, y[a], np.column_stack([X[b], np.ones(len(b))]))
            mu0[b] = _fit_pred_reg(Xa1, y[a], np.column_stack([X[b], np.zeros(len(b))]))
            e[b] = _fit_pred_clf(X[a], t[a], X[b])
        e = np.clip(e, clip, 1 - clip)
        phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        ths.append(phi.mean())
        ses.append(phi.std(ddof=1) / np.sqrt(len(y)))
    ths, ses = np.array(ths), np.array(ses)
    th = float(np.median(ths))
    return th, float(np.sqrt(np.median(ses ** 2 + (ths - th) ** 2)))


def perm_test(S, T, controls, outcome="alpha", n_perm=1000, seed=0):
    """Study-level residual permutation test for the R-learner statistic (single cross-fit)."""
    X = S[controls].values.astype(float)
    y = S[outcome].values.astype(float)
    t = S[T].values.astype(float)
    yr, tr = np.zeros(len(y)), np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(X):
        yr[b] = y[b] - _fit_pred_reg(X[a], y[a], X[b])
        tr[b] = t[b] - _fit_pred_reg(X[a], t[a], X[b])
    th = np.sum(tr * yr) / np.sum(tr ** 2)
    rng = np.random.default_rng(seed)
    perm = np.array([np.sum(rng.permutation(tr) * yr) / np.sum(tr ** 2) for _ in range(n_perm)])
    return float(th), float((np.sum(np.abs(perm) >= abs(th)) + 1) / (n_perm + 1)), float(perm.std())
