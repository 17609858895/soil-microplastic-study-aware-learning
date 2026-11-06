# -*- coding: utf-8 -*-
"""
08_robustness.py — robustness of the within-study (method-invariant) findings.

A. Wild cluster restricted bootstrap (WCR, Webb 6-point weights, B = 1999) for
   (i) each within-study FE effect (land use vs cropland, depth slope, greenhouse),
   (ii) the joint method-invariance test (land use x spectroscopic counting interactions), which has few
        informative clusters, so cluster-robust Wald p-values are unreliable.
B. Leave-one-study-out stability of the FE effects.
C. Within-study depth response: ALE of fixed-effect boosting, study-bootstrap band (B = 60), vs linear FE.
D. Non-detects (56 zero records): FE effects dropping zeros / substituting half a nominal detection limit;
   left-censored XGBoost AFT ("survival" likelihood) vs squared-error XGBoost under leave-study-out CV.
Outputs -> results/robustness/
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_level as sl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def load():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna()].reset_index(drop=True)
    return d


def demean(M, sid):
    M = np.asarray(M, float)
    df = pd.DataFrame(M)
    return M - df.groupby(sid).transform("mean").values


def crve(X, u, sid, XtXi):
    G = np.unique(sid)
    meat = np.zeros((X.shape[1], X.shape[1]))
    s = pd.Series(np.arange(len(sid))).groupby(sid).indices
    for g, idx in s.items():
        sg = X[idx].T @ u[idx]
        meat += np.outer(sg, sg)
    k = len(G) / (len(G) - 1)
    return k * XtXi @ meat @ XtXi


def wald(X, y, sid, R_idx):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    V = crve(X, u, sid, XtXi)
    Rb = b[R_idx]
    VR = V[np.ix_(R_idx, R_idx)]
    return float(Rb @ np.linalg.pinv(VR) @ Rb), b, V


def wcr(X, y, sid, R_idx, B=1999, seed=0):
    """Wild cluster restricted bootstrap p-value for H0: beta[R_idx] = 0."""
    W, b, V = wald(X, y, sid, R_idx)
    keep = [j for j in range(X.shape[1]) if j not in R_idx]
    Xr = X[:, keep]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, u_r = Xr @ br, y - Xr @ br
    groups, inv = np.unique(sid, return_inverse=True)
    rng = np.random.default_rng(seed)
    XtXi = np.linalg.pinv(X.T @ X)
    Ws = np.empty(B)
    for bi in range(B):
        w = rng.choice(WEBB, len(groups))[inv]
        ys = fit_r + u_r * w
        bs = XtXi @ X.T @ ys
        us = ys - X @ bs
        Vs = crve(X, us, sid, XtXi)
        Rb = bs[R_idx]
        Ws[bi] = Rb @ np.linalg.pinv(Vs[np.ix_(R_idx, R_idx)]) @ Rb
    return W, float((np.sum(Ws >= W) + 1) / (B + 1)), b, V


def fe_matrix(d, with_interactions=False):
    X = sl.within_design(d)
    cols = ["Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land", "log_depth", "LU_greenhouse", "LU_mulch"]
    X = X[cols].copy()
    if with_interactions:
        for lu in ["Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]:
            X[f"{lu}:spectro"] = X[lu] * d["spectro_quant"].values
    sid = d["study_id"].values
    Xd = demean(X.values, sid)
    ok = Xd.std(0) > 1e-10
    return Xd[:, ok], np.array(X.columns)[ok], sid


def part_a(d):
    y = demean(d["y"].values[:, None], d["study_id"].values).ravel()
    X, names, sid = fe_matrix(d)
    rows = []
    for j, nm in enumerate(names):
        W, p, b, V = wcr(X, y, sid, [j], B=1999, seed=j)
        rows.append(dict(term=nm, coef=b[j], se_crve=np.sqrt(V[j, j]), p_crve=float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(b[j] / np.sqrt(V[j, j]))))), p_wcr=p))
    A = pd.DataFrame(rows)
    Xi, names_i, sid = fe_matrix(d, with_interactions=True)
    R_idx = [j for j, n in enumerate(names_i) if ":spectro" in n]
    W, p_joint, b, V = wcr(Xi, y, sid, R_idx, B=1999, seed=99)
    inter = []
    for j in R_idx:
        Wj, pj, _, _ = wcr(Xi, y, sid, [j], B=1999, seed=100 + j)
        n_cl = int(pd.Series(Xi[:, j] != 0).groupby(sid).any().sum())
        inter.append(dict(term=names_i[j], coef=b[j], se_crve=np.sqrt(V[j, j]), p_wcr=pj, informative_studies=n_cl))
    return A, pd.DataFrame(inter), dict(joint_wald=W, joint_p_wcr=p_joint)


def part_b(d):
    y_all = d["y"].values
    sid_all = d["study_id"].values
    X, names, sid = fe_matrix(d)
    y = demean(y_all[:, None], sid).ravel()
    base = np.linalg.lstsq(X, y, rcond=None)[0]
    informative = [s for s in np.unique(sid) if np.abs(X[sid == s]).sum() > 0]
    rows = []
    for s in informative:
        m = sid != s
        b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        rows.append(dict(dropped_study=s, **dict(zip(names, b))))
    L = pd.DataFrame(rows)
    summ = pd.DataFrame({"full": base, "loso_min": L[names].min().values, "loso_max": L[names].max().values,
                         "same_sign_share": [(np.sign(L[n]) == np.sign(base[i])).mean() for i, n in enumerate(names)]},
                        index=names)
    return L, summ


def fe_boost_fit(d, n_backfit=6, seed=0):
    W = sl.within_design(d)
    y, sid = d["y"].values, d["study_id"].values
    u = pd.Series(y).groupby(sid).transform("mean").values
    m = None
    for _ in range(n_backfit):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=seed).fit(W, y - u)
        u = pd.Series(y - m.predict(W)).groupby(sid).transform("mean").values
    return m, W


def ale_1d(model, W, col, grid):
    vals = W[col].values
    eff = [0.0]
    for lo, hi in zip(grid[:-1], grid[1:]):
        idx = (vals >= lo) & (vals <= hi) if lo == grid[0] else (vals > lo) & (vals <= hi)
        if idx.sum() == 0:
            eff.append(eff[-1])
            continue
        A, Bm = W[idx].copy(), W[idx].copy()
        A[col], Bm[col] = lo, hi
        eff.append(eff[-1] + float(np.mean(model.predict(Bm) - model.predict(A))))
    eff = np.array(eff)
    counts = np.histogram(vals, bins=grid)[0]
    mids = (eff[:-1] + eff[1:]) / 2
    return eff - np.sum(mids * counts) / max(counts.sum(), 1)


def part_c(d, B=60):
    grid = np.log10(np.array([0, 5, 10, 15, 20, 30, 40, 60, 100]) + 1)
    m, W = fe_boost_fit(d)
    base = ale_1d(m, W, "log_depth", grid)
    studies = d["study_id"].unique()

    def one(b):
        rng = np.random.default_rng(b)
        pick = rng.choice(studies, len(studies), replace=True)
        parts = [d[d.study_id == s].assign(study_id=k) for k, s in enumerate(pick)]
        db = pd.concat(parts, ignore_index=True)
        mb, Wb = fe_boost_fit(db, seed=b)
        return ale_1d(mb, Wb, "log_depth", grid)
    boots = np.array(Parallel(n_jobs=8)(delayed(one)(b) for b in range(B)))
    depth_cm = 10 ** grid - 1
    C = pd.DataFrame({"depth_cm": depth_cm, "log_depth": grid, "ale": base, "lo": np.quantile(boots, 0.025, 0),
                      "hi": np.quantile(boots, 0.975, 0)})
    # land-use contrasts from FE-boosting (average predicted difference vs cropland)
    lu_rows = []
    for lu in ["Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]:
        A, Bm = W.copy(), W.copy()
        for c in ["Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]:
            A[c], Bm[c] = 0.0, 0.0
        Bm[lu] = 1.0
        lu_rows.append(dict(land_use=lu, fe_boost_contrast=float(np.mean(m.predict(Bm) - m.predict(A)))))
    return C, pd.DataFrame(lu_rows)


def part_d(d):
    rows = []
    c_lod = np.log10(1 + 10.0)  # nominal: 1 particle in 100 g dry soil
    for name, dd in [("all (zeros as 0)", d), ("drop non-detects", d[d.abund_corr > 0]),
                     ("non-detects = half nominal LOD", d.assign(y=np.where(d.abund_corr == 0, np.log10(1 + 5.0), d.y)))]:
        dd = dd.reset_index(drop=True)
        X, names, sid = fe_matrix(dd)
        y = demean(dd["y"].values[:, None], sid).ravel()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(variant=name, n=len(dd), **dict(zip(names, b))))
    F = pd.DataFrame(rows)
    # survival-likelihood (AFT, left-censored) vs squared error, leave-study-out
    feats = ["lat", "lon", "abs_lat", "depth_mid", "samp_year_filled", "LU_greenhouse", "LU_mulch", "LU_urban", "LU_traffic",
             "spectro_quant", "Q_LDIR", "Q_Raman", "Q_fluor", "ID_none", "max_density", "P_Fenton", "P_alkali", "log_size_min", "log_size_max"]
    X = d[feats].astype(float).join(pd.get_dummies(d["land_use"], prefix="LU").astype(float)).values
    y = d["y"].values
    cens = (d["abund_corr"] == 0).values
    # survival framing: "time" = abundance + 1 (items kg-1); non-detects are left-censored in [1, 10**c_lod]
    t_obs = 10 ** y
    lo = np.where(cens, 1.0, t_obs)
    hi = np.where(cens, 10 ** c_lod, t_obs)
    p_se, p_aft = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, d["study_id"].values):
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                             min_child_weight=5, n_jobs=4, verbosity=0).fit(X[tr], y[tr])
        p_se[te] = m.predict(X[te])
        dtr = xgb.DMatrix(X[tr])
        dtr.set_float_info("label_lower_bound", lo[tr])
        dtr.set_float_info("label_upper_bound", hi[tr])
        bst = xgb.train(dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                             aft_loss_distribution_scale=2.0, learning_rate=0.03, max_depth=6, subsample=0.8,
                             colsample_bytree=0.7, min_child_weight=5, nthread=4, verbosity=0), dtr, num_boost_round=600)
        p_aft[te] = np.log10(bst.predict(xgb.DMatrix(X[te])))  # predicted "time" -> log10(abundance + 1)
    det = ~cens
    S = dict(n_nondetect=int(cens.sum()), r2_detected_squared_error=float(r2_score(y[det], p_se[det])),
             r2_detected_aft=float(r2_score(y[det], p_aft[det])),
             mean_pred_nondetect_squared_error=float(p_se[cens].mean()), mean_pred_nondetect_aft=float(p_aft[cens].mean()))
    return F, S


def main():
    d = load()
    if "--part-d" in sys.argv[1:]:
        F, S = part_d(d)
        F.to_csv(OUT / "nondetect_fe_sensitivity.csv", index=False)
        print(F.round(3).to_string())
        print(S)
        (OUT / "nondetect_aft_summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
        return
    A, I, J = part_a(d)
    A.to_csv(OUT / "fe_wild_cluster_bootstrap.csv", index=False)
    I.to_csv(OUT / "method_invariance_interactions_wcr.csv", index=False)
    print(A.round(4).to_string())
    print(I.round(4).to_string())
    print(J)
    L, Ls = part_b(d)
    L.to_csv(OUT / "fe_leave_one_study_out.csv", index=False)
    Ls.to_csv(OUT / "fe_leave_one_study_out_summary.csv")
    print(Ls.round(3).to_string())
    C, LUc = part_c(d)
    C.to_csv(OUT / "within_depth_ale.csv", index=False)
    LUc.to_csv(OUT / "fe_boost_landuse_contrasts.csv", index=False)
    print(C.round(3).to_string())
    print(LUc.round(3).to_string())
    F, S = part_d(d)
    F.to_csv(OUT / "nondetect_fe_sensitivity.csv", index=False)
    print(F.round(3).to_string())
    print(S)
    (OUT / "robustness_summary.json").write_text(json.dumps(dict(method_invariance=J, nondetect_aft=S), indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
