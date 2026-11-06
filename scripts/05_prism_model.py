# -*- coding: utf-8 -*-
"""
05_prism_model.py  —  PRISM: Protocol-Resolved, Identification-aware Soil-Microplastic model

    y_ij = f_W(w_ij) + beta' m_i + g_B(e_i) + u_i + e_ij
      f_W : within-study environmental response (land use, subtype, depth), fixed-effect boosting
            (study offsets are free parameters -> f_W is identified only from within-study contrasts)
      beta: protocol effects, orthogonalised at study level (Robinson partialling-out against study environment)
      g_B : regional field of lab-corrected study offsets (Gaussian process on study centroids + year)
      u_i : residual laboratory offset ~ N(0, tau^2), empirical-Bayes update from k anchor samples
      e_ij: within-study noise ~ N(0, sigma^2)

Comparators (identical folds): pooled LightGBM, pooled LightGBM + anchor shift, MERF-style random-intercept
boosting (RE-GBM), anchor-mean only.
Validation: leave-study-out GroupKFold(5) [same folds as 02_benchmark], k = 0,1,3,5 anchors,
Gaussian predictive intervals + study-level split-conformal intervals, within-study R2.
Outputs -> results/prism/
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prism"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026

W_COLS = ["log_depth", "LU_greenhouse", "LU_mulch", "LU_paddy", "LU_orchard", "LU_vegetable", "LU_industrial",
          "LU_traffic", "LU_urban", "LU_irrig_sludge", "LU_natural"]
M_COLS = ["spectro_quant", "Q_LDIR", "Q_Raman", "Q_fluor", "ID_none", "max_density", "P_Fenton", "P_alkali",
          "P_acid", "P_oil", "log_size_min", "log_size_max"]
E_STUDY = ["lat", "lon", "abs_lat", "samp_year_filled", "china_box"]
LU_LEVELS = ["Cropland", "Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]


def load():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna()].reset_index(drop=True)
    d["log_depth"] = np.log10(d["depth_mid"].fillna(d["depth_mid"].median()) + 1)
    d["max_density"] = d["max_density"].fillna(d.groupby("study_id")["max_density"].transform("median")).fillna(1.4)
    d["log_size_min"] = d["log_size_min"].fillna(np.log10(50))
    d["log_size_max"] = d["log_size_max"].fillna(np.log10(5000))
    for lu in LU_LEVELS[1:]:
        d["LUd_" + lu.replace(" ", "")] = (d["land_use"] == lu).astype(float)
    return d


def lgbm(n=400, leaves=15, mcs=30):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=mcs,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=6)


def wcols():
    return W_COLS + ["LUd_" + lu.replace(" ", "") for lu in LU_LEVELS[1:]]


def pooled_cols():
    return wcols() + M_COLS + E_STUDY + ["Q_muFTIR_img", "P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_H2O2"]


# ------------------------------------------------------------------ PRISM
class PRISM:
    def __init__(self, n_backfit=6):
        self.n_backfit = n_backfit

    def fit(self, d):
        y, sid = d["y"].values, d["study_id"].values
        W = d[wcols()].values
        u = pd.Series(y).groupby(sid).transform("mean").values
        for _ in range(self.n_backfit):
            self.fw = lgbm(300, 7, 40).fit(W, y - u)
            fw = self.fw.predict(W)
            u = pd.Series(y - fw).groupby(sid).transform("mean").values
        self.fw_center = float(np.mean(fw))
        fw = fw - self.fw_center
        res_w = y - fw
        st = d.assign(o=res_w).groupby("study_id").agg(o=("o", "mean"), n=("o", "size"),
                                                       **{c: (c, "mean") for c in M_COLS + E_STUDY})
        self.sigma2 = float(np.var(res_w - pd.Series(res_w).groupby(sid).transform("mean").values) * len(y) / (len(y) - len(st)))
        # Robinson partialling-out at study level (cross-fitted)
        E = st[E_STUDY].values
        M = st[M_COLS].values
        o = st["o"].values
        kf = KFold(5, shuffle=True, random_state=SEED)
        o_r, M_r = np.zeros_like(o), np.zeros_like(M)
        for a, b in kf.split(E):
            o_r[b] = o[b] - lgbm(200, 4, 8).fit(E[a], o[a]).predict(E[b])
            for j in range(M.shape[1]):
                M_r[b, j] = M[b, j] - lgbm(200, 4, 8).fit(E[a], M[a, j]).predict(E[b])
        wts = st["n"].values / (st["n"].values + 10.0)  # shrink very small studies
        lr = LinearRegression(fit_intercept=False).fit(M_r, o_r, sample_weight=wts)
        self.beta = pd.Series(lr.coef_, index=M_COLS)
        # regional field on protocol-corrected offsets
        o_c = o - M @ self.beta.values
        self.o_mean = float(np.average(o_c, weights=wts))
        Z = self._zstudy(st)
        k = C(0.3) * RBF(length_scale=[0.5, 0.5, 0.5, 0.3]) + WhiteKernel(0.3)
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=1, random_state=SEED)
        self.gp.fit(Z, o_c - self.o_mean)
        # tau2 from OUT-OF-SAMPLE GP residuals (in-sample GP nearly interpolates study offsets)
        resid = np.zeros_like(o_c)
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(Z):
            g = GaussianProcessRegressor(kernel=self.gp.kernel_, optimizer=None).fit(Z[a], o_c[a] - self.o_mean)
            resid[b] = o_c[b] - self.o_mean - g.predict(Z[b])
        self.tau2 = float(max(np.average(resid ** 2, weights=wts) - self.sigma2 * np.mean(1 / st["n"].values), 0.02))
        self.gp_cv_resid_var = float(np.average(resid ** 2, weights=wts))
        # cross-fitted regional field per training study (used for an honest variance decomposition)
        self.cv_regional = pd.Series(o_c - resid, index=st.index)
        self.m_ref = None
        return self

    @staticmethod
    def _zstudy(st):
        lat, lon = np.deg2rad(st["lat"].values), np.deg2rad(st["lon"].values)
        # unit-sphere coordinates (no dateline artefacts) + scaled sampling year
        return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat),
                                (st["samp_year_filled"].values - 2020) / 5.0])

    def components(self, d, m_override=None):
        fw = self.fw.predict(d[wcols()].values) - self.fw_center
        M = d[M_COLS].copy()
        if m_override is not None:
            for k, v in m_override.items():
                M[k] = v
        prot = M.values @ self.beta.values
        stq = d.groupby("study_id")[E_STUDY].transform("mean")
        gb, gsd = self.gp.predict(self._zstudy(stq), return_std=True)
        return fw, prot, gb + self.o_mean, gsd

    def predict(self, d, m_override=None):
        fw, prot, gb, gsd = self.components(d, m_override)
        mu = fw + prot + gb
        # tau2 is estimated from out-of-sample GP residuals, so it already contains GP prediction error
        var = self.sigma2 + self.tau2
        return mu, var


# ------------------------------------------------------------------ RE-GBM (MERF-style)
class REGBM:
    def fit(self, d, cols, n_iter=6):
        y, sid = d["y"].values, d["study_id"].values
        X = d[cols].values
        b = np.zeros(len(y))
        sig2, tau2 = 0.2, 0.5
        for _ in range(n_iter):
            self.m = lgbm().fit(X, y - b)
            r = y - self.m.predict(X)
            g = pd.DataFrame({"r": r, "s": sid}).groupby("s")["r"].agg(["mean", "size"])
            shrink = tau2 / (tau2 + sig2 / g["size"])
            bi = shrink * g["mean"]
            b = pd.Series(sid).map(bi).values
            sig2 = float(np.var(r - b))
            tau2 = float(np.var(bi) + np.mean(shrink * sig2 / g["size"]))
        self.sig2, self.tau2, self.cols = sig2, tau2, cols
        return self

    def predict(self, d):
        return self.m.predict(d[self.cols].values)


# ------------------------------------------------------------------ evaluation
def anchored(mu, var_total, y, idx_eval, idx_anchor, tau2, sigma2, eb=True):
    r = y[idx_anchor] - mu[idx_anchor]
    k = len(idx_anchor)
    w = (k * tau2 / (k * tau2 + sigma2)) if eb else 1.0
    shift = w * r.mean()
    post_tau2 = tau2 * sigma2 / (k * tau2 + sigma2) if eb else sigma2 / k
    return mu[idx_eval] + shift, sigma2 + post_tau2


def main():
    import sys
    if "full" in sys.argv[1:]:
        return full_fit(load())
    d = load()
    y, sid = d["y"].values, d["study_id"].values
    folds = list(GroupKFold(5).split(d, y, sid))
    rng = np.random.default_rng(SEED)
    rows, cover = [], []
    oof = {k: np.zeros(len(y)) for k in ["PRISM", "PRISM_harmonised", "Pooled_LGBM", "RE_GBM"]}
    oof_var = np.zeros(len(y))
    betas, taus = [], []
    cal_q = {}
    for fi, (tr, te) in enumerate(folds):
        dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
        pr = PRISM().fit(dtr)
        betas.append(pr.beta.rename(f"fold{fi}"))
        taus.append(dict(fold=fi, tau2=pr.tau2, sigma2=pr.sigma2))
        mu, var = pr.predict(dte)
        oof["PRISM"][te], oof_var[te] = mu, var
        pl = lgbm(600, 31, 20).fit(dtr[pooled_cols()].values, dtr["y"].values)
        oof["Pooled_LGBM"][te] = pl.predict(dte[pooled_cols()].values)
        re = REGBM().fit(dtr, pooled_cols())
        oof["RE_GBM"][te] = re.predict(dte)
        # study-level split-conformal: calibrate |residual| on an inner leave-study-out of the training fold
        inner = list(GroupKFold(4).split(dtr, dtr["y"], dtr["study_id"]))[0]
        pin = PRISM().fit(dtr.iloc[inner[0]].reset_index(drop=True))
        dcal = dtr.iloc[inner[1]].reset_index(drop=True)
        mcal, vcal = pin.predict(dcal)
        sc = np.abs(dcal["y"].values - mcal) / np.sqrt(vcal)
        q90 = np.quantile(sc, 0.9)
        cover.append(dict(fold=fi, q90_normalised=q90,
                          gauss90=float(np.mean(np.abs(dte["y"].values - mu) <= 1.645 * np.sqrt(var))),
                          conformal90=float(np.mean(np.abs(dte["y"].values - mu) <= q90 * np.sqrt(var))),
                          mean_width_conformal=float(np.mean(2 * q90 * np.sqrt(var)))))
        # anchored transfer
        te_local = pd.DataFrame({"i": np.arange(len(te)), "s": dte["study_id"].values})
        mu_pl = oof["Pooled_LGBM"][te]
        mu_re = oof["RE_GBM"][te]
        yt = dte["y"].values
        for s, grp in te_local.groupby("s"):
            idx = grp["i"].values
            for k in [1, 3, 5]:
                if len(idx) < k + 2:
                    continue
                for rep in range(5):
                    a = rng.choice(idx, k, replace=False)
                    e = np.setdiff1d(idx, a)
                    p_pr, v_pr = anchored(mu, var, yt, e, a, pr.tau2 + 0.0, pr.sigma2, eb=True)
                    p_pl, _ = anchored(mu_pl, None, yt, e, a, 1, 1, eb=False)
                    p_re, _ = anchored(mu_re, None, yt, e, a, re.tau2, re.sig2, eb=True)
                    rows.append(pd.DataFrame({"k": k, "fold": fi, "study": s, "y": yt[e], "PRISM": p_pr,
                                              "PRISM_var": v_pr, "Pooled_LGBM+shift": p_pl, "RE_GBM+BLUP": p_re,
                                              "Anchor_mean": yt[a].mean()}))
        print(f"fold {fi} done | tau2={pr.tau2:.3f} sigma2={pr.sigma2:.3f}", flush=True)
    A = pd.concat(rows, ignore_index=True)
    A.to_csv(OUT / "anchored_predictions.csv", index=False)
    # summary tables
    summ = []
    dm = lambda v: v - pd.Series(v).groupby(sid).transform("mean").values
    big = pd.Series(sid).map(pd.Series(sid).value_counts()).values >= 3
    for k_, v in oof.items():
        if k_ == "PRISM_harmonised":
            continue
        summ.append(dict(model=k_, anchors=0, R2=r2_score(y, v), RMSE=mean_squared_error(y, v) ** 0.5,
                         within_R2=r2_score(dm(y)[big], dm(v)[big])))
    for k in [1, 3, 5]:
        sub = A[A.k == k]
        for m in ["PRISM", "Pooled_LGBM+shift", "RE_GBM+BLUP", "Anchor_mean"]:
            summ.append(dict(model=m, anchors=k, R2=r2_score(sub.y, sub[m]), RMSE=mean_squared_error(sub.y, sub[m]) ** 0.5,
                             within_R2=np.nan))
        cov = np.mean(np.abs(sub.y - sub.PRISM) <= 1.645 * np.sqrt(sub.PRISM_var))
        summ.append(dict(model="PRISM_gauss90_coverage", anchors=k, R2=cov, RMSE=np.nan, within_R2=np.nan))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "prism_performance.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT / "prism_interval_coverage_k0.csv", index=False)
    B = pd.concat(betas, axis=1)
    B["mean"], B["sd"] = B.mean(1), B.std(1)
    B.to_csv(OUT / "prism_protocol_beta_by_fold.csv")
    pd.DataFrame(taus).to_csv(OUT / "prism_variance_components.csv", index=False)
    print(S.round(3).to_string())
    print(pd.DataFrame(cover).round(3).to_string())
    print(B[["mean", "sd"]].round(3).to_string())
    full_fit(d)


def full_fit(d):
    """Full-data fit: honest variance decomposition (cross-fitted regional field) + harmonised abundance."""
    y, sid = d["y"].values, d["study_id"].values
    pr = PRISM().fit(d)
    fw, prot, _, gsd = pr.components(d)
    gb = d["study_id"].map(pr.cv_regional).values  # cross-fitted, not in-sample GP interpolation
    ref = dict(spectro_quant=1, Q_LDIR=1, Q_Raman=0, Q_fluor=0, ID_none=0, max_density=1.6, P_Fenton=0,
               P_alkali=0, P_acid=0, P_oil=0, log_size_min=np.log10(20), log_size_max=np.log10(5000))
    _, prot_ref, _, _ = pr.components(d, ref)
    d_out = d[["ID", "study_id", "land_use", "spectro_quant", "lat", "lon", "y", "abund_corr"]].copy()
    d_out["f_within"], d_out["protocol"], d_out["regional"] = fw, prot, gb
    d_out["lab_offset"] = pd.Series(y - fw - prot - gb).groupby(sid).transform("mean").values
    d_out["y_harmonised"] = y - prot + prot_ref
    d_out.to_csv(OUT / "prism_components_full.csv", index=False, encoding="utf-8-sig")
    comp = d_out[["f_within", "protocol", "regional", "lab_offset"]]
    resid = y - comp.sum(1)
    vd = {c: float(np.var(comp[c])) for c in comp.columns}
    vd["within_noise"] = float(np.var(resid))
    vd["total"] = float(np.var(y))
    cov_terms = float(np.var(y) - sum(vd[c] for c in ["f_within", "protocol", "regional", "lab_offset", "within_noise"]))
    vd["covariances"] = cov_terms
    (OUT / "variance_decomposition.json").write_text(json.dumps(dict(vd, beta=pr.beta.to_dict(), tau2=pr.tau2, sigma2=pr.sigma2), indent=2), encoding="utf-8")
    print(json.dumps(vd, indent=1))
    hm = d_out.groupby(["spectro_quant", "land_use"])[["y", "y_harmonised"]].median().unstack(0)
    hm.to_csv(OUT / "landuse_median_before_after_harmonisation.csv", encoding="utf-8-sig")
    print(hm.round(2).to_string())
    gap_before = (hm[("y", 1)] - hm[("y", 0)]).abs().mean()
    gap_after = (hm[("y_harmonised", 1)] - hm[("y_harmonised", 0)]).abs().mean()
    summary = dict(mean_abs_method_gap_before=float(gap_before), mean_abs_method_gap_after=float(gap_after),
                   global_median_reported_items_kg=float(np.median(d["abund_corr"])),
                   global_median_harmonised_items_kg=float(np.median(10 ** d_out["y_harmonised"] - 1)),
                   tau2=pr.tau2, sigma2=pr.sigma2)
    (OUT / "harmonisation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
