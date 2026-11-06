# -*- coding: utf-8 -*-
"""
03_causal_decomposition.py
Identification-aware decomposition of reported soil-microplastic abundance.

Key data fact (see data_summary.json): the measurement protocol NEVER varies within a study
(0 studies mix visual and spectroscopic counting; 1 study varies the lower size limit), whereas land use
varies within 76 studies and sampling depth within 78 studies. Therefore
  * environmental contrasts (land use, subtype, depth) are identified WITHIN studies (study fixed effects
    cancel every study-level measurement offset, observed or not);
  * protocol effects are identified only BETWEEN studies, conditional on environment -> study-clustered DML.

A. Within-study FE estimates (cluster-robust by study)
B. Study-clustered double/debiased ML (PLR, LightGBM nuisances, group cross-fitting, repeated splits)
   incl. the size-limit slope on RAW abundance -> data-driven power-law exponent alpha_eff = 1 - theta
C. Cross-method concordance: land-use contrasts estimated separately from visual-based and
   spectroscopy-based studies (disjoint study sets), naive medians vs within-study FE.
Outputs -> results/causal/
"""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "causal"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
LU_ORDER = ["Cropland", "Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]


def load():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna()].reset_index(drop=True)
    d["log_depth"] = np.log10(d["depth_mid"].fillna(d["depth_mid"].median()) + 1)
    d["LU"] = pd.Categorical(d["land_use"], categories=LU_ORDER)
    return d


# ------------------------------------------------------------------ A. within-study FE
def fe_models(d):
    rows = []
    f_main = "y ~ C(LU, Treatment('Cropland')) + log_depth + C(study_id)"
    f_sub = ("y ~ C(LU, Treatment('Cropland')) + log_depth + LU_greenhouse + LU_mulch + LU_paddy + LU_orchard + "
             "LU_vegetable + LU_industrial + LU_traffic + LU_urban + LU_irrig_sludge + LU_natural + C(study_id)")
    for name, f in [("FE_main", f_main), ("FE_subtype", f_sub)]:
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["study_id"]})
        for k in m.params.index:
            if "study_id" in k or k == "Intercept":
                continue
            rows.append(dict(model=name, term=k.replace("C(LU, Treatment('Cropland'))[T.", "LU:").replace("]", ""),
                             coef=m.params[k], se=m.bse[k], p=m.pvalues[k],
                             lo=m.conf_int().loc[k, 0], hi=m.conf_int().loc[k, 1], n=int(m.nobs)))
    # naive pooled OLS (no study FE) for contrast
    m0 = smf.ols("y ~ C(LU, Treatment('Cropland')) + log_depth", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["study_id"]})
    for k in m0.params.index:
        if k == "Intercept":
            continue
        rows.append(dict(model="Pooled_noFE", term=k.replace("C(LU, Treatment('Cropland'))[T.", "LU:").replace("]", ""),
                         coef=m0.params[k], se=m0.bse[k], p=m0.pvalues[k], lo=m0.conf_int().loc[k, 0],
                         hi=m0.conf_int().loc[k, 1], n=int(m0.nobs)))
    # how many within-study contrasts support each land-use estimate
    sup = (d.groupby("study_id")["land_use"].agg(lambda s: set(s)).apply(lambda s: [lu for lu in s if "Cropland" in s and lu != "Cropland"]))
    support = pd.Series([lu for v in sup for lu in v]).value_counts().to_dict()
    r2_fe = smf.ols("y ~ C(study_id)", data=d).fit().rsquared
    r2_full = smf.ols(f_sub, data=d).fit().rsquared
    return pd.DataFrame(rows), support, dict(R2_study_FE_only=r2_fe, R2_FE_plus_env=r2_full,
                                             within_variance_explained=(r2_full - r2_fe) / (1 - r2_fe))


# ------------------------------------------------------------------ B. study-clustered DML
def lgbm(n=250):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                             verbose=-1, n_jobs=2)


ENV = ["lat", "lon", "abs_lat", "depth_up", "depth_low", "depth_mid", "layer", "samp_year_filled", "china_box",
       "LU_greenhouse", "LU_mulch", "LU_paddy", "LU_orchard", "LU_vegetable", "LU_industrial", "LU_traffic",
       "LU_urban", "LU_irrig_sludge", "LU_natural"]
INSTR = ["Q_LDIR", "Q_muFTIR_img", "Q_Raman", "Q_fluor", "spectro_quant", "ID_ATR", "ID_muFTIR", "ID_FTIR_any", "ID_Raman", "ID_none"]
PRET = ["P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_HCOOK", "P_oil", "P_H2O2", "P_Fenton", "P_alkali",
        "P_acid", "P_nilered", "max_density"]
SIZE = ["log_size_min", "log_size_max"]


def design(d, cols):
    X = d[cols].copy().astype(float)
    X = X.join(pd.get_dummies(d["land_use"], prefix="LUc").astype(float))
    X = X.join(pd.get_dummies(d["continent"], prefix="CONT").astype(float))
    return X


def group_folds(sid, k, rng):
    u = np.unique(sid)
    perm = rng.permutation(u)
    fold_of = {s: i % k for i, s in enumerate(perm)}
    return np.array([fold_of[s] for s in sid])


def dml_plr(y, T, X, sid, n_rep=5, k=5, weights=None, seed=SEED):
    """Partially linear DML, group cross-fitting by study, cluster-robust SE, median over repetitions."""
    rng = np.random.default_rng(seed)
    thetas, ses = [], []
    for r in range(n_rep):
        fold = group_folds(sid, k, rng)
        yr, tr_ = np.zeros_like(y, dtype=float), np.zeros_like(T, dtype=float)
        for f in range(k):
            tr, te = fold != f, fold == f
            yr[te] = y[te] - lgbm().fit(X[tr], y[tr]).predict(X[te])
            tr_[te] = T[te] - lgbm().fit(X[tr], T[tr]).predict(X[te])
        w = np.ones_like(y, dtype=float) if weights is None else weights
        theta = np.sum(w * tr_ * yr) / np.sum(w * tr_ ** 2)
        psi = w * tr_ * (yr - theta * tr_)
        cl = pd.Series(psi).groupby(sid).sum().values
        se = np.sqrt(np.sum(cl ** 2)) / np.sum(w * tr_ ** 2)
        thetas.append(theta)
        ses.append(se)
    thetas, ses = np.array(thetas), np.array(ses)
    th = np.median(thetas)
    se = np.sqrt(np.median(ses ** 2 + (thetas - th) ** 2))  # Chernozhukov et al. median aggregation
    resid_share = None
    return th, se


def dml_block(d):
    sid = d["study_id"].values
    specs = [
        # treatment, outcome, controls, label
        ("spectro_quant", "y", ENV + PRET + SIZE, "Spectroscopic vs visual counting -> corrected abundance"),
        ("spectro_quant", "y_raw", ENV + PRET + SIZE, "Spectroscopic vs visual counting -> raw abundance"),
        ("Q_LDIR", "y", ENV + PRET + SIZE + ["Q_Raman", "Q_muFTIR_img", "Q_fluor"], "LDIR counting (vs others) -> corrected"),
        ("max_density", "y", ENV + INSTR + SIZE + ["P_H2O2", "P_Fenton", "P_alkali", "P_acid", "P_oil"], "Flotation density (+1 g cm-3) -> corrected"),
        ("P_Fenton", "y", ENV + INSTR + SIZE + ["max_density", "P_alkali", "P_acid"], "Fenton digestion -> corrected"),
        ("P_alkali", "y", ENV + INSTR + SIZE + ["max_density", "P_Fenton", "P_acid", "P_H2O2"], "Alkaline digestion -> corrected"),
        ("log_size_min", "y_raw", ENV + INSTR + PRET + ["log_size_max"], "log10 lower size limit -> RAW abundance (1-alpha)"),
        ("log_size_min", "y", ENV + INSTR + PRET + ["log_size_max"], "log10 lower size limit -> CORRECTED abundance (residual)"),
        ("samp_year_filled", "y", [c for c in ENV if c != "samp_year_filled"] + INSTR + PRET + SIZE, "Sampling year (+1 yr) -> corrected"),
        ("log_depth", "y", [c for c in ENV if c not in ("depth_up", "depth_low", "depth_mid", "layer")] + INSTR + PRET + SIZE, "log10 depth -> corrected (between+within)"),
    ]
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=10)(delayed(_one_spec)(d, T, Y, ctrl, lab) for T, Y, ctrl, lab in specs)
    for r in rows:
        print(f"{r['label']:65s} theta={r['theta']:+.3f} ± {1.96*r['se']:.3f} | equal-study {r['theta_study_weighted']:+.3f} | placebo {r['placebo_theta']:+.3f} | China {r['theta_china']:+.3f}", flush=True)
    res = pd.DataFrame(rows)
    a = res[(res.treatment == "log_size_min") & (res.outcome == "y_raw")].iloc[0]
    alpha = dict(alpha_eff=1 - a.theta, alpha_lo=1 - a.hi, alpha_hi=1 - a.lo, alpha_ESSD=1.437)
    return res, alpha


def _one_spec(d, T, Y, ctrl, lab):
    if True:
        sub = d[d[T].notna() & d[Y].notna()].reset_index(drop=True)
        X = design(sub, [c for c in ctrl if c != T]).values
        yv, tv, s = sub[Y].values, sub[T].values.astype(float), sub["study_id"].values
        th, se = dml_plr(yv, tv, X, s, n_rep=3)
        # study-equal weighting sensitivity
        w = 1.0 / sub.groupby("study_id")["y"].transform("size").values
        thw, sew = dml_plr(yv, tv, X, s, n_rep=2, weights=w)
        # placebo: permute treatment across studies (keeps within-study structure)
        rng = np.random.default_rng(1)
        st = sub.groupby("study_id")[T].first()
        perm = pd.Series(rng.permutation(st.values), index=st.index)
        tp = sub["study_id"].map(perm).values.astype(float) if sub.groupby("study_id")[T].nunique().max() == 1 else rng.permutation(tv)
        thp, sep = dml_plr(yv, tp, X, s, n_rep=1)
        # China-only subset
        cs = sub["china_box"].values == 1
        thc, sec = dml_plr(yv[cs], tv[cs], X[cs], s[cs], n_rep=1) if cs.sum() > 300 else (np.nan, np.nan)
        # naive record-level DML (random folds, iid SE) for contrast with published practice
        thn, sen = dml_plr_iid(yv, tv, X)
        return dict(treatment=T, outcome=Y, label=lab, n=len(sub), n_studies=sub["study_id"].nunique(),
                    theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, p=2 * (1 - stats.norm.cdf(abs(th / se))),
                    theta_study_weighted=thw, se_study_weighted=sew, placebo_theta=thp, placebo_se=sep,
                    theta_china=thc, se_china=sec, theta_naive_iid=thn, se_naive_iid=sen,
                    se_inflation=se / sen)


def dml_plr_iid(y, T, X, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(y))
    yr, tr_ = np.zeros(len(y)), np.zeros(len(y))
    for f in range(k):
        a, b = fold != f, fold == f
        yr[b] = y[b] - lgbm().fit(X[a], y[a]).predict(X[b])
        tr_[b] = T[b] - lgbm().fit(X[a], T[a]).predict(X[b])
    th = np.sum(tr_ * yr) / np.sum(tr_ ** 2)
    psi = tr_ * (yr - th * tr_)
    return th, np.sqrt(np.sum(psi ** 2)) / np.sum(tr_ ** 2)


# ------------------------------------------------------------------ C. cross-method concordance
def concordance(d):
    out = {}
    lus = ["Forest Land", "Grassland", "Settlements", "Wetlands", "Other Land"]
    per = {}
    for grp, sub in d.groupby("spectro_quant"):
        g = "spectroscopy" if grp == 1 else "visual"
        med = sub.groupby("land_use")["abund_corr"].median()
        naive = {lu: np.log10(med.get(lu, np.nan) + 1) - np.log10(med.get("Cropland", np.nan) + 1) for lu in lus}
        try:
            m = smf.ols("y ~ C(LU, Treatment('Cropland')) + log_depth + C(study_id)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["study_id"]})
            fe = {lu: m.params.get(f"C(LU, Treatment('Cropland'))[T.{lu}]", np.nan) for lu in lus}
            fese = {lu: m.bse.get(f"C(LU, Treatment('Cropland'))[T.{lu}]", np.nan) for lu in lus}
        except Exception:
            fe, fese = {lu: np.nan for lu in lus}, {lu: np.nan for lu in lus}
        nstud = sub.groupby("study_id")["land_use"].nunique()
        per[g] = dict(naive=naive, fe=fe, fe_se=fese, n_records=len(sub), n_studies=sub["study_id"].nunique(),
                      n_multi_landuse_studies=int((nstud > 1).sum()),
                      median_by_lu=med.round(1).to_dict())
    rows = []
    for kind in ["naive", "fe"]:
        a = pd.Series(per["visual"][kind])
        b = pd.Series(per["spectroscopy"][kind])
        ok = a.notna() & b.notna()
        tau = stats.kendalltau(a[ok], b[ok]).statistic if ok.sum() >= 3 else np.nan
        rho = stats.spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 3 else np.nan
        rows.append(dict(estimator=kind, kendall_tau=tau, spearman=rho, n_landuses=int(ok.sum()),
                         mean_abs_diff=float(np.mean(np.abs(a[ok] - b[ok])))))
    out["per_group"] = per
    out["concordance"] = rows
    # formal test of method-invariance: do within-study land-use contrasts differ by counting method?
    mi = smf.ols("y ~ C(LU, Treatment('Cropland')) * spectro_quant + log_depth + C(study_id)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["study_id"]})
    inter = [k for k in mi.params.index if ":spectro_quant" in k and not np.isnan(mi.bse[k])]
    R = np.zeros((len(inter), len(mi.params)))
    for i, k in enumerate(inter):
        R[i, list(mi.params.index).index(k)] = 1
    wt = mi.wald_test(R, scalar=True)
    # naive counterpart: do pooled land-use differences differ by method? (no study FE)
    mn = smf.ols("y ~ C(LU, Treatment('Cropland')) * spectro_quant + log_depth", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["study_id"]})
    inter_n = [k for k in mn.params.index if ":spectro_quant" in k]
    Rn = np.zeros((len(inter_n), len(mn.params)))
    for i, k in enumerate(inter_n):
        Rn[i, list(mn.params.index).index(k)] = 1
    wn = mn.wald_test(Rn, scalar=True)
    out["method_invariance_test"] = dict(
        FE_interaction_terms={k: [float(mi.params[k]), float(mi.bse[k])] for k in inter},
        FE_wald_p=float(wn.pvalue) if False else float(wt.pvalue), FE_wald_stat=float(wt.statistic),
        pooled_interaction_terms={k: [float(mn.params[k]), float(mn.bse[k])] for k in inter_n},
        pooled_wald_p=float(wn.pvalue), pooled_wald_stat=float(wn.statistic))
    return out


def main():
    d = load()
    fe, support, r2 = fe_models(d)
    fe.to_csv(OUT / "within_study_FE_effects.csv", index=False, encoding="utf-8-sig")
    print(fe.round(3).to_string())
    print("within-study contrasts vs Cropland:", support)
    print(r2)
    conc = concordance(d)
    (OUT / "cross_method_concordance.json").write_text(json.dumps(conc, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(conc["concordance"], indent=1, default=float))
    print(json.dumps(conc["method_invariance_test"], indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('naive', 'fe', 'n_studies', 'n_multi_landuse_studies')} for k, v in conc['per_group'].items()}, indent=1, default=float))
    res, alpha = dml_block(d)
    res.to_csv(OUT / "dml_protocol_effects.csv", index=False, encoding="utf-8-sig")
    (OUT / "alpha_eff.json").write_text(json.dumps(alpha, indent=2), encoding="utf-8")
    (OUT / "fe_summary.json").write_text(json.dumps(dict(support=support, **r2), indent=2, default=float), encoding="utf-8")
    print(alpha)


if __name__ == "__main__":
    main()
