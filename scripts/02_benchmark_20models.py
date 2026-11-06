# -*- coding: utf-8 -*-
"""
02_benchmark_20models.py
20-algorithm benchmark + nested super-learner stacking under three validation designs:
  random  : record-level 5-fold (what most published soil-MP ML papers use)
  site    : coordinate-grouped 5-fold (all layers of one coordinate stay together)
  study   : leave-study-out GroupKFold(5) (prediction for a new, unseen study/laboratory)
plus the "study-fingerprint null": models given ONLY random numbers that are constant within a study
(they carry no environmental or methodological information, only study identity).

Outputs -> results/benchmark/
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026


def load():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    g = json.loads((ROOT / "data" / "processed" / "feature_groups.json").read_text(encoding="utf-8"))
    d = d[d["y"].notna()].reset_index(drop=True)
    X = d[g["M_protocol"] + g["E_env"] + g["E_soil_sparse"]].copy()
    X = X.join(pd.get_dummies(d["land_use"], prefix="LUc").astype(int))
    X = X.join(pd.get_dummies(d["continent"], prefix="CONT").astype(int))
    X = X.astype(float)
    site = (d["lat"].round(4).astype(str) + "_" + d["lon"].round(4).astype(str)).astype("category").cat.codes.values
    return d, X, d["y"].values, d["study_id"].values, site


def models():
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    lin = lambda est: make_pipeline(imp(), StandardScaler(), est)
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    return {
        "Ridge": lin(Ridge(alpha=10.0)),
        "Lasso": lin(Lasso(alpha=0.005, max_iter=20000)),
        "ElasticNet": lin(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
        "BayesRidge": lin(BayesianRidge()),
        "Huber": lin(HuberRegressor(alpha=1e-3, max_iter=2000)),
        "PLS": lin(PLSRegression(n_components=10)),
        "KNN": lin(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "SVR": lin(SVR(C=3.0, epsilon=0.1, gamma="scale")),
        "KRR-Nystroem": make_pipeline(imp(), StandardScaler(), Nystroem(gamma=0.02, n_components=600, random_state=SEED), Ridge(alpha=1.0)),
        "MLP": lin(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=600,
                                early_stopping=True, random_state=SEED)),
        "DecisionTree": make_pipeline(imp(), DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=SEED)),
        "RandomForest": make_pipeline(imp(), RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=SEED)),
        "ExtraTrees": make_pipeline(imp(), ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.6, n_jobs=2, random_state=SEED)),
        "Bagging": make_pipeline(imp(), BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), n_estimators=200, n_jobs=2, random_state=SEED)),
        "AdaBoost": make_pipeline(imp(), AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300, learning_rate=0.05, random_state=SEED)),
        "GBDT": make_pipeline(imp(), GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=SEED)),
        "HistGB": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
        "XGBoost": xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                                    min_child_weight=5, reg_lambda=1.0, n_jobs=2, random_state=SEED, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=2, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3, thread_count=2,
                                      random_seed=SEED, verbose=0),
    }


STACK_BASE = ["Ridge", "KNN", "SVR", "ExtraTrees", "RandomForest", "LightGBM", "CatBoost", "MLP"]


def splitter(scheme, n_splits=5):
    if scheme == "random":
        return KFold(n_splits, shuffle=True, random_state=SEED)
    return GroupKFold(n_splits)


def groups_for(scheme, sid, site):
    return {"random": None, "site": site, "study": sid}[scheme]


def fit_predict(est, Xtr, ytr, Xte):
    m = clone(est).fit(Xtr, ytr)
    return np.ravel(m.predict(Xte)), np.ravel(m.predict(Xtr))


def run_one(name, est, scheme, X, y, sid, site):
    t0 = time.time()
    cv = splitter(scheme)
    grp = groups_for(scheme, sid, site)
    oof = np.zeros_like(y)
    tr_r2 = []
    for tr, te in cv.split(X, y, grp):
        p_te, p_tr = fit_predict(est, X.iloc[tr], y[tr], X.iloc[te])
        oof[te] = p_te
        tr_r2.append(r2_score(y[tr], p_tr))
    return dict(model=name, scheme=scheme, R2=r2_score(y, oof), RMSE=mean_squared_error(y, oof) ** 0.5,
                MAE=mean_absolute_error(y, oof), train_R2=float(np.mean(tr_r2)), sec=time.time() - t0), oof


def run_stack(base, scheme, X, y, sid, site):
    """Nested super learner: inner OOF (same validation design) -> NNLS meta-weights."""
    t0 = time.time()
    cv = splitter(scheme)
    grp = groups_for(scheme, sid, site)
    oof = np.zeros_like(y)
    weights = []
    for tr, te in cv.split(X, y, grp):
        Xtr, ytr = X.iloc[tr], y[tr]
        inner = splitter(scheme)
        g_in = None if grp is None else grp[tr]
        Z = np.zeros((len(tr), len(STACK_BASE)))
        for itr, ite in inner.split(Xtr, ytr, g_in):
            for j, n in enumerate(STACK_BASE):
                Z[ite, j] = fit_predict(base[n], Xtr.iloc[itr], ytr[itr], Xtr.iloc[ite])[0]
        Z1 = np.column_stack([Z, np.ones(len(tr))])
        w, _ = nnls(np.column_stack([Z, np.ones(len(tr)), -np.ones(len(tr))]), ytr)
        w_base, b = w[:-2], w[-2] - w[-1]
        weights.append(w_base)
        Zte = np.column_stack([fit_predict(base[n], Xtr, ytr, X.iloc[te])[0] for n in STACK_BASE])
        oof[te] = Zte @ w_base + b
    wmean = np.mean(weights, axis=0)
    return dict(model="SuperLearner", scheme=scheme, R2=r2_score(y, oof), RMSE=mean_squared_error(y, oof) ** 0.5,
                MAE=mean_absolute_error(y, oof), train_R2=np.nan, sec=time.time() - t0,
                weights=json.dumps(dict(zip(STACK_BASE, np.round(wmean, 3))))), oof


def fingerprint_null(y, sid, site, n_rep=5):
    """Random features constant within study -> how much R2 is pure study identity?"""
    import lightgbm as lgb
    rows = []
    rng = np.random.default_rng(SEED)
    for rep in range(n_rep):
        codes = rng.normal(size=(sid.max() + 1, 8))
        Xn = pd.DataFrame(codes[sid], columns=[f"hash{i}" for i in range(8)])
        for scheme in ["random", "site", "study"]:
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=5, verbose=-1, n_jobs=4)
            r, _ = run_one("StudyHash-null", m, scheme, Xn, y, sid, site)
            r["rep"] = rep
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    d, X, y, sid, site = load()
    print(f"n={len(y)} records, {len(np.unique(sid))} studies, {len(np.unique(site))} coordinate sites, p={X.shape[1]}", flush=True)
    base = models()
    jobs = [(n, e, s) for s in ["random", "site", "study"] for n, e in base.items()]
    res = Parallel(n_jobs=14, verbose=5)(delayed(run_one)(n, e, s, X, y, sid, site) for n, e, s in jobs)
    rows = [r for r, _ in res]
    oofs = {f"{r['model']}|{r['scheme']}": o for r, o in res}
    print("base models done", flush=True)
    st = Parallel(n_jobs=3)(delayed(run_stack)(base, s, X, y, sid, site) for s in ["random", "site", "study"])
    for r, o in st:
        rows.append(r)
        oofs[f"SuperLearner|{r['scheme']}"] = o
    tab = pd.DataFrame(rows)
    tab["gap_train_minus_cv"] = tab["train_R2"] - tab["R2"]
    tab.to_csv(OUT / "benchmark_20models_3schemes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oofs).assign(y=y, study_id=sid, site=site).to_csv(OUT / "oof_predictions.csv", index=False)
    null = fingerprint_null(y, sid, site)
    null.to_csv(OUT / "study_fingerprint_null.csv", index=False)
    piv = tab.pivot(index="model", columns="scheme", values="R2")[["random", "site", "study"]].sort_values("random", ascending=False)
    print(piv.round(3).to_string())
    print(null.groupby("scheme")[["R2", "RMSE"]].mean().round(3).to_string())
    # intra-class (study) correlation ceiling
    sm = pd.Series(y).groupby(sid).transform("mean")
    print("between-study variance share:", round(1 - np.var(y - sm) / np.var(y), 3))


if __name__ == "__main__":
    main()
