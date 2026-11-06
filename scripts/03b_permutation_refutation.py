# -*- coding: utf-8 -*-
"""
03b_permutation_refutation.py
Study-level residual permutation test for each DML protocol effect (replaces single placebo draws).

For each specification: cross-fit (group folds by study) y~X and T~X once -> residuals (y_t, T_t).
theta_obs = sum(T_t * y_t) / sum(T_t^2).
Null: protocol has no effect -> study-mean residual treatments are exchangeable across studies.
Permute study-mean T_t across studies (1000 draws), recompute theta -> two-sided permutation p-value.
Also reports the record-level (iid) permutation p-value, which ignores clustering, for contrast.
Outputs -> results/causal/dml_permutation_tests.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cd", ROOT / "scripts" / "03_causal_decomposition.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

SPECS = [
    ("spectro_quant", "y", cd.ENV + cd.PRET + cd.SIZE),
    ("Q_LDIR", "y", cd.ENV + cd.PRET + cd.SIZE + ["Q_Raman", "Q_muFTIR_img", "Q_fluor"]),
    ("max_density", "y", cd.ENV + cd.INSTR + cd.SIZE + ["P_H2O2", "P_Fenton", "P_alkali", "P_acid", "P_oil"]),
    ("P_Fenton", "y", cd.ENV + cd.INSTR + cd.SIZE + ["max_density", "P_alkali", "P_acid"]),
    ("P_alkali", "y", cd.ENV + cd.INSTR + cd.SIZE + ["max_density", "P_Fenton", "P_acid", "P_H2O2"]),
    ("log_size_min", "y_raw", cd.ENV + cd.INSTR + cd.PRET + ["log_size_max"]),
    ("log_size_min", "y", cd.ENV + cd.INSTR + cd.PRET + ["log_size_max"]),
    ("samp_year_filled", "y", [c for c in cd.ENV if c != "samp_year_filled"] + cd.INSTR + cd.PRET + cd.SIZE),
]


def run(d, T, Y, ctrl, n_perm=1000):
    sub = d[d[T].notna() & d[Y].notna()].reset_index(drop=True)
    X = cd.design(sub, [c for c in ctrl if c != T]).values
    y, t, sid = sub[Y].values, sub[T].values.astype(float), sub["study_id"].values
    rng = np.random.default_rng(7)
    fold = cd.group_folds(sid, 5, rng)
    yr, tr = np.zeros(len(y)), np.zeros(len(y))
    for f in range(5):
        a, b = fold != f, fold == f
        yr[b] = y[b] - cd.lgbm().fit(X[a], y[a]).predict(X[b])
        tr[b] = t[b] - cd.lgbm().fit(X[a], t[a]).predict(X[b])
    th = np.sum(tr * yr) / np.sum(tr ** 2)
    g = pd.DataFrame({"s": sid, "t": tr, "y": yr})
    sy = g.groupby("s")["y"].sum().values
    st = g.groupby("s")["t"].mean().values
    n = g.groupby("s").size().values
    within = tr - g.groupby("s")["t"].transform("mean").values
    perm = []
    for _ in range(n_perm):
        p = rng.permutation(st)
        tp_rec = np.repeat(p, n)  # aligned with groupby order below
        # rebuild record-level permuted residual treatment in groupby order
        num = np.sum(p * sy)
        den = np.sum(n * p ** 2)
        perm.append(num / den)
    perm = np.array(perm)
    p_study = (np.sum(np.abs(perm) >= abs(th)) + 1) / (n_perm + 1)
    perm_iid = np.array([np.sum(rng.permutation(tr) * yr) / np.sum(tr ** 2) for _ in range(n_perm)])
    p_iid = (np.sum(np.abs(perm_iid) >= abs(th)) + 1) / (n_perm + 1)
    return dict(treatment=T, outcome=Y, theta=th, perm_sd_study=perm.std(), p_perm_study=p_study,
                perm_q025=np.quantile(perm, 0.025), perm_q975=np.quantile(perm, 0.975),
                perm_sd_iid=perm_iid.std(), p_perm_iid=p_iid, n_studies=len(n),
                within_study_T_share=float(np.var(within) / np.var(tr)))


def main():
    d = cd.load()
    rows = Parallel(n_jobs=8)(delayed(run)(d, T, Y, c) for T, Y, c in SPECS)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "causal" / "dml_permutation_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
