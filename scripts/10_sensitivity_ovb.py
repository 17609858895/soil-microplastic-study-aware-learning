# -*- coding: utf-8 -*-
"""
10_sensitivity_ovb.py — how strong would UNOBSERVED laboratory confounding have to be?

Semi-synthetic scenario S2 showed that no estimator can separate an instrument effect from unmeasured lab
practice that co-varies with the instrument. We therefore report omitted-variable-bias robustness values
(Cinelli & Hazlett 2020, JRSS-B) at the study level (n = 281 studies):

  RV        : minimum partial R2 that an unobserved confounder Z must have with BOTH the protocol (given X)
              and the study offset (given protocol and X) to drive the estimate to zero;
  RV_alpha  : ... to make it statistically non-significant at alpha = 0.05.

Computed for (i) study-level OLS with the same controls as the R-learner (exact sensemakr formulas) and
(ii) the R-learner t-statistic (df = n - 2; approximate). Benchmarks: partial R2 of each OBSERVED covariate
group with the protocol and with the outcome, i.e. "would Z need to be stronger than everything we observe?".
Outputs -> results/causal/sensitivity_ovb.csv, sensitivity_benchmarks.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_level as sl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "causal"
spec = importlib.util.spec_from_file_location("sle", ROOT / "scripts" / "07_study_level_effects.py")
sle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sle)

GROUPS = {
    "Location": ["sx", "sy", "sz", "abs_lat", "china"],
    "Sampling period": ["year"],
    "Land-use mix": [c for c in sl.ENV_S if c.startswith("share_") and not c.startswith("share_LU_")],
    "Land-use subtypes": [c for c in sl.ENV_S if c.startswith("share_LU_")],
    "Depth design": ["depth_mean", "depth_max", "multi_layer"],
    "Study size": ["log_n"],
}


def rv(f, df, alpha=None):
    if alpha is not None:
        f = f - abs(stats.t.ppf(alpha / 2, df - 1)) / np.sqrt(df - 1)
        if f <= 0:
            return 0.0
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def partial_r2(y, X_full, X_red):
    r_full = sm.OLS(y, X_full).fit().ssr
    r_red = sm.OLS(y, X_red).fit().ssr
    return max((r_red - r_full) / r_red, 0.0)


def analyse(S, T, Y, ctrl, label, rl_row):
    ctrl = [c for c in ctrl if c != T and S[c].std() > 0]
    Xc = (S[ctrl] - S[ctrl].mean()) / S[ctrl].std()
    X = sm.add_constant(pd.concat([S[[T]], Xc], axis=1))
    y = S[Y].values
    m = sm.OLS(y, X).fit()
    t, df = m.tvalues[T], m.df_resid
    f = abs(t) / np.sqrt(df)
    out = dict(label=label, treatment=T, outcome=Y, n=len(S), theta_ols=m.params[T], se_ols=m.bse[T], t_ols=t, df=df,
               r2_yd_given_x=t ** 2 / (t ** 2 + df), RV_ols=rv(f, df), RValpha_ols=rv(f, df, 0.05))
    t_r = rl_row["theta_R"] / rl_row["se_R"]
    df_r = len(S) - 2
    out.update(theta_R=rl_row["theta_R"], t_R=t_r, RV_R=rv(abs(t_r) / np.sqrt(df_r), df_r),
               RValpha_R=rv(abs(t_r) / np.sqrt(df_r), df_r, 0.05))
    bench = []
    Xc_const = sm.add_constant(Xc)
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in Xc.columns]
        if not cols:
            continue
        rest = [c for c in Xc.columns if c not in cols]
        r2_d = partial_r2(S[T].values, Xc_const, sm.add_constant(Xc[rest]))
        r2_y = partial_r2(y, X, sm.add_constant(pd.concat([S[[T]], Xc[rest]], axis=1)))
        bench.append(dict(label=label, group=g, r2_with_protocol=r2_d, r2_with_outcome=r2_y,
                          min_r2=min(r2_d, r2_y)))
    B = pd.DataFrame(bench)
    out["strongest_observed_group"] = B.loc[B["min_r2"].idxmax(), "group"]
    out["strongest_observed_min_r2"] = B["min_r2"].max()
    out["strongest_observed_r2_protocol"] = B.loc[B["min_r2"].idxmax(), "r2_with_protocol"]
    out["strongest_observed_r2_outcome"] = B.loc[B["min_r2"].idxmax(), "r2_with_outcome"]
    return out, B


def main():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna() & d["y_raw"].notna()].reset_index(drop=True)
    S = sl.study_table(d, "y")
    S["alpha_raw"] = sl.study_table(d, "y_raw")["alpha"]
    R = pd.read_csv(OUT / "study_level_protocol_effects.csv")
    rows, benches = [], []
    keep = {"Spectroscopic vs visual counting", "LDIR counting", "No polymer identification step",
            "Flotation density (+1 g cm-3)", "Lower size limit, raw (= 1 - alpha)"}
    for T, Y, ctrl, binary, label in sle.SPECS:
        if label not in keep:
            continue
        rl = R[R["label"] == label].iloc[0]
        o, B = analyse(S, T, Y, ctrl, label, rl)
        rows.append(o)
        benches.append(B)
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "sensitivity_ovb.csv", index=False, encoding="utf-8-sig")
    pd.concat(benches).to_csv(OUT / "sensitivity_benchmarks.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(O[["label", "theta_ols", "t_ols", "RV_ols", "RValpha_ols", "theta_R", "t_R", "RV_R", "RValpha_R",
             "strongest_observed_group", "strongest_observed_r2_protocol", "strongest_observed_r2_outcome"]].round(3).to_string())
    print(pd.concat(benches).round(3).to_string())


if __name__ == "__main__":
    main()
