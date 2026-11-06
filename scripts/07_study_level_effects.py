# -*- coding: utf-8 -*-
"""
07_study_level_effects.py — protocol effects with the study as the unit (Mundlak-style R-learner and AIPW).

Outcome alpha_i = within-study FE offset of study i (composition-adjusted study level).
Estimators per treatment: R-learner (unweighted), R-learner (precision-weighted 1/(tau2+sigma2/n_i)),
AIPW (binary treatments), study-level permutation p-value.
Outputs -> results/causal/study_level_protocol_effects.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_level as sl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "causal"
TAU2 = 0.6  # residual between-study variance (PRISM, out-of-sample)

INSTR = ["Q_LDIR", "Q_Raman", "Q_muFTIR_img", "Q_fluor", "spectro_quant", "ID_none"]
PRET = ["max_density", "P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_H2O2", "P_Fenton", "P_alkali", "P_acid", "P_oil"]
SIZE = ["log_size_min", "log_size_max"]
SPECS = [
    ("spectro_quant", "alpha", sl.ENV_S + PRET + SIZE, True, "Spectroscopic vs visual counting"),
    ("Q_LDIR", "alpha", sl.ENV_S + PRET + SIZE + ["Q_Raman", "Q_muFTIR_img", "Q_fluor"], True, "LDIR counting"),
    ("ID_none", "alpha", sl.ENV_S + PRET + SIZE + ["spectro_quant"], True, "No polymer identification step"),
    ("max_density", "alpha", sl.ENV_S + INSTR + SIZE + ["P_H2O2", "P_Fenton", "P_alkali", "P_acid", "P_oil"], False, "Flotation density (+1 g cm-3)"),
    ("P_Fenton", "alpha", sl.ENV_S + INSTR + SIZE + ["max_density", "P_alkali", "P_acid"], True, "Fenton digestion"),
    ("P_alkali", "alpha", sl.ENV_S + INSTR + SIZE + ["max_density", "P_Fenton", "P_acid", "P_H2O2"], True, "Alkaline digestion"),
    ("log_size_min", "alpha_raw", sl.ENV_S + INSTR + PRET + ["log_size_max"], False, "Lower size limit, raw (= 1 - alpha)"),
    ("log_size_min", "alpha", sl.ENV_S + INSTR + PRET + ["log_size_max"], False, "Lower size limit, corrected"),
    ("year", "alpha", [c for c in sl.ENV_S if c != "year"] + INSTR + PRET + SIZE, False, "Sampling year (+1 yr)"),
]


def run(S, T, Y, ctrl, binary, label):
    ctrl = [c for c in ctrl if c != T]
    th, se = sl.rlearner(S, T, ctrl, outcome=Y, R=10)
    w = sl.precision_weights(S, TAU2)
    thw, sew = sl.rlearner(S, T, ctrl, outcome=Y, weights=w, R=10)
    tha, sea = sl.aipw(S, T, ctrl, outcome=Y, R=10) if binary else (np.nan, np.nan)
    thp, pp, psd = sl.perm_test(S, T, ctrl, outcome=Y)
    return dict(label=label, treatment=T, outcome=Y, n_studies=len(S), n_treated=int((S[T] > 0.5).sum()) if binary else np.nan,
                theta_R=th, se_R=se, lo_R=th - 1.96 * se, hi_R=th + 1.96 * se,
                theta_Rw=thw, se_Rw=sew, theta_AIPW=tha, se_AIPW=sea, lo_AIPW=tha - 1.96 * sea, hi_AIPW=tha + 1.96 * sea,
                p_perm_study=pp, perm_sd=psd)


def main():
    d = pd.read_pickle(ROOT / "data" / "processed" / "model_table.pkl")
    d = d[d["y"].notna() & d["y_raw"].notna()].reset_index(drop=True)
    S = sl.study_table(d, "y")
    S["alpha_raw"] = sl.study_table(d, "y_raw")["alpha"]
    S.to_csv(OUT / "study_level_table.csv", encoding="utf-8-sig")
    rows = Parallel(n_jobs=9)(delayed(run)(S, *sp) for sp in SPECS)
    R = pd.DataFrame(rows)
    a = R[(R.treatment == "log_size_min") & (R.outcome == "alpha_raw")].iloc[0]
    R.attrs["alpha_eff"] = 1 - a.theta_R
    R.to_csv(OUT / "study_level_protocol_effects.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(R[["label", "n_treated", "theta_R", "se_R", "theta_Rw", "se_Rw", "theta_AIPW", "se_AIPW", "p_perm_study"]].round(3).to_string())
    print(f"alpha_eff (study-level R-learner) = {1 - a.theta_R:.3f}  [{1 - a.hi_R:.3f}, {1 - a.lo_R:.3f}]")


if __name__ == "__main__":
    main()
