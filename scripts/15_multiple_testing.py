"""Multiple-testing adjustment for the two families of hypothesis tests.

Family 1: the nine study-level protocol specifications (permutation p-values).
Family 2: the eight within-study fixed-effect terms (wild cluster restricted bootstrap p-values).
Benjamini-Hochberg (false discovery rate) and Holm (family-wise) adjustments are reported.

Output: results/causal/multiple_testing_adjusted.csv
"""
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

study = pd.read_csv(RES / "causal" / "study_level_protocol_effects.csv")
fam1 = pd.DataFrame({
    "family": "study-level protocol (permutation)",
    "term": study["label"],
    "estimate": study["theta_R"],
    "p_raw": study["p_perm_study"],
})

fe = pd.read_csv(RES / "robustness" / "fe_wild_cluster_bootstrap.csv")
fam2 = pd.DataFrame({
    "family": "within-study fixed effects (wild cluster bootstrap)",
    "term": fe["term"],
    "estimate": fe["coef"],
    "p_raw": fe["p_wcr"],
})

out = []
for fam in (fam1, fam2):
    fam = fam.copy()
    fam["q_BH"] = multipletests(fam["p_raw"], method="fdr_bh")[1]
    fam["p_Holm"] = multipletests(fam["p_raw"], method="holm")[1]
    fam["m_tests"] = len(fam)
    out.append(fam)
res = pd.concat(out, ignore_index=True)
res.to_csv(RES / "causal" / "multiple_testing_adjusted.csv", index=False)
with pd.option_context("display.width", 200):
    print(res.to_string())
