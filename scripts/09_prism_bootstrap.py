# -*- coding: utf-8 -*-
"""
09_prism_bootstrap.py — study-level cluster bootstrap of the full PRISM fit (B = 80).
Each replicate resamples studies with replacement (duplicates relabelled), refits PRISM, and applies the
replicate's protocol coefficients to the ORIGINAL records -> uncertainty for
  * protocol coefficients beta,
  * harmonised global median abundance (reference protocol: LDIR, 20-5000 um, 1.6 g cm-3, H2O2),
  * mean absolute visual-vs-spectroscopic land-use gap after harmonisation,
  * variance components tau2, sigma2.
Outputs -> results/prism/prism_bootstrap.csv, prism_bootstrap_summary.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pm", ROOT / "scripts" / "05_prism_model.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)
_lgbm_orig = pm.lgbm


def _lgbm_single(n=400, leaves=15, mcs=30):
    """Single-threaded LightGBM inside bootstrap workers (OpenMP oversubscription made multi-threaded fits stall)."""
    m = _lgbm_orig(n, leaves, mcs)
    m.set_params(n_jobs=1)
    return m


pm.lgbm = _lgbm_single
OUT = ROOT / "results" / "prism"
REF = dict(spectro_quant=1, Q_LDIR=1, Q_Raman=0, Q_fluor=0, ID_none=0, max_density=1.6, P_Fenton=0,
           P_alkali=0, P_acid=0, P_oil=0, log_size_min=np.log10(20), log_size_max=np.log10(5000))


def harmonise(d, beta):
    M = d[pm.M_COLS].copy()
    prot = M.values @ beta
    Mr = M.copy()
    for k, v in REF.items():
        Mr[k] = v
    return d["y"].values - prot + Mr.values @ beta


def stats_from(d, beta):
    yh = harmonise(d, beta)
    t = d.assign(yh=yh).groupby(["spectro_quant", "land_use"])[["y", "yh"]].median().unstack(0)
    gap_before = float((t[("y", 1)] - t[("y", 0)]).abs().mean())
    gap_after = float((t[("yh", 1)] - t[("yh", 0)]).abs().mean())
    return dict(global_median_harmonised=float(np.median(10 ** yh - 1)), gap_before=gap_before, gap_after=gap_after)


def one(d, b):
    pm.lgbm = _lgbm_single  # re-apply inside loky worker
    rng = np.random.default_rng(b)
    studies = d["study_id"].unique()
    pick = rng.choice(studies, len(studies), replace=True)
    db = pd.concat([d[d.study_id == s].assign(study_id=k) for k, s in enumerate(pick)], ignore_index=True)
    pr = pm.PRISM().fit(db)
    out = dict(rep=b, tau2=pr.tau2, sigma2=pr.sigma2, **{f"beta_{k}": v for k, v in pr.beta.items()})
    out.update(stats_from(d, pr.beta.values))
    return out


def main(B=200):
    d = pm.load()
    full = pm.PRISM().fit(d)
    point = dict(tau2=full.tau2, sigma2=full.sigma2, **{f"beta_{k}": v for k, v in full.beta.items()}, **stats_from(d, full.beta.values))
    rows = Parallel(n_jobs=12, verbose=10)(delayed(one)(d, b) for b in range(B))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "prism_bootstrap.csv", index=False)
    summ = {k: dict(point=float(point[k]), lo=float(R[k].quantile(0.025)), hi=float(R[k].quantile(0.975))) for k in point}
    summ["reported_global_median"] = float(np.median(d["abund_corr"]))
    (OUT / "prism_bootstrap_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for k, v in summ.items():
        print(k, v)


if __name__ == "__main__":
    main()
