"""Run the core numerical analyses in dependency order.

Usage
  python run_all.py                 # all steps
  python run_all.py --from 05       # resume from a step
  python run_all.py --only 01          # prepare the modelling table
  python run_all.py --list          # show the steps
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "A global dataset of soil microplastics (2005-2026).xlsx"
RAW_MD5 = "9900e8814efde19aed1f54ec88bd1bf1"  # checksum listed on the Zenodo record (version 1)

STEPS = [
    ("01", "01_prepare_data.py", "modelling table from the workbook"),
    ("02", "02_benchmark_20models.py", "20 algorithms + super learner under three validation designs; study-code null"),
    ("03", "03_causal_decomposition.py", "within-study fixed effects; record-level DML; cross-method concordance"),
    ("03b", "03b_permutation_refutation.py", "study-level and record-level permutation tests of the DML effects"),
    ("07", "07_study_level_effects.py", "protocol effects with the study as the unit (R-learner, AIPW)"),
    ("28", "28_density_endpoint_sensitivity.py", "independent density-range endpoint coding sensitivity"),
    ("10", "10_sensitivity_ovb.py", "robustness values for unobserved confounding"),
    ("04", "04_semisynthetic_validation.py", "semi-synthetic benchmark with known effects (S1, S2)"),
    ("05", "05_prism_model.py", "PRISM transfer model: leave-study-out CV, calibration samples, harmonisation"),
    ("08", "08_robustness.py", "wild cluster bootstrap, leave-one-study-out, fixed-effect boosting ALE, non-detects"),
    ("09", "09_prism_bootstrap.py", "study-level bootstrap of PRISM and the harmonised median"),
    ("11", "11_shap_fingerprint.py", "SHAP attributions with and without random study codes"),
    ("12", "12_conformal_by_region.py", "interval coverage by region"),
    ("27", "27_audit_query_matched_transfer.py", "zero-calibration baseline scored on the identical query prediction occasions"),
    ("15", "15_multiple_testing.py", "Benjamini-Hochberg and Holm adjustments"),
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", help="first step to run (e.g. 05)")
    ap.add_argument("--only", nargs="+", help="run only these steps")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()
    ids = [s[0] for s in STEPS]
    if args.list:
        for sid, script, what in STEPS:
            print(f"{sid:>4}  {script:<34} {what}")
        return 0
    if args.only:
        bad = [s for s in args.only if s not in ids]
        if bad:
            ap.error(f"unknown step(s) {bad}; choose from {ids}")
        todo = [s for s in STEPS if s[0] in args.only]
    elif args.start:
        if args.start not in ids:
            ap.error(f"unknown step {args.start!r}; choose from {ids}")
        todo = STEPS[ids.index(args.start):]
    else:
        todo = STEPS
    if todo[0][0] == "01":
        if not RAW.exists():
            sys.exit(f"Workbook not found: {RAW.relative_to(ROOT)}\nDownload it from doi:10.5281/zenodo.20806151 "
                     "(see data/README.md).")
        got = md5(RAW)
        if got != RAW_MD5:
            sys.exit(f"MD5 of the workbook is {got}, expected {RAW_MD5} (Zenodo version 1).")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for sid, script, what in todo:
        print(f"\n=== {sid}  {script}  ({what})", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)], cwd=ROOT, env=env, check=True)
        print(f"=== {sid} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
