# -*- coding: utf-8 -*-
"""
01_prepare_data.py
Build the record-level modelling table from the Zenodo workbook (v1, doi:10.5281/zenodo.20806151).

Design rules
- The raw workbook (data/raw/) is never modified; all derived columns are written to data/processed/.
- Target = log10(abundance_corrected + 1)  (items kg-1, power-law size-corrected, 20-5000 um).
  Raw abundance (abundance_numeric) is kept for the "learned correction" comparison.
- Measurement-protocol variables (M) and environmental/context variables (E) are separated explicitly,
  because the paper's central question is to disentangle them.
- Composition variables (size/shape/polymer %) are *post-measurement* descriptors (they depend on the
  detection window and instrument), so they are flagged as such and are NOT used as confounders.
- Study identity = Title (297 source papers, 281 with a usable abundance) -> grouped CV / clustering.
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "A global dataset of soil microplastics (2005-2026).xlsx"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def to_num(s):
    return pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def flag(series, pattern):
    return series.fillna("").str.contains(pattern, case=False, regex=True).astype(int)


def main():
    info = pd.read_excel(RAW, sheet_name="information")
    ch = pd.read_excel(RAW, sheet_name="characteristics", header=[0, 1])
    info.columns = [c.replace("(μm)", "_um").strip() for c in info.columns]
    ch.columns = ["ID" if a == "ID" else ("Land use" if a == "Land use" else f"{a.split(' ')[0]}_{b}")
                  for a, b in ch.columns]
    ch = ch.drop(columns=["Land use"])
    ch.columns = [c.replace(" μm", "um").replace("(%)", "").replace(" ", "") for c in ch.columns]
    df = info.merge(ch, on="ID", how="left", validate="one_to_one")

    d = pd.DataFrame({"ID": df["ID"]})
    # ---------------- identifiers ----------------
    d["study"] = df["Title"].str.strip().str.lower()
    d["study_id"] = d["study"].astype("category").cat.codes
    d["author"] = df["Author"]
    d["journal"] = df["Journal"]
    d["location"] = df["Location"]

    # ---------------- targets ----------------
    d["abund_raw"] = df["abundance_numeric"]
    d["CF"] = df["CF"]
    d["abund_corr"] = df["abundance_corrected"]
    d["y"] = np.log10(d["abund_corr"] + 1.0)
    d["y_raw"] = np.log10(d["abund_raw"] + 1.0)
    d["nondetect"] = (d["abund_corr"] == 0).astype(int)

    # ---------------- measurement protocol (M) ----------------
    ident = df["Identification and Quantification of MPs"].fillna("").str.replace("\t", " ")
    d["method_text"] = ident
    quant_part = ident.str.extract(r"^([^+]*?)\(quantification")[0].fillna(ident)
    q = quant_part.str.lower()
    # quantification instrument (the step that counts particles)
    d["Q_LDIR"] = q.str.contains("ldir").astype(int)
    d["Q_muFTIR_img"] = (q.str.contains("ftir") & ~q.str.contains("ldir")).astype(int)
    d["Q_Raman"] = q.str.contains("raman").astype(int)
    d["Q_fluor"] = q.str.contains("fluoresc|nile").astype(int)
    d["Q_visual"] = ((q.str.contains("microscop|stereo|binocular|visual|optical|naked|magnif"))
                     & (d[["Q_LDIR", "Q_muFTIR_img", "Q_Raman", "Q_fluor"]].sum(1) == 0)).astype(int)
    d["Q_thermal"] = q.str.contains("py-gc|tga|ted-gc|pyrol").astype(int)
    d["spectro_quant"] = ((d["Q_LDIR"] + d["Q_muFTIR_img"] + d["Q_Raman"]) > 0).astype(int)
    # identification support
    il = ident.str.lower()
    d["ID_ATR"] = il.str.contains("atr").astype(int)
    d["ID_muFTIR"] = il.str.contains(r"μ-ftir|μftir|micro-ftir|mu-ft|μ-ft|micro-atr").astype(int)
    d["ID_FTIR_any"] = il.str.contains("ftir|ft-ir").astype(int)
    d["ID_Raman"] = il.str.contains("raman").astype(int)
    d["ID_hotneedle"] = il.str.contains("hot").astype(int)
    d["ID_SEM"] = il.str.contains("sem").astype(int)
    d["ID_none"] = (~il.str.contains("identification")).astype(int)
    md = to_num(df["Matching degree"].astype(str))
    d["match_degree"] = md.where((md > 0) & (md <= 1))

    pre = df["Sample pretreatment"].fillna("").str.replace("\t", " ")
    d["pretreat_text"] = pre
    d["P_NaCl"] = flag(pre, r"NaCl")
    d["P_ZnCl2"] = flag(pre, r"ZnCl")
    d["P_CaCl2"] = flag(pre, r"CaCl")
    d["P_NaI_KI"] = flag(pre, r"NaI|KI \(|KI\(|potassium iodide|sodium iodide")
    d["P_NaBr"] = flag(pre, r"NaBr")
    d["P_HCOOK"] = flag(pre, r"HCOOK|formate")
    d["P_oil"] = flag(pre, r"oil")
    d["P_H2O2"] = flag(pre, r"H2O2|hydrogen peroxide")
    d["P_Fenton"] = flag(pre, r"Fenton|FeSO4|Fe\(II\)|Fe2\+")
    d["P_alkali"] = flag(pre, r"KOH|NaOH|NaClO")
    d["P_acid"] = flag(pre, r"HNO3|H2SO4|HCl")
    d["P_enzyme"] = flag(pre, r"enzym")
    d["P_nilered"] = flag(pre, r"Nile")
    d["P_airflot"] = flag(pre, r"flotation|elutriat")
    d["n_density_salts"] = d[["P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_HCOOK"]].sum(1)
    dens = pre.str.findall(r"(\d\.\d+)\s*(?:-\s*\d\.\d+)?\s*g\s*/?\s*(?:cm|mL|ml)")
    d["max_density"] = dens.apply(lambda v: max(map(float, v)) if len(v) else np.nan)
    sat = pre.str.contains("saturat", case=False)
    d.loc[d["max_density"].isna() & sat & (d["P_ZnCl2"] == 1), "max_density"] = 1.7
    d.loc[d["max_density"].isna() & sat & (d["P_NaCl"] == 1), "max_density"] = 1.2
    d.loc[d["max_density"].isna() & sat & (d["P_CaCl2"] == 1), "max_density"] = 1.4

    d["size_min"] = df["Size min_um"]
    d["size_max"] = df["Size max_um"]
    d["log_size_min"] = np.log10(d["size_min"])
    d["log_size_max"] = np.log10(d["size_max"])
    d["log_window"] = d["log_size_max"] - d["log_size_min"]
    d["pub_year"] = df["Publication year"]

    # ---------------- environment / context (E) ----------------
    d["land_use"] = df["Land use"]
    olu = df["Original Land use"].fillna("").str.lower()
    d["olu_text"] = df["Original Land use"]
    d["LU_greenhouse"] = olu.str.contains("greenhouse|polytunnel|facility|plastic shed|tunnel").astype(int)
    d["LU_mulch"] = (olu.str.contains("mulch") & ~olu.str.contains("unmulch|non-mulch|no mulch")).astype(int)
    d["LU_paddy"] = olu.str.contains("paddy|rice").astype(int)
    d["LU_orchard"] = olu.str.contains("orchard|vineyard|viticulture|tea|plantation|banana|rubber").astype(int)
    d["LU_vegetable"] = olu.str.contains("vegetable|garden|allotment").astype(int)
    d["LU_industrial"] = olu.str.contains("industr|mining|landfill|waste|dump").astype(int)
    d["LU_traffic"] = olu.str.contains("traffic|road|highway|transport").astype(int)
    d["LU_urban"] = olu.str.contains("urban|residential|commercial|park|school|built|city").astype(int)
    d["LU_irrig_sludge"] = olu.str.contains("irrigat|sludge|sewage|wastewater|compost|manure").astype(int)
    d["LU_natural"] = olu.str.contains("natural|primary|pristine|nature|desert|uncultivat").astype(int)

    sd = df["Sampling date"].astype(str)
    d["samp_year"] = pd.to_numeric(sd.str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
    d["samp_year_filled"] = d["samp_year"].fillna(d["pub_year"] - 2)
    d["lat"] = df["Latitude"]
    d["lon"] = df["Longitude"]
    d["abs_lat"] = d["lat"].abs()
    d["depth_up"] = df["Upper depth (cm)"]
    d["depth_low"] = df["Lower depth (cm)"]
    d["depth_mid"] = (d["depth_up"] + d["depth_low"]) / 2
    d["depth_thick"] = d["depth_low"] - d["depth_up"]
    d["layer"] = df["Layer"]
    d["SOC"] = to_num(df["SOC/SOM (g/kg)"])
    d["clay"] = to_num(df["Clay (%)"])
    d["silt"] = to_num(df["Silt (%)"])
    d["sand"] = to_num(df["Sand (%)"])
    d["pH"] = to_num(df["pH"])
    d.loc[(d["pH"] < 2) | (d["pH"] > 11), "pH"] = np.nan

    # crude continent from coordinates (no external data)
    def continent(lat, lon):
        if lat < -60:
            return "Antarctica"
        if -35 <= lat <= 38 and -20 <= lon <= 52 and not (lat > 30 and lon > 35):
            return "Africa"
        if lon < -30:
            return "N_America" if lat > 13 else "S_America"
        if lat > 35 and -25 <= lon <= 45:
            return "Europe"
        if lon > 110 and lat < -10:
            return "Oceania"
        return "Asia"
    d["continent"] = [continent(a, b) for a, b in zip(d["lat"], d["lon"])]
    d["china_box"] = (d["lat"].between(18, 54) & d["lon"].between(73, 135)).astype(int)

    # ---------------- post-measurement composition (C) ----------------
    comp_cols = [c for c in df.columns if c.startswith(("Size_", "Shape_", "Composition"))]
    for c in comp_cols:
        d["C_" + c.replace("Composition", "Poly").replace("Size_", "Size").replace("Shape_", "Shape")] = df[c]
    d["dom_polymer"] = df["Dominant polymer"]
    d["dom_shape"] = df["Dominant shape"]
    d["size_bin"] = df["Standardized Size Bin"]

    d.to_csv(OUT / "model_table.csv", index=False, encoding="utf-8-sig")
    d.to_pickle(OUT / "model_table.pkl")

    groups = {
        "M_protocol": ["Q_LDIR", "Q_muFTIR_img", "Q_Raman", "Q_fluor", "Q_visual", "Q_thermal", "spectro_quant",
                       "ID_ATR", "ID_muFTIR", "ID_FTIR_any", "ID_Raman", "ID_hotneedle", "ID_SEM", "ID_none",
                       "P_NaCl", "P_ZnCl2", "P_CaCl2", "P_NaI_KI", "P_NaBr", "P_HCOOK", "P_oil", "P_H2O2",
                       "P_Fenton", "P_alkali", "P_acid", "P_enzyme", "P_nilered", "P_airflot", "n_density_salts",
                       "max_density", "log_size_min", "log_size_max", "log_window"],
        "E_env": ["lat", "lon", "abs_lat", "depth_up", "depth_low", "depth_mid", "depth_thick", "layer",
                  "samp_year_filled", "china_box", "LU_greenhouse", "LU_mulch", "LU_paddy", "LU_orchard",
                  "LU_vegetable", "LU_industrial", "LU_traffic", "LU_urban", "LU_irrig_sludge", "LU_natural"],
        "E_cat": ["land_use", "continent"],
        "E_soil_sparse": ["SOC", "clay", "silt", "sand", "pH"],
        "C_post": [c for c in d.columns if c.startswith("C_")],
    }
    (OUT / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")

    m = d["y"].notna()
    summ = {
        "records": int(len(d)), "records_with_target": int(m.sum()), "studies": int(d["study"].nunique()),
        "studies_with_target": int(d.loc[m, "study"].nunique()), "nondetects": int(d["nondetect"].sum()),
        "spectro_quant_share": float(d.loc[m, "spectro_quant"].mean()),
        "continent_counts": d.loc[m, "continent"].value_counts().to_dict(),
        "studies_with_both_visual_and_spectro": int((d[m].groupby("study")["spectro_quant"].nunique() > 1).sum()),
        "studies_with_multiple_size_min": int((d[m].groupby("study")["size_min"].nunique() > 1).sum()),
        "studies_with_multiple_landuse": int((d[m].groupby("study")["land_use"].nunique() > 1).sum()),
        "studies_with_multiple_depth": int((d[m].groupby("study")["depth_mid"].nunique() > 1).sum()),
    }
    (OUT / "data_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(d[groups["M_protocol"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
