# Input data

The analysis uses one file, the workbook of the global soil microplastic compilation:

Wu, P., Su, Y., Wang, M., Yang, X., Rillig, M. C., Wang, X., Liu, R., Ye, S., Zhou, L., Shi, Z., and Chen, S. (2026).
A global dataset of soil microplastics (2005–2026). Dataset version 1, Zenodo,
https://doi.org/10.5281/zenodo.20806151 (licence CC BY 4.0). Data description:
*Earth System Science Data Discussions*, https://doi.org/10.5194/essd-2026-536.

The workbook is not redistributed here. Download it from the Zenodo record and save it, without renaming, as

```
data/raw/A global dataset of soil microplastics (2005-2026).xlsx
```

| Property | Value |
|---|---|
| Size | 1,382,513 bytes |
| MD5 (as listed on Zenodo) | `9900e8814efde19aed1f54ec88bd1bf1` |
| SHA-256 | `8dba8a1e8a36aca6f5ce6f1b9dcc1091da7d5567b08ec802c742b4c0804ea06e` |

`run_all.py` checks the MD5 before the first step. Step 01 reads the sheets `information` and `characteristics`
and writes the modelling table to `data/processed/` (not tracked; it is rebuilt in seconds).
