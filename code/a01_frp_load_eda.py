"""
a01_frp_load_eda.py — load + canonicalise + EDA of the figshare FRP shear DB.

Source: Univ. Sheffield ORDA "Shear database of RC FRP beams without shear
reinforcement", DOI 10.15131/shef.data.5057527, CC BY-NC 4.0. 327 specimens,
beams WITHOUT web/shear reinforcement. Columns selected by position (the file
has unicode headers). Saves processed/frp_clean.csv.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path("../data/raw/figshare_5057527_FRP_RC_beams_shear_no_web_reinforcement.xlsx")
OUT = Path("../data/processed"); OUT.mkdir(parents=True, exist_ok=True)

# positional map (schema order is fixed, 27 cols) -> canonical names
COLS = {2: "source", 7: "bw", 8: "d", 9: "h", 12: "fc", 17: "rho_l",
        18: "Ef", 19: "ffu", 20: "frp_type", 23: "failure", 24: "V_test_N",
        25: "a", 26: "a_d"}


def load():
    raw = pd.read_excel(RAW)
    df = raw.iloc[:, list(COLS)].copy()
    df.columns = list(COLS.values())
    for c in ["bw", "d", "h", "fc", "rho_l", "Ef", "ffu", "V_test_N", "a", "a_d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["V_test_kN"] = df["V_test_N"] / 1e3
    # recompute a/d where missing
    df.loc[df["a_d"].isna() & df["d"].gt(0), "a_d"] = df["a"] / df["d"]
    df["v_test_MPa"] = df["V_test_N"] / (df["bw"] * df["d"])   # nominal shear stress
    return df


def main():
    df = load()
    print("raw rows:", len(df))
    core = ["bw", "d", "fc", "rho_l", "Ef", "a_d", "V_test_N"]
    miss = df[core].isna().sum()
    print("\nmissing in core cols:\n", miss.to_string())
    clean = df.dropna(subset=core).copy()
    clean = clean[(clean["V_test_N"] > 0) & (clean["d"] > 0) & (clean["fc"] > 0)
                  & (clean["rho_l"] > 0) & (clean["Ef"] > 0)]
    print("\nclean rows:", len(clean))

    print("\n== ranges (clean) ==")
    desc = clean[["bw", "d", "h", "fc", "rho_l", "Ef", "ffu", "a_d",
                  "V_test_kN", "v_test_MPa"]].describe(
        percentiles=[.05, .25, .5, .75, .95]).T
    print(desc[["count", "mean", "std", "min", "5%", "50%", "95%", "max"]].to_string())

    print("\n== a/d regime (slender = a/d>=2.5) ==")
    print((clean["a_d"] >= 2.5).value_counts().rename({True: "slender", False: "deep/short"}).to_string())

    print("\n== size (d) distribution ==")
    for lo, hi in [(0, 200), (200, 300), (300, 450), (450, 700), (700, 9999)]:
        n = ((clean["d"] >= lo) & (clean["d"] < hi)).sum()
        print(f"  d in [{lo},{hi}) mm : {n}")

    print("\n== top source studies ==")
    print(clean["source"].value_counts().head(10).to_string())
    print("\n== FRP type ==")
    print(clean["frp_type"].value_counts().to_string())

    clean.to_csv(OUT / "frp_clean.csv", index=False)
    print(f"\nsaved -> {OUT/'frp_clean.csv'}  ({len(clean)} rows)")
    return clean


if __name__ == "__main__":
    main()
