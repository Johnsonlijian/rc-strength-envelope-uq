"""
a31_sfrc_closure.py — close the fallback loop at the THIRD material (SFRC).

Mechanical arm: Narayanan & Darwish (1987), ACI Structural Journal 84(3),
doi:10.14359/2654 — the classical SFRC shear equation using the fibre factor
F = (Lf/df)·Vf·beta_bond, which the Lantsoght database tabulates directly
(column 'F'), so no fibre-property guessing is needed:

    v_u = e·[0.24·f_spfc + 80·rho·(d/a)] + 0.41·tau·F        [MPa]
    f_spfc = f_cuf/(20 - sqrt(F)) + 0.7 + sqrt(F)            [MPa]
    e = 1.0 for a/d > 2.8;  e = 2.8·(d/a) for a/d <= 2.8
    tau = 4.15 MPa;  f_cuf = cube strength ~= f_c(cyl)/0.8

Gate: global model-uncertainty stats must land in the published family for
this equation before the closure result is used (bias ~0.9-1.4, COV <~0.40).

Then the identical same-metric closure pipeline of a28 (bias-correction on
the in-envelope calibration split + ln-space split conformal) is run with
this mechanical arm against the ML arms on the slender SFRC set.

Outputs -> ../data/processed/sfrc_closure.csv (+ prints validation stats)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
from a28_steel_closure import run_closure, m_stats

RAW = Path("../data/raw/Zenodo_2578061_SFRC_shear_beams.xlsx")
PROC = Path("../data/processed")
POS = {"b": 2, "h": 3, "d": 4, "rho_l": 16, "fy": 17, "a_d": 18, "fc": 21,
       "beta_f": 23, "Lf": 24, "Vf": 25, "df_fib": 26, "aspect": 27,
       "F": 29, "V_test_kN": 38}
FEATS = ["b", "d", "fc", "rho_l", "a_d", "fy", "Vf"]


def load():
    raw = pd.read_excel(RAW, header=None)
    df = raw.iloc[3:, list(POS.values())].copy()
    df.columns = list(POS)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["b", "d", "fc", "rho_l", "a_d", "F", "V_test_kN"])
    df = df[(df.V_test_kN > 0) & (df.d > 0) & (df.a_d >= 2.5) & (df.F >= 0)]
    df["fy"] = df["fy"].fillna(df["fy"].median())
    df["Vf"] = df["Vf"].fillna(df["Vf"].median())
    return df.reset_index(drop=True)


def nd1987_V_kN(r):
    F = r.F
    fcu = r.fc / 0.8                       # cylinder -> cube
    fsp = fcu / (20.0 - np.sqrt(F)) + 0.7 + np.sqrt(F)
    e = 1.0 if r.a_d > 2.8 else 2.8 / r.a_d
    vb = 0.41 * 4.15 * F
    v = e * (0.24 * fsp + 80.0 * r.rho_l / r.a_d) + vb   # rho*(d/a) = rho/(a/d)
    return v * r.b * r.d / 1e3


def main():
    df = load()
    print(f"== SFRC closure (slender, n={len(df)}; d {df.d.min():.0f}-{df.d.max():.0f} mm; "
          f"F {df.F.min():.2f}-{df.F.max():.2f}) ==")
    Vnd = np.array([nd1987_V_kN(r) for r in df.itertuples()])
    s = m_stats(df.V_test_kN.values, Vnd)
    print(f"Narayanan-Darwish GLOBAL validation: bias={s['bias']:.2f} COV={s['cov']:.2f} "
          f"unsafe={s['unsafe']:.2f}  (publication family: bias ~0.9-1.4, COV <~0.40)")
    ok = 0.85 <= s["bias"] <= 1.55 and s["cov"] <= 0.42
    print("VALIDATION:", "PASS" if ok else "FAIL -> do not use closure result")
    if not ok:
        return

    d_hi = df.d.quantile(0.75)
    big = df[df.d >= d_hi]
    sb = m_stats(big.V_test_kN.values,
                 np.array([nd1987_V_kN(r) for r in big.itertuples()]))
    print(f"  out-of-envelope subset (d>={d_hi:.0f}, n={len(big)}): "
          f"bias={sb['bias']:.2f} COV={sb['cov']:.2f} unsafe={sb['unsafe']:.2f}")

    R = run_closure(df, FEATS, "V_test_kN",
                    {"Narayanan-Darwish": nd1987_V_kN}, label="SFRC")
    R.to_csv(PROC / "sfrc_closure.csv", index=False)
    g = (R.groupby("predictor")
           .agg(cov_ln_ind=("cov_ln_ind", "mean"), cov_ln_big=("cov_ln_big", "mean"),
                cov_ln_big_sd=("cov_ln_big", "std"), cov_raw_big=("cov_raw_big", "mean"),
                ooe_cov=("ooe_cov", "mean"), width=("width_factor_ln", "mean")).round(3))
    print("\nSame-metric closure on SFRC (target 0.90):")
    print(g.to_string())
    print("saved -> sfrc_closure.csv")


if __name__ == "__main__":
    main()
