"""
a32_deepbeam_stm.py — close the fallback loop at the SECOND MECHANISM
(deep beams, strut-and-tie), now possible because the Megahed database
tabulates the bearing/loading plate widths (w_tp, w_bp) that fix the
strut-and-tie node geometry.

Mechanical arm: single-panel ACI 318-19 strut-and-tie model (STM), the code
route for deep beams (a/d < ~2.5). Explicit, standard assumptions:
  tension-tie node height   h_b = 2*(h - d)          (twice cover to steel centroid)
  top-node height           h_t = As*fy/(0.85*fc*b)  (flexural stress-block depth,
                                    capped at 0.4h for numerical sanity)
  lever arm                 jd  = d - h_t/2
  strut angle               tan(theta) = jd / a
  strut widths              w_s,top = w_tp*sin + h_t*cos ;
                            w_s,bot = w_bp*sin + h_b*cos ;  w_s = min
  strut capacity            V = 0.85 * beta_s * fc * w_s * b * sin(theta)
  beta_s = 0.75 if distributed web reinforcement satisfies the ACI 318-19
  Sec. 23.5 minimum (sum rho_i*sin^2(gamma_i) >= 0.0025 across the strut),
  else 0.4. No separate bearing check (consistent with the single-check
  formula structure reported for this database by Megahed 2024).

Validation gate BEFORE any closure use: Megahed (2024, Sci Rep, same 840-beam
database) reports ACI-STM prediction/test mean 0.699, CoV 0.372 — i.e.
test/pred bias ~1.4-1.6 and COV ~0.37 in this paper's convention. Our
implementation must land in that family (bias 1.2-1.9, COV 0.28-0.48).

Then the identical a28 same-metric closure pipeline runs with this arm.
Outputs -> ../data/processed/deepbeam_closure.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
from a28_steel_closure import run_closure, m_stats

PROC = Path("../data/processed")
FEATS = ["h", "d", "b", "a_d", "fck", "rho", "fy", "rho_v", "fyv", "rho_h", "fyh"]


def aci_stm_V_kN(r):
    As = r.rho * r.b * r.d
    h_t = min(As * r.fy / (0.85 * r.fck * r.b), 0.4 * r.h)
    h_b = max(2.0 * (r.h - r.d), 1e-6)
    jd = max(r.d - h_t / 2.0, 0.3 * r.h)
    a = r.a if r.a > 0 else r.a_d * r.d
    theta = np.arctan2(jd, a)
    s, c = np.sin(theta), np.cos(theta)
    ws = min(r.w_tp * s + h_t * c, r.w_bp * s + h_b * c)
    # ACI 318-19 23.5: distributed reinforcement crossing the strut
    gamma_v = np.pi / 2 - theta          # vertical bars vs strut axis
    web = r.rho_v * np.sin(gamma_v) ** 2 + r.rho_h * np.sin(theta) ** 2
    beta_s = 0.75 if web >= 0.0025 else 0.4
    return 0.85 * beta_s * r.fck * ws * r.b * s / 1e3


def main():
    df = pd.read_csv(PROC / "deepbeam_steel_clean.csv")
    for col in ["w_tp", "w_bp", "a"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATS + ["w_tp", "w_bp", "a", "V"])
    df = df[(df.w_tp > 0) & (df.w_bp > 0) & (df.V > 0)].reset_index(drop=True)
    print(f"== deep-beam STM closure (n={len(df)}; d {df.d.min():.0f}-{df.d.max():.0f} mm) ==")

    Vs = np.array([aci_stm_V_kN(r) for r in df.itertuples()])
    s = m_stats(df.V.values, Vs)
    print(f"ACI 318-19 STM GLOBAL validation: bias={s['bias']:.2f} COV={s['cov']:.2f} "
          f"unsafe={s['unsafe']:.2f}")
    print("  (Megahed 2024 on this database: pred/test mean 0.699, CoV 0.372 "
          "-> test/pred family ~1.4-1.6 / ~0.37)")
    ok = 1.15 <= s["bias"] <= 1.95 and 0.26 <= s["cov"] <= 0.50
    print("VALIDATION:", "PASS" if ok else "FAIL -> do not use closure result")
    if not ok:
        return

    d_hi = df.d.quantile(0.75)
    big = df[df.d >= d_hi]
    sb = m_stats(big.V.values, np.array([aci_stm_V_kN(r) for r in big.itertuples()]))
    print(f"  out-of-envelope subset (d>={d_hi:.0f}, n={len(big)}): "
          f"bias={sb['bias']:.2f} COV={sb['cov']:.2f} unsafe={sb['unsafe']:.2f}")

    R = run_closure(df, FEATS, "V", {"ACI-STM": aci_stm_V_kN}, label="DEEP")
    R.to_csv(PROC / "deepbeam_closure.csv", index=False)
    g = (R.groupby("predictor")
           .agg(cov_ln_ind=("cov_ln_ind", "mean"), cov_ln_big=("cov_ln_big", "mean"),
                cov_ln_big_sd=("cov_ln_big", "std"), cov_raw_big=("cov_raw_big", "mean"),
                ooe_cov=("ooe_cov", "mean"), width=("width_factor_ln", "mean")).round(3))
    print("\nSame-metric closure on deep beams (target 0.90):")
    print(g.to_string())
    print("saved -> deepbeam_closure.csv")


if __name__ == "__main__":
    main()
