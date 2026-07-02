"""
a33_anchor_robustness.py — is the residual-target recovery anchor-specific?
Repeats the a29 residual-target arm on the canonical steel benchmark with a
DIFFERENT mechanical anchor (ACI 318-19 instead of MCFT). If the recovery
persists, the conclusion "the physics must carry the extrapolation" is a
property of the construction, not of one favoured model.

Outputs -> ../data/processed/anchor_robustness.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
from a29_physics_arms import run
from a12_mech_models import aci318_19

PROC = Path("../data/processed")


def main():
    steel = pd.read_csv(PROC / "steel_zhang_clean.csv")
    steel = steel[steel.a_d >= 2.5].dropna(
        subset=["bw", "d", "a_d", "rho", "fc", "ag", "fy", "Vu_kN"]).reset_index(drop=True)
    steel["V31819_kN"] = [aci318_19(r.bw, r.d, r.rho, r.fc) / 1e3
                          for r in steel.itertuples()]
    A, _ = run("STEEL-a19", steel, ["bw", "d", "a_d", "rho", "fc", "ag", "fy"],
               "Vu_kN", "V31819_kN")
    A = A[A.arm == "residual-target"].copy()
    A.to_csv(PROC / "anchor_robustness.csv", index=False)
    g = (A.groupby("model")
           .agg(cov_ind=("cov_ind", "mean"), cov_big=("cov_big", "mean"),
                ooe_cov=("ooe_cov", "mean"), width=("width_factor", "mean")).round(3))
    print("== residual-target arm, anchor = ACI 318-19 (steel) ==")
    print(g.to_string())
    print("(compare MCFT anchor: cov_big 0.830/0.860, ooe_cov 0.156/0.144)")
    print("saved -> anchor_robustness.csv")


if __name__ == "__main__":
    main()
