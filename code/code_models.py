"""
code_models.py — design-code shear strength of RC beams WITHOUT stirrups (SI).

Mechanical baselines + the size-effect narrative spine:
  - ACI 318-14: Vc = 0.17 lambda sqrt(f'c) bw d   (NO size effect)
  - ACI 318-19: Vc = 0.66 lambda_s lambda (rho_w)^(1/3) sqrt(f'c) bw d
                lambda_s = sqrt(2/(1+0.004 d)) <= 1   (size effect, members w/o min stirrups)
  - fib MC2010 Level II approx (optional; strain-based kv) — added when needed.

Units: SI throughout — f'c [MPa], bw,d [mm], Vc [N]. sqrt(f'c) capped per code
(f'c effectively <= 64 MPa via sqrt(f'c) <= 8.3 MPa).

Constants hand-verified (see header calc); to be cross-checked against the
ACI-DAfStb database's published V_test/V_ACI ratio statistics once data lands.

Refs: ACI 318-14 Eq. 22.5.5.1; ACI 318-19 Table 22.5.5.1 + Eq. 22.5.5.1.3.
"""
from __future__ import annotations
import numpy as np


def _sqrt_fc(fc, cap=8.3):
    """sqrt(f'c) with the ACI cap sqrt(f'c) <= 8.3 MPa (f'c ~ 69 MPa)."""
    return np.minimum(np.sqrt(np.asarray(fc, float)), cap)


def lambda_s_aci19(d_mm):
    """ACI 318-19 size effect factor, d in mm."""
    return np.minimum(np.sqrt(2.0 / (1.0 + 0.004 * np.asarray(d_mm, float))), 1.0)


def vc_aci318_14(bw, d, fc, lam=1.0):
    """ACI 318-14 simplified Vc (N). No size effect — unconservative for large d."""
    bw = np.asarray(bw, float); d = np.asarray(d, float)
    return 0.17 * lam * _sqrt_fc(fc) * bw * d


def vc_aci318_19(bw, d, fc, rho_w, lam=1.0, Nu=0.0, Ag=None):
    """ACI 318-19 Vc (N) for members WITHOUT at least minimum shear reinforcement
    (Table 22.5.5.1, the (rho_w)^(1/3) row with size-effect lambda_s).
    rho_w: longitudinal reinforcement ratio As/(bw d). Nu axial (N, +comp), Ag (mm^2).
    """
    bw = np.asarray(bw, float); d = np.asarray(d, float)
    rho = np.clip(np.asarray(rho_w, float), 1e-6, None)
    ls = lambda_s_aci19(d)
    vc = 0.66 * ls * lam * np.cbrt(rho) * _sqrt_fc(fc) * bw * d
    if Nu and Ag is not None:
        vc = vc + (Nu / (6.0 * np.asarray(Ag, float))) * bw * d
    # ACI cap: Vc <= 0.42 sqrt(f'c) bw d (5 sqrt(f'c) psi)
    vc = np.minimum(vc, 0.42 * _sqrt_fc(fc) * bw * d)
    return vc


def _selftest():
    print("== code_models self-test (hand-checked case) ==")
    bw, d, fc, rho = 300.0, 500.0, 30.0, 0.02
    v14 = vc_aci318_14(bw, d, fc) / 1e3
    v19 = vc_aci318_19(bw, d, fc, rho) / 1e3
    ls = float(lambda_s_aci19(d))
    print(f"ACI318-14 Vc = {v14:.1f} kN  (expect ~139.7)")
    print(f"ACI318-19 Vc = {v19:.1f} kN  (expect ~120.2), lambda_s={ls:.4f} (expect 0.8165)")
    # size effect: lambda_s must fall with d
    ls_small, ls_big = float(lambda_s_aci19(150)), float(lambda_s_aci19(1500))
    mono = ls_small > ls_big
    ok = (abs(v14 - 139.7) < 1.0 and abs(v19 - 120.2) < 1.5
          and abs(ls - 0.8165) < 0.001 and mono)
    print(f"lambda_s(150mm)={ls_small:.3f} > lambda_s(1500mm)={ls_big:.3f}: {mono}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
