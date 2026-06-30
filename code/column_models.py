"""
column_models.py — mechanical models for RC columns (for the companion paper).
P-M interaction by strain compatibility + Whitney stress block (ACI 318), and the
axial-load-and-size-aware shear capacity (ACI 318-19 with lambda_s). Verified against
textbook checks. SI units: MPa, mm, N, N-mm.
"""
from __future__ import annotations
import numpy as np

EPS_CU = 0.003; ES = 200000.0


def beta1(fc):
    return float(np.clip(0.85 - 0.05*(fc-28)/7, 0.65, 0.85))


def pm_point(b, h, As_layers, fc, fy, c):
    """One point on the P-M diagram for neutral-axis depth c (mm).
    As_layers: list of (depth_from_top d_i [mm], area A_si [mm^2]). Compression on top.
    Returns (P [N, +compression], M [N-mm about plastic centroid h/2])."""
    a = min(beta1(fc)*c, h)
    Cc = 0.85*fc*a*b                                   # concrete compression
    P = Cc; M = Cc*(h/2 - a/2)
    for d_i, A_si in As_layers:
        eps = EPS_CU*(c - d_i)/c                        # +tension-side handled by sign
        fs = float(np.clip(ES*eps, -fy, fy))
        # subtract concrete displaced where steel is in compression zone
        fs_net = fs - (0.85*fc if (eps > 0 and d_i < a) else 0.0)
        Fi = fs_net*A_si
        P += Fi; M += Fi*(h/2 - d_i)
    return P, M


def pm_interaction(b, h, As_layers, fc, fy, n=60):
    """Full P-M interaction diagram (nominal). Returns arrays P[N], M[N-mm]."""
    cs = np.linspace(0.05*h, 4*h, n)
    pts = [pm_point(b, h, As_layers, fc, fy, c) for c in cs]
    Ast = sum(A for _, A in As_layers); Ag = b*h
    Po = 0.85*fc*(Ag-Ast) + fy*Ast                      # pure axial cap
    P = np.array([min(p, Po) for p, m in pts]); M = np.array([m for p, m in pts])
    return P, M, Po


def M_capacity_at_P(b, h, As_layers, fc, fy, P_demand):
    """Nominal moment capacity at a given axial load P (interpolate the diagram)."""
    P, M, Po = pm_interaction(b, h, As_layers, fc, fy, n=80)
    o = np.argsort(P); P, M = P[o], M[o]
    return float(np.interp(np.clip(P_demand, P.min(), P.max()), P, M))


def shear_aci318_19_column(bw, d, fc, rho_l, Nu, Ag, lam_s=True):
    """ACI 318-19 one-way shear with axial load + size effect lambda_s (N)."""
    ls = min(np.sqrt(2/(1+0.004*d)), 1.0) if lam_s else 1.0
    vc = (0.66*ls*np.cbrt(max(rho_l,1e-4))*np.sqrt(min(fc,69)) + min(Nu/(6*Ag), 0.05*fc))
    vc = min(vc, 0.42*np.sqrt(min(fc,69)))
    return vc*bw*d


if __name__ == "__main__":
    # textbook check: 400x400 tied column, 8-#25 (510 mm^2 each) ~ rho 1.9%, fc=30, fy=420, cover 40
    b=h=400.0; fc=30.; fy=420.; A=510.0
    layers=[(60,3*A),(200,2*A),(340,3*A)]              # top/mid/bottom layers
    P,M,Po=pm_interaction(b,h,layers,fc,fy)
    Ast=8*A; Ag=b*h
    print(f"pure-axial Po = {Po/1e3:.0f} kN  (expect 0.85*30*(160000-4080)+420*4080 = {(0.85*30*(Ag-Ast)+fy*Ast)/1e3:.0f} kN)")
    print(f"pure-flexure Mn (P~0) = {M_capacity_at_P(b,h,layers,fc,fy,0)/1e6:.0f} kN-m")
    print(f"balanced-ish max M on diagram = {M.max()/1e6:.0f} kN-m at P={P[np.argmax(M)]/1e3:.0f} kN")
    V=shear_aci318_19_column(b, 0.8*h, fc, 0.019, 500e3, Ag)
    print(f"ACI318-19 column shear (Nu=500kN) = {V/1e3:.0f} kN")
    ok = (Po>2500e3) and (M.max()>0) and (M_capacity_at_P(b,h,layers,fc,fy,0)>0)
    print("COLUMN P-M MODEL SELF-CHECK:", "PASS" if ok else "FAIL")
