"""
a12_mech_models.py — field-standard mechanical shear models (verified constants).
Each returns shear capacity Vc [N] for a beam without stirrups (SI: MPa, mm, N).
Constants verified against primary sources (Bentz 2006; Muttoni 2008; Bazant-Kim
1984; Zsutty 1968/71; fib MC2010; ACI 318). epsilon_x iteration where needed.

Size-effect-AWARE: MCFT, CSCT, Bazant-Kim, fib-MC2010, ACI318-19(lambda_s).
NO size effect:    ACI318-14, Zsutty.
"""
from __future__ import annotations
import numpy as np
Es = 200000.0


def _sqrt_fc(fc): return np.sqrt(min(fc, 64.0))      # sqrt(f'c)<=8 MPa cap


def _eps_x(V, M, d, As, dv):
    return max((M/dv + V)/(2*Es*As), 0.0)


def mcft(bw, d, a_d, rho, fc, dg=16.0):
    """Simplified MCFT (Bentz/Vecchio/Collins 2006): vc=beta*sqrt(fc); beta=0.4/(1+1500 ex)*1300/(1000+sxe)."""
    dv = 0.9*d; As = rho*bw*d; a = a_d*d
    sx = dv; ag = 0.0 if fc > 70 else dg
    sxe = max(35*sx/(ag+16), 0.85*sx)
    V = 0.3*_sqrt_fc(fc)*bw*dv
    for _ in range(40):
        M = max(V*(a-dv), V*dv)
        ex = _eps_x(V, M, d, As, dv)
        beta = (0.4/(1+1500*ex))*(1300.0/(1000+sxe))
        Vn = beta*_sqrt_fc(fc)*bw*dv
        Vn = min(Vn, 0.25*fc*bw*dv)
        if abs(Vn-V) < 1e-4*max(V,1): V=Vn; break
        V = 0.5*(V+Vn)
    return V


def csct(bw, d, a_d, rho, fc, dg=16.0):
    """Muttoni CSCT beam closed form: VR/(bw d sqrt fc)=1/3 * 1/(1+120 eps d/(16+dg))."""
    As = rho*bw*d; a = a_d*d; ddg = min(16+dg, 40.0)
    V = 0.3*_sqrt_fc(fc)*bw*d
    for _ in range(40):
        M = max(V*(a-0.9*d), V*0.5*d)
        eps = max(M/(0.9*d*Es*As), 1e-6)
        VR = (1.0/3.0)/(1+120*eps*d/ddg)*_sqrt_fc(fc)*bw*d
        if abs(VR-V) < 1e-4*max(V,1): V=VR; break
        V = 0.5*(V+VR)
    return V


def bazant_kim(bw, d, a_d, rho, fc, da=16.0):
    """Bazant-Kim 1984 (SI restatement): vu=0.83 rho^0.375 [sqrt fc +249 sqrt(rho/(a/d)^5)]/sqrt(1+d/(25 da))."""
    vu = 0.83*rho**0.375*(_sqrt_fc(fc) + 249*np.sqrt(rho/a_d**5))/np.sqrt(1+d/(25*da))
    return vu*bw*d


def zsutty(bw, d, a_d, rho, fc):
    """Zsutty 1968/71 slender (MPa, k=2.17) — NO size effect."""
    return 2.17*(fc*rho*d/(a_d*d)*d)**(1/3)*bw*d / 1.0 if False else 2.17*(fc*rho/a_d)**(1/3)*bw*d


def aci318_14(bw, d, fc):
    return 0.17*_sqrt_fc(fc)*bw*d


def aci318_19(bw, d, rho, fc):
    ls = min(np.sqrt(2/(1+0.004*d)), 1.0)
    vc = 0.66*ls*np.cbrt(rho)*_sqrt_fc(fc)
    return min(vc, 0.42*_sqrt_fc(fc))*bw*d


def fib_mc2010(bw, d, a_d, rho, fc, dg=16.0):
    dv = 0.9*d; As = rho*bw*d; a = a_d*d
    kdg = max(32.0/(16+(0.0 if fc>70 else dg)), 0.75)
    V = 0.3*_sqrt_fc(fc)*bw*dv
    for _ in range(40):
        M = max(V*(a-dv), V*dv)
        ex = _eps_x(V, M, d, As, dv)
        kv = (0.4/(1+1500*ex))*(1300.0/(1000+kdg*dv))
        Vn = kv*_sqrt_fc(fc)*dv*bw
        if abs(Vn-V) < 1e-4*max(V,1): V=Vn; break
        V = 0.5*(V+Vn)
    return V


MODELS = {"ACI318-14": lambda b,d,ad,r,fc: aci318_14(b,d,fc),
          "Zsutty": lambda b,d,ad,r,fc: zsutty(b,d,ad,r,fc),
          "ACI318-19": lambda b,d,ad,r,fc: aci318_19(b,d,r,fc),
          "MCFT": mcft, "CSCT": csct, "fib-MC2010": fib_mc2010,
          "Bazant-Kim": lambda b,d,ad,r,fc: bazant_kim(b,d,ad,r,fc)}
SIZE_AWARE = {"ACI318-19","MCFT","CSCT","fib-MC2010","Bazant-Kim"}


if __name__ == "__main__":
    # reference beam sanity + size-effect slope per model
    bw,a_d,rho,fc=200.,3.,0.015,30.
    print("reference beam d=500, bw=200, a/d=3, rho=0.015, fc=30 -> Vc (kN) and v=Vc/(bw d) MPa:")
    for n,f in MODELS.items():
        V=f(bw,500,a_d,rho,fc); print(f"  {n:11s} Vc={V/1e3:6.1f} kN   v={V/(bw*500):.3f} MPa")
    print("\nlarge-size log-log slope of v=Vc/(bw d) (d 300->2000):")
    ds=np.array([300,500,800,1200,2000.])
    for n,f in MODELS.items():
        v=np.array([f(bw,d,a_d,rho,fc)/(bw*d) for d in ds])
        sl=np.polyfit(np.log(ds),np.log(v),1)[0]
        print(f"  {n:11s} slope={sl:+.2f}   {'(size-aware)' if n in SIZE_AWARE else '(no size effect)'}")
