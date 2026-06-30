"""
damage_fe.py — nonlinear crack-band damage FE for RC beam shear (skfem).

Isotropic Rankine (max principal tensile strain) damage with CRACK-BAND
regularisation: the softening strain kappa_f is scaled by element size h so the
fracture energy Gf dissipated per crack area is mesh-objective (Bazant crack
band). This is what makes the SIZE EFFECT emerge from mechanics rather than as a
mesh artefact. Longitudinal steel is a non-damaging stiffened bottom layer to
force web diagonal-shear cracking. Displacement control + secant iteration tracks
the peak load = shear capacity.

shear_capacity_fe(d,...) returns peak nominal shear stress v = V/(b*d).
"""
from __future__ import annotations
import numpy as np
from skfem import (MeshQuad, Basis, ElementVector, ElementQuad1, ElementQuad0,
                   BilinearForm, asm, condense, solve)
from skfem.helpers import sym_grad, ddot, trace


def shear_capacity_fe(d=300.0, a_d=3.0, rho=0.02, fc=30.0, ag=16.0, b=200.0,
                      n_d=16, steps=60, verbose=False, gf_scale=1.0):
    # geometry: shear span a = a_d*d on the left; total span ~ 2a + load plate
    a = a_d * d; L = 2.2 * a; h = 1.15 * d
    nx = max(40, int(round(n_d * L / h))); ny = n_d
    m = MeshQuad.init_tensor(np.linspace(0, L, nx + 1), np.linspace(0, h, ny + 1))
    basis = Basis(m, ElementVector(ElementQuad1()))
    d0b = Basis(m, ElementQuad0())                 # piecewise-constant fields
    p = m.p; ec = m.p[:, m.t].mean(1)              # element centroids (2, nelem)
    nelem = m.t.shape[1]

    Ec = 4700 * np.sqrt(fc); nu = 0.2
    mu = Ec / (2 * (1 + nu)); lam = Ec * nu / ((1 + nu) * (1 - 2 * nu))
    lam = 2 * lam * mu / (lam + 2 * mu)            # plane stress
    ft = 0.33 * np.sqrt(fc); Gf = gf_scale * 0.073 * fc ** 0.18   # N/mm (fib), scalable for robustness
    k0 = ft / Ec
    h_el = h / ny                                  # crack-band width
    kf = max(2.2 * k0, 2 * Gf / (ft * h_el))       # linear softening, mesh-objective

    # steel: bottom row elements -> stiffened, non-damaging
    bot = ec[1] < h_el * 1.01
    steel_fac = np.ones(nelem)
    Es_over_Ec = 200000.0 / Ec
    steel_fac[bot] = 1.0 + rho * (h / h_el) * (Es_over_Ec - 1.0)   # smeared bar stiffness in bottom band
    # bearing zones (no damage) under the load and over the supports to avoid spurious
    # stress-concentration cracking; width ~ 1 element each side
    plate = 1.5 * h_el
    no_dmg = (bot |
              ((np.abs(ec[0]-a) < plate) & (ec[1] > h - 2*plate)) |    # under load
              ((ec[0] < plate) & (ec[1] < 2*plate)) |                  # left support
              ((ec[0] > L-plate) & (ec[1] < 2*plate)))                 # right support

    nqp = 9                                        # ElementQuad1 default 3x3 Gauss
    nd = basis.nodal_dofs
    def node(x0, y0): return int(np.argmin((p[0]-x0)**2 + (p[1]-y0)**2))
    n_pin, n_roll = node(0, 0), node(L, 0)
    n_load = node(a, h)
    D_supp = [nd[0, n_pin], nd[1, n_pin], nd[1, n_roll]]
    ld = nd[1, n_load]

    dmg = np.zeros(nelem); kappa = np.full(nelem, k0)
    @BilinearForm
    def stiff(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        return b * w['fac'] * (2*mu*ddot(eu, ev) + lam*trace(eu)*trace(ev))

    def elem_principal_tensile_strain(u):
        eps = sym_grad(basis.interpolate(u))       # (2,2,nelem,nqp)
        exx = eps[0,0].mean(1); eyy = eps[1,1].mean(1); exy = eps[0,1].mean(1)
        c = 0.5*(exx+eyy); r = np.sqrt((0.5*(exx-eyy))**2 + exy**2)
        return c + r                               # max principal strain

    def damage_of(kap):
        dd = np.zeros_like(kap); m1 = (kap > k0) & (kap < kf)
        dd[m1] = 1 - (k0/kap[m1]) * (kf - kap[m1])/(kf - k0)
        dd[kap >= kf] = 0.999
        return np.clip(dd, 0, 0.999)

    Vpeak = 0.0; hist = []
    dmax = 0.012 * d                               # max imposed deflection
    for s in range(1, steps + 1):
        delta = -dmax * s / steps
        for it in range(8):                        # secant iteration on damage
            fac = ((1 - dmg) * steel_fac)
            facfield = np.tile(fac[:, None], (1, nqp))
            K = asm(stiff, basis, fac=facfield)
            # displacement control: prescribe ld = delta
            x = np.zeros(K.shape[0]); x[ld] = delta
            u = solve(*condense(K, x=x, D=np.array(D_supp + [ld])))
            eps1 = elem_principal_tensile_strain(u)
            kap_new = np.maximum(kappa, eps1)
            dmg_new = damage_of(kap_new); dmg_new[no_dmg] = 0.0
            if np.max(np.abs(dmg_new - dmg)) < 1e-3:
                dmg = dmg_new; kappa = kap_new; break
            dmg = dmg_new; kappa = kap_new
        # reaction at load node = (K u)_ld with current (converged) K
        R = float((K @ u)[ld])
        V = abs(R)                                  # applied shear ~ reaction (point load near support)
        hist.append((abs(delta), V, float(dmg.max())))
        Vpeak = max(Vpeak, V)
        if V < 0.5 * Vpeak and dmg.max() > 0.9:     # post-peak softening -> stop
            break
    v_nom = Vpeak / (b * d)                          # nominal shear stress MPa (V in N? -> need N)
    # NOTE V is in N because Ec in MPa, lengths mm, delta mm -> force N
    if verbose:
        print(f" d={d:.0f} a/d={a_d} rho={rho} fc={fc}: Vpeak={Vpeak/1e3:.1f} kN, v_nom={v_nom:.3f} MPa, "
              f"kf/k0={kf/k0:.1f}, steps={len(hist)}, dmax_dmg={hist[-1][2]:.2f}")
    return v_nom, Vpeak, hist


if __name__ == "__main__":
    # test on one beam: does it crack, soften, and reach a peak?
    v, V, hist = shear_capacity_fe(d=300, a_d=3.0, rho=0.02, fc=30, n_d=16, steps=50, verbose=True)
    print("load-deflection (first/peak/last):")
    H = np.array(hist)
    ip = int(np.argmax(H[:,1]))
    for i in [0, ip, len(H)-1]:
        print(f"  delta={H[i,0]:.3f} mm  V={H[i,1]/1e3:.1f} kN  dmax={H[i,2]:.2f}")
    print("PEAK FOUND + SOFTENING" if ip < len(H)-1 and H[-1,1] < H[ip,1] else "no clear softening (check)")
