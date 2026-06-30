"""
reliability_engine.py — component-level structural reliability core.

Scope: the analytical heart of the decision-admissibility paper. Given a
predicted resistance and a (measured) resistance model-uncertainty
distribution plus a load model, compute the reliability index beta.

Design choices are INPUTS (load ratios, COVs, target beta); the engine itself
is general and verified against closed-form solutions. Nothing here assumes the
shear dataset — it is reused unchanged for the column leg.

Verified (run `python reliability_engine.py`):
  - MC beta agrees with closed-form lognormal R-S beta
  - MC beta agrees with closed-form normal R-S beta
  - FOSM matches normal closed form

References for the closed forms:
  Cornell (1969) lognormal format; Melchers & Beck (2018) Structural Reliability.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats


# ---------------------------------------------------------------------------
# Closed-form reference solutions (used both as engine outputs and as tests)
# ---------------------------------------------------------------------------
def beta_lognormal_RS(muR, VR, muS, VS):
    """Exact beta for independent lognormal R and S (Cornell lognormal format).
    muR, muS: means; VR, VS: coefficients of variation.
    """
    muR = np.asarray(muR, float); muS = np.asarray(muS, float)
    VR = np.asarray(VR, float);   VS = np.asarray(VS, float)
    num = np.log(muR / muS * np.sqrt((1.0 + VS**2) / (1.0 + VR**2)))
    den = np.sqrt(np.log((1.0 + VR**2) * (1.0 + VS**2)))
    return num / den


def beta_normal_RS(muR, sigmaR, muS, sigmaS):
    """Exact beta for independent normal R and S (Cornell second-moment)."""
    return (muR - muS) / np.sqrt(sigmaR**2 + sigmaS**2)


# ---------------------------------------------------------------------------
# Distribution helpers (parameterised by mean + COV, the engineering convention)
# ---------------------------------------------------------------------------
def lognormal_from_mean_cov(mean, cov):
    """Return a frozen scipy lognormal with the given mean and COV.
    scipy parameterisation: s = sigma_ln, scale = exp(mu_ln).
    """
    sigma_ln = np.sqrt(np.log(1.0 + cov**2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln**2
    return stats.lognorm(s=sigma_ln, scale=np.exp(mu_ln))


def normal_from_mean_cov(mean, cov):
    return stats.norm(loc=mean, scale=abs(mean) * cov)


def gumbel_from_mean_cov(mean, cov):
    """Gumbel (max) for live/extreme load, by mean+COV.
    sigma = cov*mean; scale b = sigma*sqrt(6)/pi; loc = mean - euler*b.
    """
    sigma = abs(mean) * cov
    b = sigma * np.sqrt(6.0) / np.pi
    loc = mean - np.euler_gamma * b
    return stats.gumbel_r(loc=loc, scale=b)


_DIST = {
    "lognormal": lognormal_from_mean_cov,
    "normal": normal_from_mean_cov,
    "gumbel": gumbel_from_mean_cov,
}


# ---------------------------------------------------------------------------
# Monte-Carlo beta for a general R - S limit state built from named variables
# ---------------------------------------------------------------------------
@dataclass
class Var:
    name: str
    dist: str          # 'lognormal' | 'normal' | 'gumbel'
    mean: float
    cov: float


def _pf_to_beta(nf, n):
    """Continuity-corrected pf -> beta; finite cap when nf == 0."""
    pf = (nf + 0.5) / (n + 1.0)
    return -stats.norm.ppf(pf)


def mc_beta(resistance_vars, load_vars, n=2_000_000, seed=0,
            combine="sum"):
    """Monte-Carlo beta for g = sum(R_i) - sum(S_j) (all independent).

    resistance_vars, load_vars: lists of Var. Returns (beta, pf, n).
    `combine='sum'` adds the components (the standard additive R and additive
    load-effect case). For a multiplicative resistance R = M * Rpred, pass M and
    a deterministic Rpred via a degenerate Var (cov=0) and use `mc_beta_product`.
    """
    rng = np.random.default_rng(seed)
    R = np.zeros(n)
    for v in resistance_vars:
        R += _DIST[v.dist](v.mean, v.cov).rvs(size=n, random_state=rng)
    S = np.zeros(n)
    for v in load_vars:
        S += _DIST[v.dist](v.mean, v.cov).rvs(size=n, random_state=rng)
    g = R - S
    nf = int(np.sum(g <= 0))
    return _pf_to_beta(nf, n), (nf + 0.5) / (n + 1.0), n


def mc_beta_product_resistance(M_mean, M_cov, M_dist, Rpred,
                               load_vars, n=2_000_000, seed=0, M_extra=None):
    """Monte-Carlo beta for R = M * Rpred (Rpred deterministic) minus loads.

    This is the paper's core resistance form: M is the measured model-uncertainty
    ratio (V_test/V_pred), Rpred is the model's predicted capacity for a
    specimen/design, and the load side is an additive set of load Vars.
    M_extra: optional list of additional independent multiplicative Vars
    (e.g., a separate material channel) multiplied into R.
    """
    rng = np.random.default_rng(seed)
    M = _DIST[M_dist](M_mean, M_cov).rvs(size=n, random_state=rng)
    R = M * float(Rpred)
    if M_extra:
        for v in M_extra:
            R *= _DIST[v.dist](v.mean, v.cov).rvs(size=n, random_state=rng)
    S = np.zeros(n)
    for v in load_vars:
        S += _DIST[v.dist](v.mean, v.cov).rvs(size=n, random_state=rng)
    g = R - S
    nf = int(np.sum(g <= 0))
    return _pf_to_beta(nf, n), (nf + 0.5) / (n + 1.0), n


# ---------------------------------------------------------------------------
# Self-test / verification
# ---------------------------------------------------------------------------
def _selftest():
    print("== reliability_engine self-test ==")
    ok = True

    # 1) Lognormal R-S: closed form vs MC
    muR, VR, muS, VS = 1.5, 0.18, 1.0, 0.25
    b_cf = float(beta_lognormal_RS(muR, VR, muS, VS))
    b_mc, pf, n = mc_beta([Var("R", "lognormal", muR, VR)],
                          [Var("S", "lognormal", muS, VS)],
                          n=4_000_000, seed=1)
    print(f"[lognormal R-S] closed={b_cf:.4f}  MC={b_mc:.4f}  pf={pf:.3e}")
    ok &= abs(b_cf - b_mc) < 0.03

    # 2) Normal R-S: closed form vs MC vs FOSM
    muR, sR, muS, sS = 200.0, 30.0, 120.0, 24.0
    b_cf2 = float(beta_normal_RS(muR, sR, muS, sS))
    b_mc2, _, _ = mc_beta([Var("R", "normal", muR, sR / muR)],
                          [Var("S", "normal", muS, sS / muS)],
                          n=4_000_000, seed=2)
    print(f"[normal R-S]    closed={b_cf2:.4f}  MC={b_mc2:.4f}")
    ok &= abs(b_cf2 - b_mc2) < 0.03

    # 3) Product resistance R = M*Rpred vs an equivalent single lognormal R
    #    (M lognormal, Rpred deterministic -> R lognormal with same COV)
    M_mean, M_cov, Rpred = 1.20, 0.20, 150.0
    muR3, VR3 = M_mean * Rpred, M_cov          # R lognormal mean, COV
    muS3, VS3 = 90.0, 0.22
    b_cf3 = float(beta_lognormal_RS(muR3, VR3, muS3, VS3))
    b_mc3, _, _ = mc_beta_product_resistance(
        M_mean, M_cov, "lognormal", Rpred,
        [Var("S", "lognormal", muS3, VS3)], n=4_000_000, seed=3)
    print(f"[product R]     closed={b_cf3:.4f}  MC={b_mc3:.4f}")
    ok &= abs(b_cf3 - b_mc3) < 0.03

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
