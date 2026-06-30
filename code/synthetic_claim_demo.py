"""
synthetic_claim_demo.py — end-to-end DRY RUN of the paper's claim chain (F1-F4)
on synthetic data with KNOWN injected heteroscedasticity.

PURPOSE: prove the analysis machinery can recover the decision-admissibility
result *if* the real data has regime-dependent model uncertainty. This makes NO
claim about real data — it only validates that the pipeline is sound, so that on
real data the only open question is the empirical one (does F2 hold?).

Claim chain demonstrated:
  F1  global conformal reaches ~90% MARGINAL coverage.
  F2  model-uncertainty COV is much larger in the 'large' regime (injected).
  F3  a reliability decision calibrated with the GLOBAL COV overstates beta in
      the large regime -> false-admissions (designs accepted that are actually
      below target beta when their true regime COV is used).
  F4  a regime-conditioned (decision-calibrated) COV removes the false-admissions.

Uses the verified reliability_engine + uq_utils.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from reliability_engine import beta_lognormal_RS
from uq_utils import (split_conformal_qhat, coverage, conditional_coverage,
                      mondrian_qhat, regime_model_uncertainty)

BETA_T = 3.8          # target reliability index (demo)
VS = 0.15             # combined dead+live load-effect COV (demo, lognormal S)


def design_load_mean(Rpred, bias, cov_M):
    """Mean load S that puts the design exactly at BETA_T, using a lognormal
    R = M*Rpred (mean bias*Rpred, COV cov_M) and lognormal S (COV VS).
    Closed-form inverse of the Cornell lognormal index."""
    VR = cov_M
    A = np.sqrt((1 + VS**2) / (1 + VR**2))
    B = np.sqrt(np.log((1 + VR**2) * (1 + VS**2)))
    return bias * Rpred * A / np.exp(BETA_T * B)


def true_beta(Rpred, muS, bias_r, cov_r):
    """True beta of a design with load mean muS when the REAL (regime) model
    uncertainty (bias_r, cov_r) is used."""
    return float(beta_lognormal_RS(bias_r * Rpred, cov_r, muS, VS))


def main():
    print("== synthetic end-to-end claim dry run (F1-F4) ==")
    rng = np.random.default_rng(7)
    n = 8000
    size = rng.uniform(0, 1, n)
    cov_true = 0.12 + 0.38 * size                 # injected heteroscedasticity
    y_pred = rng.uniform(60, 220, n)
    y_true = y_pred * (1.0 + rng.normal(0, cov_true))
    regime = np.where(size < 0.5, "small", "large")

    idx = rng.permutation(n); cal, test = idx[:4000], idx[4000:]
    res = y_true - y_pred

    # --- F1: global marginal coverage ---
    q_glob = split_conformal_qhat(res[cal], alpha=0.10)
    cov_marg = coverage(res[test], q_glob)

    # --- F2: regime model uncertainty ---
    mu_all = regime_model_uncertainty(y_true, y_pred, regime).set_index("regime")
    bias_g, cov_g = float(mu_all["bias"].mean()), None
    # global (regime-blind) model uncertainty:
    M = y_true / y_pred
    bias_glob = float(np.mean(M)); cov_glob = float(np.std(M, ddof=1) / np.mean(M))

    # --- conditional coverage (global interval) shows the gap ---
    cc = conditional_coverage(res[test], regime[test], q_glob)

    # --- F3 / F4: false-admission rate, global-COV vs regime-COV calibration ---
    rows = []
    for calib in ("global", "regime"):
        fa = {"small": 0, "large": 0}; tot = {"small": 0, "large": 0}
        for g in ("small", "large"):
            m = (regime[test] == g)
            Rp = y_pred[test][m]
            bias_r = float(mu_all.loc[g, "bias"]); cov_r = float(mu_all.loc[g, "cov"])
            if calib == "global":
                # engineer designs using global (regime-blind) bias+COV
                muS = design_load_mean(Rp, bias_glob, cov_glob)
            else:
                # decision-calibrated: uses the regime's own bias+COV
                muS = design_load_mean(Rp, bias_r, cov_r)
            bt = np.array([true_beta(Rp[i], muS[i], bias_r, cov_r)
                           for i in range(len(Rp))])
            fa[g] = int(np.sum(bt < BETA_T - 1e-9)); tot[g] = len(Rp)
        rows.append({"calibration": calib,
                     "false_admit_small": fa["small"] / tot["small"],
                     "false_admit_large": fa["large"] / tot["large"]})
    fa_df = pd.DataFrame(rows)

    print(f"\nF1 global qhat={q_glob:.2f}  MARGINAL coverage={cov_marg:.3f} (target 0.90)")
    print("\nF2 model-uncertainty by regime:")
    print(mu_all.reset_index()[["regime", "n", "bias", "cov"]].to_string(index=False))
    print(f"   (global regime-blind: bias={bias_glob:.3f}, cov={cov_glob:.3f})")
    print("\nGLOBAL interval conditional coverage (the misleading metric):")
    print(cc.to_string(index=False))
    print("\nF3/F4 false-admission rate (design accepted but true beta < target):")
    print(fa_df.to_string(index=False))

    # validation checks
    glob_row = fa_df.set_index("calibration").loc["global"]
    reg_row = fa_df.set_index("calibration").loc["regime"]
    ok = (abs(cov_marg - 0.90) < 0.03
          and float(mu_all.loc["large", "cov"]) > 1.5 * float(mu_all.loc["small", "cov"])
          and glob_row["false_admit_large"] > 0.25         # global mis-admits large
          and reg_row["false_admit_large"] < 0.10)          # regime calib fixes it
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
