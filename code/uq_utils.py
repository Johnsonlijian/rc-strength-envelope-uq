"""
uq_utils.py — uncertainty-quantification + model-uncertainty utilities.

Conformal prediction (global + regime-conditioned / Mondrian) and resistance
model-uncertainty characterisation M = y_true / y_pred. Standard, dataset-
agnostic; reused for the shear and column legs.

The distinction this paper turns on:
  - MARGINAL coverage: averaged over all specimens (what UQ papers report).
  - CONDITIONAL coverage: within a regime (what a reliability decision needs).
A globally coverage-calibrated interval can have good marginal coverage and bad
conditional coverage in a safety-critical regime; these functions expose that.

Verified by `python uq_utils.py` on synthetic data with KNOWN injected
heteroscedasticity (method validation only — no claim about real data).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Split conformal
# ---------------------------------------------------------------------------
def split_conformal_qhat(residuals_cal, alpha=0.10):
    """Split-conformal half-width: the (1-alpha) conformal quantile of the
    absolute calibration residuals, with the finite-sample (n+1) correction.
    residuals_cal: array of (y_cal - yhat_cal).
    """
    r = np.abs(np.asarray(residuals_cal, float))
    n = r.size
    # conformal level with finite-sample correction, capped at 1.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, level, method="higher"))


def coverage(residuals_test, qhat):
    """Empirical coverage of a symmetric interval of half-width qhat."""
    r = np.abs(np.asarray(residuals_test, float))
    return float(np.mean(r <= qhat))


def mondrian_qhat(residuals_cal, regime_cal, alpha=0.10):
    """Regime-conditioned (Mondrian) conformal half-widths: one qhat per regime
    label. Returns dict regime -> qhat. Falls back to global where a regime is
    too small (n < 1/alpha rows can't reach the level)."""
    residuals_cal = np.asarray(residuals_cal, float)
    regime_cal = np.asarray(regime_cal)
    glob = split_conformal_qhat(residuals_cal, alpha)
    out = {}
    for g in pd.unique(regime_cal):
        m = regime_cal == g
        if m.sum() >= int(np.ceil(1.0 / alpha)):
            out[g] = split_conformal_qhat(residuals_cal[m], alpha)
        else:
            out[g] = glob
    return out


def conditional_coverage(residuals_test, regime_test, qhat, by_regime=None):
    """Coverage within each regime. qhat is a scalar (global interval) or, if
    by_regime is a dict, the per-regime half-width is used. Returns a DataFrame.
    """
    residuals_test = np.asarray(residuals_test, float)
    regime_test = np.asarray(regime_test)
    rows = []
    for g in pd.unique(regime_test):
        m = regime_test == g
        q = (by_regime[g] if by_regime is not None else qhat)
        rows.append({"regime": g, "n": int(m.sum()),
                     "qhat": float(q),
                     "coverage": coverage(residuals_test[m], q)})
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Resistance model uncertainty  M = y_true / y_pred
# ---------------------------------------------------------------------------
def model_uncertainty(y_true, y_pred):
    """Return (bias, cov) of M = y_true / y_pred (the reliability convention).
    bias = mean(M) (model conservatism: >1 means model under-predicts capacity);
    cov  = std(M)/mean(M).
    """
    M = np.asarray(y_true, float) / np.asarray(y_pred, float)
    M = M[np.isfinite(M) & (M > 0)]
    mu = float(np.mean(M)); sd = float(np.std(M, ddof=1))
    return mu, sd / mu


def regime_model_uncertainty(y_true, y_pred, regime):
    """Per-regime model-uncertainty (bias, cov, n). The empirical core of F2:
    does COV grow in the safety-critical regime?"""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    regime = np.asarray(regime)
    rows = []
    for g in pd.unique(regime):
        m = regime == g
        mu, cov = model_uncertainty(y_true[m], y_pred[m])
        rows.append({"regime": g, "n": int(m.sum()), "bias": mu, "cov": cov})
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-test on synthetic data with KNOWN injected heteroscedasticity
# ---------------------------------------------------------------------------
def _selftest():
    print("== uq_utils self-test (synthetic, known ground truth) ==")
    rng = np.random.default_rng(0)
    n = 6000
    size = rng.uniform(0, 1, n)                  # a 'size' covariate in [0,1]
    # injected model error: COV of multiplicative error grows with size
    cov_true = 0.10 + 0.40 * size                # 0.10 small -> 0.50 large
    y_pred = rng.uniform(50, 200, n)
    err = rng.normal(0, cov_true) * y_pred       # heteroscedastic abs error
    y_true = y_pred + err

    # split cal/test
    idx = rng.permutation(n); cal, test = idx[:3000], idx[3000:]
    res_cal = (y_true - y_pred)[cal]
    res_test = (y_true - y_pred)[test]

    # regime: bottom vs top size tercile
    def regime_of(s):
        return np.where(s < 0.5, "small", "large")
    reg_cal, reg_test = regime_of(size[cal]), regime_of(size[test])

    # global conformal
    q_glob = split_conformal_qhat(res_cal, alpha=0.10)
    cov_marg = coverage(res_test, q_glob)
    cc_glob = conditional_coverage(res_test, reg_test, q_glob)

    # mondrian conformal
    q_reg = mondrian_qhat(res_cal, reg_cal, alpha=0.10)
    cc_mond = conditional_coverage(res_test, reg_test, None, by_regime=q_reg)

    print(f"global qhat={q_glob:.2f}  MARGINAL coverage={cov_marg:.3f} (target 0.90)")
    print("GLOBAL interval, conditional coverage by regime:")
    print(cc_glob.to_string(index=False))
    print("MONDRIAN interval, conditional coverage by regime:")
    print(cc_mond.to_string(index=False))

    mu = regime_model_uncertainty(y_true, y_pred, regime_of(size))
    print("model-uncertainty by regime (bias, cov):")
    print(mu.to_string(index=False))

    # checks: global interval under-covers the 'large' regime; mondrian fixes it
    glob_large = cc_glob.set_index("regime").loc["large", "coverage"]
    mond_large = cc_mond.set_index("regime").loc["large", "coverage"]
    cov_large = mu.set_index("regime").loc["large", "cov"]
    cov_small = mu.set_index("regime").loc["small", "cov"]
    ok = (abs(cov_marg - 0.90) < 0.03) and (glob_large < 0.85) \
        and (mond_large > glob_large) and (cov_large > 1.5 * cov_small)
    print("RESULT:", "PASS" if ok else "FAIL",
          f"(glob_large={glob_large:.3f} -> mond_large={mond_large:.3f}; "
          f"cov large/small={cov_large:.2f}/{cov_small:.2f})")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
