"""
a06_reliability_consequence.py — engineering payload + remedy on real data.

(1) CONSEQUENCE: a member certified at target reliability beta_T using the
    CV-calibrated ML model uncertainty actually has reliability beta_real under
    the TRUE size-extrapolation model uncertainty. We report beta_real and the
    one-sided design-value (5%-fractile) violation rate.
(2) REMEDY: an admissibility GATE that does NOT use the size variable d — a kNN
    out-of-distribution score in standardised feature space, calibrated on the
    training pool. We show it flags the failing large members (high abstention on
    the extrapolation set) while abstaining little in-domain (retained coverage
    recovers). i.e. it tells the engineer 'ML UQ untrustworthy here -> use the
    mechanical model', instead of the false reassurance of 90% CV coverage.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from reliability_engine import beta_lognormal_RS
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed")
BETA_T, VS = 3.8, 0.15


def design_then_realized_beta(b_cv, c_cv, b_ex, c_ex):
    """beta actually realised by a member designed to BETA_T with (b_cv,c_cv)
    but whose true model uncertainty is (b_ex,c_ex). Resistance lognormal R=M*Rpred,
    load lognormal (COV VS). Rpred cancels."""
    A = np.sqrt((1 + VS**2) / (1 + c_cv**2)); B = np.sqrt(np.log((1 + c_cv**2) * (1 + VS**2)))
    muS = b_cv * A / np.exp(BETA_T * B)             # load mean (Rpred=1)
    return float(beta_lognormal_RS(b_ex, c_ex, muS, VS))


def m_stats(y, pred):
    M = y / np.clip(pred, 1e-9, None)
    return float(np.mean(M)), float(np.std(M, ddof=1) / np.mean(M))


def run(name, df, feats, target, seed=0):
    df = df.sort_values("d").reset_index(drop=True)
    d_hi = df["d"].quantile(0.75)
    pool = df[df.d < d_hi].reset_index(drop=True)
    test = df[df.d >= d_hi].reset_index(drop=True)
    Xp, yp = pool[feats].values, pool[target].values
    Xe, ye = test[feats].values, test[target].values

    models = {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=seed),
        "RF": RandomForestRegressor(n_estimators=600, min_samples_leaf=3,
                  max_features=0.7, random_state=seed, n_jobs=-1),
        "Linear": LinearRegression(),
    }
    print(f"\n================ {name} (pool n={len(pool)} d<{d_hi:.0f}; extrap n={len(test)}) ================")
    for mname, model in models.items():
        # interpolation model-uncertainty (CV OOF on pool)
        oof = np.exp(cross_val_predict(model, Xp, np.log(yp), cv=KFold(10, shuffle=True, random_state=seed)))
        b_cv, c_cv = m_stats(yp, oof)
        # fit, conformal q on pool, extrapolation stats
        model.fit(Xp, np.log(yp))
        pred_pool = np.exp(model.predict(Xp)); res_pool = yp - pred_pool
        q = split_conformal_qhat(res_pool, 0.10)
        ql = np.quantile(res_pool, 0.10)            # ~one-sided lower (10% below pred-> design fractile)
        pred_e = np.exp(model.predict(Xe)); res_e = ye - pred_e
        b_ex, c_ex = m_stats(ye, pred_e)
        cov_in = coverage(yp - oof, q)
        cov_ex = coverage(res_e, q)
        # one-sided design value L = pred + ql (ql<0): violation = ye < L
        L_e = pred_e + ql
        viol_ex = float(np.mean(ye < L_e))
        L_p = pred_pool + ql; viol_in = float(np.mean(yp < L_p))
        beta_real = design_then_realized_beta(b_cv, c_cv, b_ex, c_ex)
        print(f"  {mname:7s}: CV(M) bias={b_cv:.2f} cov={c_cv:.2f} | EXTRAP(M) bias={b_ex:.2f} cov={c_ex:.2f}")
        print(f"           coverage interp={cov_in:.2f} extrap={cov_ex:.2f} | 5%-fractile violation interp={viol_in:.2f} extrap={viol_ex:.2f}")
        print(f"           >> design CERTIFIED beta_T={BETA_T}  ->  REALISED beta under true extrap uncertainty = {beta_real:.2f}"
              + ("   (UNSAFE)" if beta_real < BETA_T - 0.3 else "   (over-conservative)" if beta_real > BETA_T + 0.3 else ""))

    # ---- REMEDY: OOD admissibility gate (kNN, does NOT use d) ----
    rng = np.random.default_rng(seed)
    # in-domain held-out from pool
    idx = rng.permutation(len(pool)); cut = int(0.7 * len(pool))
    tr, ind = idx[:cut], idx[cut:]
    mu, sd = Xp[tr].mean(0), Xp[tr].std(0) + 1e-9
    Z = (Xp[tr] - mu) / sd
    nn = NearestNeighbors(n_neighbors=6).fit(Z)
    def ood(Xq):
        Zq = (Xq - mu) / sd
        d, _ = nn.kneighbors(Zq)
        return d[:, 1:].mean(1)                      # mean dist to 5 nearest (excl self for tr)
    thr = np.quantile(ood(Xp[tr]), 0.95)
    s_ind, s_ext = ood(Xp[ind]), ood(Xe)
    ab_in = float(np.mean(s_ind > thr)); ab_ex = float(np.mean(s_ext > thr))
    print(f"  REMEDY (kNN OOD gate, no d): abstention IN-DOMAIN={ab_in:.2f}  vs  EXTRAPOLATION={ab_ex:.2f}"
          f"   (gate flags {ab_ex*100:.0f}% of the failing large members)")


def main():
    frp = pd.read_csv(PROC / "frp_clean.csv")
    frp = frp[(frp.a_d >= 2.5) & (frp.rho_l < 0.1)].dropna(
        subset=["bw", "d", "fc", "rho_l", "Ef", "a_d", "V_test_kN"])
    run("FRP slender (no stirrups)", frp,
        ["bw", "d", "fc", "rho_l", "Ef", "a_d"], "V_test_kN")

    st = pd.read_csv(PROC / "deepbeam_steel_clean.csv")
    run("STEEL deep beams", st,
        ["h", "d", "b", "a_d", "fck", "rho", "fy", "rho_v", "fyv", "rho_h", "fyh"], "V")


if __name__ == "__main__":
    main()
