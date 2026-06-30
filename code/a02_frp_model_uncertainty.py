"""
a02_frp_model_uncertainty.py — honest characterisation on real FRP shear data.

log-target ML (fixes near-zero extrapolation), out-of-fold model uncertainty
M = V_test/V_pred, heteroscedasticity vs MULTIPLE regime variables (d, a/d, rho),
and ABSOLUTE vs RELATIVE conformal conditional coverage. Reports whatever is true.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from uq_utils import split_conformal_qhat, coverage, conditional_coverage, mondrian_qhat

PROC = Path("../data/processed")
FEATURES = ["bw", "d", "fc", "rho_l", "Ef", "a_d"]


def load_slender():
    df = pd.read_csv(PROC / "frp_clean.csv")
    df = df[(df["a_d"] >= 2.5) & (df["rho_l"] < 0.1)].copy()
    return df.dropna(subset=FEATURES + ["V_test_kN"]).reset_index(drop=True)


def oof_logpred(df, model, seed=0):
    X = df[FEATURES].values
    y = np.log(df["V_test_kN"].values)
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    return np.exp(cross_val_predict(model, X, y, cv=cv))


def tercile_regime(x, lo_name, mid_name, hi_name):
    q = np.quantile(x, [1/3, 2/3])
    return np.where(x < q[0], lo_name, np.where(x < q[1], mid_name, hi_name)), q


def cov_by(M, reg):
    out = []
    for g in sorted(pd.unique(reg)):
        m = reg == g
        out.append((g, int(m.sum()), float(np.mean(M[m])),
                    float(np.std(M[m], ddof=1)/np.mean(M[m]))))
    return pd.DataFrame(out, columns=["regime", "n", "bias", "cov"])


def main():
    df = load_slender(); n = len(df)
    print(f"== FRP slender beams, honest characterisation (n={n}) ==")
    models = {
        "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                  max_leaf_nodes=8, min_samples_leaf=15, random_state=0),
        "RF": RandomForestRegressor(n_estimators=800, min_samples_leaf=3,
                  max_features=0.7, random_state=0, n_jobs=-1),
    }
    dreg, qd = tercile_regime(df["d"].values, "1_small", "2_mid", "3_large")
    areg, qa = tercile_regime(df["a_d"].values, "1_lowad", "2_midad", "3_hiad")
    rreg, qr = tercile_regime(df["rho_l"].values, "3_lowrho", "2_midrho", "1_hirho")  # low rho = worse
    df["d_regime"] = dreg
    print(f"d terciles {qd.round(0)}, a/d terciles {qa.round(2)}, rho terciles {qr.round(4)}")

    for mname, model in models.items():
        pred = np.clip(oof_logpred(df, model), 1e-6, None)
        M = df["V_test_kN"].values / pred
        res = df["V_test_kN"].values - pred
        r2 = 1 - np.sum(res**2)/np.sum((df.V_test_kN-df.V_test_kN.mean())**2)
        print(f"\n#### {mname}: OOF R^2={r2:.3f}  bias(mean M)={np.mean(M):.3f}  global COV(M)={np.std(M,ddof=1)/np.mean(M):.3f}")

        for label, reg in [("SIZE d", dreg), ("a/d", areg), ("rho_l", rreg)]:
            t = cov_by(M, reg)
            lo = t.iloc[0]; hi = t.iloc[-1]
            print(f"  COV(M) by {label}: " +
                  " | ".join(f"{r.regime}:{r['cov']:.3f}(n{r.n})" for _, r in t.iterrows()) +
                  f"   [hi/lo={hi['cov']/lo['cov']:.2f}x]")
        # het stat vs d
        rho_s, p_s = stats.spearmanr(df["d"].values, np.abs(np.log(M)))
        print(f"  Spearman(|logM|, d): rho={rho_s:.3f} p={p_s:.4f}")

        # conformal: split cal/test; ABSOLUTE vs RELATIVE conditional coverage by size
        rng = np.random.default_rng(1); idx = rng.permutation(n)
        cal, test = idx[:n//2], idx[n//2:]
        # absolute
        qa_abs = split_conformal_qhat(res[cal], alpha=0.10)
        cc_abs = conditional_coverage(res[test], df["d_regime"].values[test], qa_abs)
        # relative (residual / prediction)
        rel = res / pred
        qa_rel = split_conformal_qhat(rel[cal], alpha=0.10)
        # coverage of relative interval = |rel| <= qa_rel, per regime
        rows = []
        for g in sorted(pd.unique(df["d_regime"].values[test])):
            m = df["d_regime"].values[test] == g
            rows.append((g, int(m.sum()), float(np.mean(np.abs(rel[test][m]) <= qa_rel))))
        cc_rel = pd.DataFrame(rows, columns=["regime", "n", "coverage"])
        # mondrian absolute
        qreg = mondrian_qhat(res[cal], df["d_regime"].values[cal], alpha=0.10)
        cc_mond = conditional_coverage(res[test], df["d_regime"].values[test], None, by_regime=qreg)

        print(f"  marginal coverage(test): abs={coverage(res[test],qa_abs):.3f} rel={np.mean(np.abs(rel[test])<=qa_rel):.3f}")
        merged = cc_abs[["regime","n","coverage"]].rename(columns={"coverage":"abs_cov"})
        merged["mondrian_cov"] = cc_mond["coverage"].values
        merged["relative_cov"] = cc_rel["coverage"].values
        print("  conditional coverage by SIZE (target 0.90):")
        print("   " + merged.to_string(index=False).replace("\n","\n   "))


if __name__ == "__main__":
    main()
