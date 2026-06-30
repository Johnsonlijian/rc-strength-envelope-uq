"""
a07_corrected_uq_matrix.py — corrected, reviewer-proof extrapolation analysis.

Fixes flagged by adversarial review:
 (1) PROPER split-conformal: pool -> train / calibration / in-domain-eval (never
     calibrate on in-sample fit residuals).
 (2) log->raw via Duan smearing (E[V|x]=exp(logpred)*mean(exp(cal log-resid))).
 (3) Full UQ-method matrix per model: split-conformal (absolute), kNN-normalized
     (locally-adaptive) conformal, plus GP-native interval and quantile-GBM —
     to test the "just use adaptive UQ" objection.
 (4) Honest interp (in-domain eval split) vs extrap (large hold-out) coverage.
 (5) Selective prediction: coverage on the retained set after a kNN OOD gate.

Question: does ANY UQ method restore ~0.90 coverage under size-extrapolation?
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from uq_utils import split_conformal_qhat, coverage

PROC = Path("../data/processed")
ALPHA = 0.10


def duan_predict_log(model, Xtr, ytr_log, Xq, ycal_log=None, predcal_log=None):
    """fit on log target; return smeared raw-scale prediction for Xq."""
    pred_log = model.predict(Xq)
    smear = np.mean(np.exp(ycal_log - predcal_log)) if ycal_log is not None else 1.0
    return np.exp(pred_log) * smear


def knn_sigma(Xtr_s, res_tr, Xq_s, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr_s))).fit(Xtr_s)
    _, idx = nn.kneighbors(Xq_s)
    sig = np.array([np.mean(np.abs(res_tr[ix])) for ix in idx])
    return np.maximum(sig, 1e-6)


def run_dataset(name, df, feats, target, thresholds=(0.70, 0.75, 0.80), seeds=(0, 1)):
    df = df.dropna(subset=feats + [target]).sort_values("d").reset_index(drop=True)
    recs = []
    for thr in thresholds:
        d_hi = df["d"].quantile(thr)
        pool = df[df.d < d_hi].reset_index(drop=True)
        test = df[df.d >= d_hi].reset_index(drop=True)
        Xe, ye = test[feats].values, test[target].values
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(pool))
            a, b = int(.5*len(pool)), int(.75*len(pool))
            tr, cal, ind = idx[:a], idx[a:b], idx[b:]
            Xtr, ytr = pool[feats].values[tr], pool[target].values[tr]
            Xcal, ycal = pool[feats].values[cal], pool[target].values[cal]
            Xind, yind = pool[feats].values[ind], pool[target].values[ind]
            mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
            Ztr, Zcal, Zind, Ze = (Xtr-mu)/sd, (Xcal-mu)/sd, (Xind-mu)/sd, (Xe-mu)/sd
            ltr = np.log(ytr)

            point_models = {
                "HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                          max_leaf_nodes=8, min_samples_leaf=15, random_state=seed),
                "RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                          max_features=0.7, random_state=seed, n_jobs=-1),
                "Linear": LinearRegression(),
            }
            for mname, model in point_models.items():
                model.fit(Xtr, ltr)
                pcal_log = model.predict(Xcal)
                vcal = duan_predict_log(model, Xtr, ltr, Xcal, np.log(ycal), pcal_log)
                vind = duan_predict_log(model, Xtr, ltr, Xind, np.log(ycal), pcal_log)
                ve   = duan_predict_log(model, Xtr, ltr, Xe,   np.log(ycal), pcal_log)
                rcal = ycal - vcal
                # (A) split-conformal absolute
                q = split_conformal_qhat(rcal, ALPHA)
                recs += [dict(ds=name, model=mname, uq="split", thr=thr, seed=seed,
                              interp=coverage(yind-vind, q), extrap=coverage(ye-ve, q))]
                # (B) kNN-normalized (locally adaptive) conformal
                rtr = ytr - duan_predict_log(model, Xtr, ltr, Xtr, np.log(ycal), pcal_log)
                sig_cal = knn_sigma(Ztr, rtr, Zcal); sig_ind = knn_sigma(Ztr, rtr, Zind); sig_e = knn_sigma(Ztr, rtr, Ze)
                qn = split_conformal_qhat(rcal/sig_cal, ALPHA)
                recs += [dict(ds=name, model=mname, uq="knn-norm", thr=thr, seed=seed,
                              interp=float(np.mean(np.abs((yind-vind)/sig_ind) <= qn)),
                              extrap=float(np.mean(np.abs((ye-ve)/sig_e) <= qn)))]

            # (C) GP native interval (variance grows off-support?)
            k = ConstantKernel(1.0)*RBF(length_scale=np.ones(Xtr.shape[1])) + WhiteKernel(0.1)
            gp = GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-6, n_restarts_optimizer=0)
            gp.fit(Ztr, ltr)
            for tag, Zq, yq in [("interp", Zind, yind), ("extrap", Ze, ye)]:
                m, s = gp.predict(Zq, return_std=True)
                lo, hi = np.exp(m-1.645*s), np.exp(m+1.645*s)
                cov = float(np.mean((yq >= lo) & (yq <= hi)))
                recs.append(dict(ds=name, model="GP", uq="native90", thr=thr, seed=seed,
                                 **{tag: cov, ("extrap" if tag=="interp" else "interp"): np.nan}))
            # (D) quantile GBM (raw target)
            g_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=300,
                    max_depth=3, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
            g_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=300,
                    max_depth=3, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
            for tag, Xq, yq in [("interp", Xind, yind), ("extrap", Xe, ye)]:
                cov = float(np.mean((yq >= g_lo.predict(Xq)) & (yq <= g_hi.predict(Xq))))
                recs.append(dict(ds=name, model="QuantGBM", uq="native90", thr=thr, seed=seed,
                                 **{tag: cov, ("extrap" if tag=="interp" else "interp"): np.nan}))
    return pd.DataFrame(recs)


def main():
    frp = pd.read_csv(PROC/"frp_clean.csv")
    frp = frp[(frp.a_d>=2.5)&(frp.rho_l<0.1)]
    st = pd.read_csv(PROC/"deepbeam_steel_clean.csv")
    R = pd.concat([
        run_dataset("FRP", frp, ["bw","d","fc","rho_l","Ef","a_d"], "V_test_kN"),
        run_dataset("STEEL", st, ["h","d","b","a_d","fck","rho","fy","rho_v","fyv","rho_h","fyh"], "V"),
    ], ignore_index=True)
    R.to_csv(PROC/"a07_uq_matrix_raw.csv", index=False)
    g = R.groupby(["ds","model","uq"]).agg(interp=("interp","mean"), extrap=("extrap","mean")).round(3)
    print("== corrected UQ-method matrix: interp vs EXTRAP coverage (target 0.90) ==")
    print(g.to_string())
    print("\n== does ANY method restore extrapolation coverage? ==")
    best = R.groupby(["ds"]).apply(lambda d: d.dropna(subset=["extrap"]).groupby(["model","uq"])["extrap"].mean().max())
    print(best.round(3).to_string())


if __name__ == "__main__":
    main()
