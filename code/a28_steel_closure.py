"""
a28_steel_closure.py — CLOSE THE LOOP on the canonical steel benchmark.

The manuscript (r23) condemns ML intervals by their out-of-envelope coverage but
scores the mechanical fallback only by bias/COV — an apples-to-oranges gap a
reviewer will attack, and the Limitations section admits the steel gate+fallback
deployment is "the remaining next step". This script closes both gaps with the
SAME protocol and SAME metric used against the ML models:

 (1) Same-metric interval test: every predictor (ML HistGB/RF and the mechanical
     MCFT / CSCT / fib-MC2010 / ACI318-19) goes through the IDENTICAL split-
     conformal pipeline — bias-corrected on the in-envelope calibration split,
     interval calibrated on the same split, coverage evaluated on the held-out
     large members. Two interval spaces:
       raw: absolute residuals (the manuscript's existing convention), and
       ln:  multiplicative residuals (the code-style form for a ratio-type
            model error M; interval V_hat * exp(+-qhat)).
 (2) OOE model-error stats (bias / COV / %unsafe) per predictor per split —
     extends Table S2b across the full threshold/seed grid.
 (3) Protocol-level realised reliability: the envelope-gated protocol
     (ML inside the trained size range, mechanical fallback outside) per size
     bin, mixing failure probabilities member-wise; compares beta_ML,
     beta_mech, beta_protocol against beta_T = 3.8. Same load model, splits and
     seeds as a21_steel_reliability.py.
 (4) The efficiency price: in-envelope median interval width factor per
     predictor (what routing to the mechanical model costs where ML is valid).
 (5) FRP closure: the same same-metric interval test with ACI 440.1R-15 on the
     FRP database (completes the fig-remedy quantification).

Outputs -> ../data/processed/steel_closure_raw.csv, steel_closure_summary.csv,
           frp_closure.csv, protocol_beta.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from scipy.stats import norm
from uq_utils import split_conformal_qhat, coverage
from a12_mech_models import mcft, csct, fib_mc2010, aci318_19
from a08_mechanical_fallback import aci440_Vc_kN
from reliability_engine import (lognormal_from_mean_cov, normal_from_mean_cov,
                                gumbel_from_mean_cov)

PROC = Path("../data/processed")
F = ["bw", "d", "a_d", "rho", "fc", "ag", "fy"]
ALPHA = 0.10
BETA_T = 3.8
THRESHOLDS = (0.70, 0.75, 0.80)
SEEDS = (0, 1, 2)

MECH_STEEL = {
    "MCFT":       lambda r: mcft(r.bw, r.d, r.a_d, r.rho, r.fc) / 1e3,
    "CSCT":       lambda r: csct(r.bw, r.d, r.a_d, r.rho, r.fc) / 1e3,
    "fib-MC2010": lambda r: fib_mc2010(r.bw, r.d, r.a_d, r.rho, r.fc) / 1e3,
    "ACI318-19":  lambda r: aci318_19(r.bw, r.d, r.rho, r.fc) / 1e3,
}


def ml_models(seed):
    return {
        "ML-HistGB": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                     max_leaf_nodes=8, min_samples_leaf=15, random_state=seed),
        "ML-RF": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                 max_features=0.7, random_state=seed, n_jobs=-1),
    }


def m_stats(y, v):
    M = y / np.clip(v, 1e-9, None)
    return dict(bias=float(M.mean()), cov=float(M.std(ddof=1) / M.mean()),
                unsafe=float(np.mean(M < 1.0)))


def interval_records(name, ycal, vcal, targets):
    """Identical split-conformal treatment for any predictor.
    vcal / targets' v are ALREADY bias-corrected on the calibration split.
    targets: dict tag -> (y, v). Returns one record with raw + ln coverage."""
    q_raw = split_conformal_qhat(ycal - vcal, ALPHA)
    q_ln = split_conformal_qhat(np.log(ycal) - np.log(vcal), ALPHA)
    rec = dict(predictor=name, q_raw=q_raw, q_ln=q_ln,
               width_factor_ln=float(np.exp(2 * q_ln)))
    for tag, (y, v) in targets.items():
        rec[f"cov_raw_{tag}"] = coverage(y - v, q_raw)
        rec[f"cov_ln_{tag}"] = coverage(np.log(y) - np.log(v), q_ln)
    return rec


def run_closure(df, feats, ycol, mech, thresholds=THRESHOLDS, seeds=SEEDS,
                label="STEEL"):
    """Same-metric closure matrix over the threshold/seed grid."""
    df = df.dropna(subset=feats + [ycol]).sort_values("d").reset_index(drop=True)
    # mechanical predictions do not depend on the split: compute once
    for name, fn in mech.items():
        df[f"V::{name}"] = [fn(r) for r in df.itertuples()]
    recs = []
    for thr in thresholds:
        d_hi = df.d.quantile(thr)
        pool = df[df.d < d_hi].reset_index(drop=True)
        big = df[df.d >= d_hi].reset_index(drop=True)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(pool))
            a, b = int(.5 * len(pool)), int(.75 * len(pool))
            tr, cal, ind = idx[:a], idx[a:b], idx[b:]
            Xtr, ytr = pool[feats].values[tr], pool[ycol].values[tr]
            Xcal, ycal = pool[feats].values[cal], pool[ycol].values[cal]
            Xind, yind = pool[feats].values[ind], pool[ycol].values[ind]
            Xe, ye = big[feats].values, big[ycol].values

            preds = {}
            for mname, model in ml_models(seed).items():
                model.fit(Xtr, np.log(ytr))
                shift = np.mean(np.log(ycal) - model.predict(Xcal))
                preds[mname] = {t: np.exp(model.predict(X) + shift)
                                for t, X in [("cal", Xcal), ("ind", Xind), ("big", Xe)]}
            for name in mech:
                vc = pool[f"V::{name}"].values
                shift = np.mean(np.log(ycal) - np.log(vc[cal]))
                preds[name] = {"cal": vc[cal] * np.exp(shift),
                               "ind": vc[ind] * np.exp(shift),
                               "big": big[f"V::{name}"].values * np.exp(shift)}

            for name, p in preds.items():
                rec = interval_records(name, ycal, p["cal"],
                                       {"ind": (yind, p["ind"]), "big": (ye, p["big"])})
                rec.update(ds=label, thr=thr, seed=seed, n_big=len(big),
                           d_hi=float(d_hi),
                           **{f"ooe_{k}": v for k, v in m_stats(ye, p["big"]).items()})
                recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Protocol-level realised reliability (same split/load model as a21)
# ---------------------------------------------------------------------------
def mc_pf(M_sample, util, n=400_000, seed=0):
    """Failure probability with empirical-lognormal model error and the a21
    dead+live load model, at design utilisation util."""
    mu, cov = float(np.mean(M_sample)), float(np.std(M_sample, ddof=1) / np.mean(M_sample))
    rng = np.random.default_rng(seed)
    S = util * mu
    D = normal_from_mean_cov(0.6 * S, 0.10).rvs(n, random_state=rng)
    L = gumbel_from_mean_cov(0.4 * S, 0.25).rvs(n, random_state=rng)
    R = lognormal_from_mean_cov(mu, cov).rvs(n, random_state=rng)
    return max(float(np.mean(R - (D + L) <= 0)), 1.0 / n)


def util_for_target(cov, n=200_000):
    lo, hi = 0.1, 0.9
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(1)
        D = normal_from_mean_cov(0.6 * mid, 0.10).rvs(n, random_state=rng)
        L = gumbel_from_mean_cov(0.4 * mid, 0.25).rvs(n, random_state=rng)
        R = lognormal_from_mean_cov(1.0, cov).rvs(n, random_state=rng)
        beta = -norm.ppf(max(float(np.mean(R - (D + L) <= 0)), 1.0 / n))
        if beta > BETA_T:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def protocol_reliability(df):
    df = df[df.a_d >= 2.5].dropna(subset=F + ["Vu_kN"]).reset_index(drop=True)
    d_hi = df.d.quantile(0.75)
    pool_idx = np.where(df.d.values < d_hi)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(pool_idx)
    cut = int(0.70 * len(pool_idx))
    tr_idx, cal_idx = pool_idx[:cut], pool_idx[cut:]
    ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
         max_leaf_nodes=8, min_samples_leaf=15, random_state=0)
    ml.fit(df.loc[tr_idx, F].values, np.log(df.loc[tr_idx, "Vu_kN"].values))
    sm = np.mean(np.exp(np.log(df.loc[cal_idx, "Vu_kN"].values)
                        - ml.predict(df.loc[cal_idx, F].values)))
    df["Vml"] = np.exp(ml.predict(df[F].values)) * sm
    df["Vmcft"] = [mcft(r.bw, r.d, r.a_d, r.rho, r.fc) / 1e3 for r in df.itertuples()]
    df["V31819"] = [aci318_19(r.bw, r.d, r.rho, r.fc) / 1e3 for r in df.itertuples()]

    d_train_max = df.loc[tr_idx, "d"].max()  # the range gate boundary
    covs, utils = {}, {}
    for tag, vcol in [("ml", "Vml"), ("mcft", "Vmcft"), ("a19", "V31819")]:
        Mc = df.loc[cal_idx, "Vu_kN"] / df.loc[cal_idx, vcol]
        covs[tag] = Mc.std(ddof=1) / Mc.mean()
        utils[tag] = util_for_target(covs[tag])

    bins = [(0, 150), (150, 250), (250, 400), (400, 650), (650, 2000)]
    rows = []
    for lo, hi in bins:
        m = (df.d >= lo) & (df.d < hi)
        if m.sum() < 8:
            continue
        sub = df[m]
        beta = {}
        for tag, vcol in [("ml", "Vml"), ("mcft", "Vmcft"), ("a19", "V31819")]:
            beta[tag] = -norm.ppf(mc_pf(sub.Vu_kN / sub[vcol], utils[tag]))
        # gated protocol: members inside the trained size range keep ML,
        # members beyond it are routed to the mechanical fallback
        inside = sub.d <= d_train_max
        pf_parts, n_parts = [], []
        for mask, vcol, tag in [(inside, "Vml", "ml"), (~inside, "Vmcft", "mcft")]:
            if mask.sum() >= 8:
                pf_parts.append(mc_pf(sub.Vu_kN[mask] / sub.loc[mask, vcol], utils[tag]))
                n_parts.append(int(mask.sum()))
            elif mask.sum() > 0:  # tiny slice: score it with the other branch's pf
                pf_parts.append(None)
                n_parts.append(int(mask.sum()))
        pf_known = [p for p in pf_parts if p is not None]
        pf_fill = max(pf_known) if pf_known else 1e-6
        pf_mix = sum((p if p is not None else pf_fill) * n
                     for p, n in zip(pf_parts, n_parts)) / sum(n_parts)
        beta_proto = -norm.ppf(pf_mix)
        # protocol variant with the ACI318-19 fallback
        pf_parts2 = []
        for mask, vcol, tag in [(inside, "Vml", "ml"), (~inside, "V31819", "a19")]:
            if mask.sum() >= 8:
                pf_parts2.append(mc_pf(sub.Vu_kN[mask] / sub.loc[mask, vcol], utils[tag]))
            elif mask.sum() > 0:
                pf_parts2.append(None)
        pf_known2 = [p for p in pf_parts2 if p is not None]
        pf_fill2 = max(pf_known2) if pf_known2 else 1e-6
        pf_mix2 = sum((p if p is not None else pf_fill2) * n
                      for p, n in zip(pf_parts2, n_parts)) / sum(n_parts)
        rows.append(dict(d_lo=lo, d_hi=hi, d_mid=(lo + hi) / 2 if hi < 2000 else 900,
                         n=int(m.sum()), n_inside=int(inside.sum()),
                         n_routed=int((~inside).sum()),
                         beta_ml=beta["ml"], beta_mcft=beta["mcft"],
                         beta_a19=beta["a19"], beta_protocol=beta_proto,
                         beta_protocol_a19=-norm.ppf(pf_mix2),
                         cal_cov_ml=float(covs["ml"]), cal_cov_mcft=float(covs["mcft"]),
                         cal_cov_a19=float(covs["a19"]),
                         util_ml=float(utils["ml"]), util_mcft=float(utils["mcft"]),
                         util_a19=float(utils["a19"]),
                         d_train_max=float(d_train_max)))
        print(f"  bin {lo:4d}-{hi:4d} n={m.sum():4d} routed={int((~inside).sum()):4d} | "
              f"beta ML={beta['ml']:.2f}  MCFT={beta['mcft']:.2f}  "
              f"PROTOCOL={beta_proto:.2f}  PROTOCOL(318-19)={-norm.ppf(pf_mix2):.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "protocol_beta.csv", index=False)
    return out


def main():
    steel = pd.read_csv(PROC / "steel_zhang_clean.csv")
    sl = steel[steel.a_d >= 2.5].dropna(subset=F + ["Vu_kN"]).reset_index(drop=True)
    print(f"== canonical steel closure (slender n={len(sl)}) ==")
    R = run_closure(sl, F, "Vu_kN", MECH_STEEL, label="STEEL")
    R.to_csv(PROC / "steel_closure_raw.csv", index=False)
    summ = (R.groupby("predictor")
              .agg(cov_ln_ind=("cov_ln_ind", "mean"), cov_ln_big=("cov_ln_big", "mean"),
                   cov_raw_ind=("cov_raw_ind", "mean"), cov_raw_big=("cov_raw_big", "mean"),
                   ooe_bias=("ooe_bias", "mean"), ooe_cov=("ooe_cov", "mean"),
                   ooe_unsafe=("ooe_unsafe", "mean"),
                   width_factor_ln=("width_factor_ln", "mean"))
              .round(3).sort_values("cov_ln_big", ascending=False))
    summ.to_csv(PROC / "steel_closure_summary.csv")
    print("\nSame-metric interval test (mean over threshold/seed grid; target 0.90):")
    print(summ.to_string())

    frp = pd.read_csv(PROC / "frp_clean.csv")
    frp = frp[(frp.a_d >= 2.5) & (frp.rho_l < 0.1)].reset_index(drop=True)
    frp["V::ACI440"] = aci440_Vc_kN(frp)
    Rf = run_closure(frp, ["bw", "d", "fc", "rho_l", "Ef", "a_d"], "V_test_kN",
                     {"ACI440": lambda r: aci440_Vc_kN(pd.DataFrame([r._asdict()])).iloc[0]},
                     label="FRP")
    Rf.to_csv(PROC / "frp_closure.csv", index=False)
    sf = (Rf.groupby("predictor")
            .agg(cov_ln_ind=("cov_ln_ind", "mean"), cov_ln_big=("cov_ln_big", "mean"),
                 ooe_bias=("ooe_bias", "mean"), ooe_cov=("ooe_cov", "mean"),
                 ooe_unsafe=("ooe_unsafe", "mean")).round(3))
    print("\nFRP closure (ACI 440.1R-15 vs ML, same metric):")
    print(sf.to_string())

    print("\n== protocol-level realised reliability (gate + fallback), steel ==")
    protocol_reliability(steel)
    print("saved -> steel_closure_raw/summary.csv, frp_closure.csv, protocol_beta.csv")


if __name__ == "__main__":
    main()
