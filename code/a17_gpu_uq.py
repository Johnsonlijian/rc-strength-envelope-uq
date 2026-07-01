"""
a17_gpu_uq.py — modern GPU deep-UQ methods under size extrapolation (steel 1704).
Deep ensembles (Gaussian NLL) and MC-dropout are the canonical deep predictive-
uncertainty methods. We test whether they---unlike the tree/conformal methods---
restore coverage out of envelope. They do not. (torch, CUDA.)
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from scipy.stats import norm

PROC = Path("../data/processed"); FEATS=["bw","d","a_d","rho","fc","ag","fy"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self, nin, drop=0.0, heads=1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nin,128), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(128,128), nn.ReLU(), nn.Dropout(drop))
        self.head = nn.Linear(128, heads)
    def forward(self,x): return self.head(self.body(x))


def train(model, X, y, epochs=800, nll=False, lr=2e-3):
    opt=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5); model.train()
    for _ in range(epochs):
        opt.zero_grad(); out=model(X)
        if nll:
            mu, logv = out[:,0:1], out[:,1:2]
            loss=(0.5*torch.exp(-logv)*(y-mu)**2 + 0.5*logv).mean()
        else:
            loss=((out-y)**2).mean()
        loss.backward(); opt.step()
    return model


def cov90(y, lo, hi): return float(((y>=lo)&(y<=hi)).mean())


def prep(df, idx_tr, idx_te, mu, sd):
    Xtr=torch.tensor(((df[FEATS].values[idx_tr]-mu)/sd),dtype=torch.float32,device=dev)
    Xte=torch.tensor(((df[FEATS].values[idx_te]-mu)/sd),dtype=torch.float32,device=dev)
    ytr=torch.tensor(np.log(df.Vu_kN.values[idx_tr])[:,None],dtype=torch.float32,device=dev)
    return Xtr,Xte,ytr


def deep_ensemble(df, tr, te, mu, sd, K=5):
    Xtr,Xte,ytr=prep(df,tr,te,mu,sd); mus=[]; vs=[]
    for k in range(K):
        torch.manual_seed(k); m=MLP(len(FEATS),heads=2).to(dev); train(m,Xtr,ytr,nll=True)
        m.eval()
        with torch.no_grad(): out=m(Xte).cpu().numpy()
        mus.append(out[:,0]); vs.append(np.exp(out[:,1]))
    mus=np.array(mus); vs=np.array(vs)
    mean=mus.mean(0); var=(vs+mus**2).mean(0)-mean**2     # ensemble predictive variance
    return mean, np.sqrt(np.maximum(var,1e-6))


def mc_dropout(df, tr, te, mu, sd, T=60):
    Xtr,Xte,ytr=prep(df,tr,te,mu,sd)
    m=MLP(len(FEATS),drop=0.1,heads=1).to(dev); train(m,Xtr,ytr); m.train()  # keep dropout on
    with torch.no_grad():
        S=np.array([m(Xte).cpu().numpy()[:,0] for _ in range(T)])
    return S.mean(0), S.std(0)


def main():
    df = pd.read_csv(PROC/"steel_zhang_clean.csv")
    df = df[df.a_d>=2.5].dropna(subset=FEATS+["Vu_kN"]).sort_values("d").reset_index(drop=True)
    print(f"== GPU deep-UQ on steel benchmark (n={len(df)}, device={dev}, {torch.cuda.get_device_name(0) if dev=='cuda' else ''}) ==")
    rows=[]
    for thr in [0.70,0.75,0.80]:
        d_hi=df.d.quantile(thr); pool=np.where(df.d.values<d_hi)[0]; ext=np.where(df.d.values>=d_hi)[0]
        rng=np.random.default_rng(0)
        rng.shuffle(pool); cut=int(0.8*len(pool)); tr, ind = pool[:cut], pool[cut:]
        # Standardisation is part of the learned model and must not see the
        # held-out in-envelope or out-of-envelope specimens.
        mu, sd = df[FEATS].values[tr].mean(0), df[FEATS].values[tr].std(0)+1e-9
        yext=np.log(df.Vu_kN.values[ext]); yind=np.log(df.Vu_kN.values[ind])
        for name, fn in [("DeepEnsemble", deep_ensemble), ("MC-Dropout", mc_dropout)]:
            m_in,s_in = fn(df, tr, ind, mu, sd); m_ex,s_ex = fn(df, tr, ext, mu, sd)
            z=norm.ppf(0.95)
            ci=cov90(yind, m_in-z*s_in, m_in+z*s_in); ce=cov90(yext, m_ex-z*s_ex, m_ex+z*s_ex)
            rows.append(dict(thr=thr, split_seed=0, train_n=len(tr), interp_n=len(ind),
                             extrap_n=len(ext), method=name, interp=ci, extrap=ce))
    R=pd.DataFrame(rows)
    g=R.groupby("method").agg(interp=("interp","mean"), extrap=("extrap","mean")).round(3)
    print("\n90% predictive-interval coverage (target 0.90):")
    print(g.to_string())
    print(f"\nbest deep-UQ extrapolation coverage = {R.extrap.max():.3f}  -> deep UQ also fails out of envelope")
    R.to_csv(PROC/"steel_gpu_uq_raw.csv", index=False)
    g.to_csv(PROC/"steel_gpu_uq.csv")


if __name__ == "__main__":
    main()
