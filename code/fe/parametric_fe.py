"""Parametric crack-band FE study (multi-core): size effect across (d,a/d,rho,fc).
Shows the simulated size effect is robust across the parameter space, not one config."""
import numpy as np, json, time, itertools
from multiprocessing import Pool
from damage_fe import shear_capacity_fe

DS   = [100,150,225,300,450,600,900,1350]
ADS  = [2.5,3.0,3.5,4.0]
RHOS = [0.010,0.015,0.020,0.025]
FCS  = [25.0,35.0,50.0]
GRID = list(itertools.product(DS,ADS,RHOS,FCS))     # 8*4*4*3 = 384 runs

def run(args):
    d,ad,rho,fc=args
    try:
        v,V,_=shear_capacity_fe(d=d,a_d=ad,rho=rho,fc=fc,n_d=12,steps=40)
        return dict(d=d,a_d=ad,rho=rho,fc=fc,v_nom=float(v),Vpeak_kN=float(V/1e3))
    except Exception as e:
        return dict(d=d,a_d=ad,rho=rho,fc=fc,v_nom=None,err=str(e)[:60])

if __name__=="__main__":
    t0=time.time()
    with Pool(48) as p:
        rows=p.map(run, GRID)
    ok=[r for r in rows if r.get("v_nom")]
    print(f"{len(ok)}/{len(GRID)} FE runs ok in {time.time()-t0:.0f}s")
    # per-(a/d,rho,fc) size-effect exponent across d
    import pandas as pd
    df=pd.DataFrame(ok); slopes=[]
    for (ad,rho,fc),g in df.groupby(["a_d","rho","fc"]):
        g=g[g.d>=225]
        if len(g)>=4:
            sl=np.polyfit(np.log(g.d),np.log(g.v_nom),1)[0]; slopes.append(sl)
    slopes=np.array(slopes)
    print(f"size-effect exponent across {len(slopes)} parameter combos: "
          f"mean={slopes.mean():.2f} sd={slopes.std():.2f} (range {slopes.min():.2f}..{slopes.max():.2f})")
    json.dump({"rows":ok,"slope_mean":float(slopes.mean()),"slope_sd":float(slopes.std())},
              open("../../data/processed/parametric_fe.json","w"),indent=2)
    print("saved -> data/processed/parametric_fe.json")
