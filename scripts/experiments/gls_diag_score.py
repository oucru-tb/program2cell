import json,glob,os,sys,numpy as np
D="benchmarks/dev/causal_loop"
def sc(b,e,lo,hi):
    c=(lo>0)|(hi<0); big=np.abs(b)>=0.1; null=np.abs(b)<0.05
    ws=[np.corrcoef(e[:,p],b[:,p])[0,1] for p in range(b.shape[1]) if b[:,p].std()>=0.05]
    return [((b>=lo)&(b<=hi)).mean(), np.corrcoef(e.ravel(),b.ravel())[0,1], np.sqrt(((e-b)**2).mean()),
            (c&big&(np.sign(e)==np.sign(b))).sum()/max(big.sum(),1), (c&null).sum()/max(null.sum(),1),
            np.mean(ws) if ws else np.nan, (hi-lo).mean()]
G=sys.argv[2] if len(sys.argv)>2 else "benchmarks/dev/gls_diag"
print("dataset     variant      n  cov    r     rmse  power  nullcalls  within-r  width")
for d in sys.argv[1].split(","):
    R={}
    for f in sorted(glob.glob(f"{G}/{d}__s*.json")):
        g=json.load(open(f)); s=int(f.rsplit("__s",1)[1][:-5]); b=np.array(g["beta"])
        for k,v in g.items():
            if isinstance(v,dict) and "eff" in v:
                R.setdefault(k,[]).append(sc(b,np.array(v["eff"]),np.array(v["lo"]),np.array(v["hi"])))
        v=json.load(open(f"{D}/{d}__s{s}__auto_v2.json"))["arrays"]
        R.setdefault("auto_v2",[]).append(sc(b,np.array(v["effects"]),np.array(v["lo"]),np.array(v["hi"])))
    for m,x in R.items():
        print("%-11s %-11s %2d "%(d,m,len(x))+" ".join("%6.2f"%v for v in np.nanmean(np.array(x,float),0)))
