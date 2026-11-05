"""Idea register 9 (2026-11-05): coarser functionals scored from saved causal_loop outputs (no reruns).

Per program with a clear response (max_c |beta_cp| >= 0.1), from effects and truth beta (C x P):
  shared   mu_p = mean over types (unweighted; read-fraction weights are not saved): r(mu_est, mu_true) over
           programs, and its rmse
  spread   h_p = sd_c(beta_cp) / (|mu_p| + sd_c(beta_cp)) in [0, 1] (0 = shared by all types, 1 = restricted /
           opposite signs): r(h_est, h_true), and accuracy of "restricted" (h >= 0.5)
  n_resp   number of types with |beta| >= 0.1 (estimate: |effect| >= 0.1 and called): mean |error|
Compared with the per-type table: r over all (type, program) pairs.
Usage: python coarse_functionals.py [SPLITS=20-29] [ESTIMATORS=auto_v2,v4,v4s]
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")


def _h(x):
    mu, sd = x.mean(0), x.std(0)
    return sd / (np.abs(mu) + sd + 1e-12), mu


def stats(files):
    M, H, N, R = [], [], [], []
    for f in files:
        a = json.load(open(f))["arrays"]
        e, b = np.asarray(a["effects"], float), np.asarray(a["beta"], float)
        lo, hi = np.asarray(a["lo"], float), np.asarray(a["hi"], float)
        called = (lo > 0) | (hi < 0)
        R.append(np.corrcoef(e.ravel(), b.ravel())[0, 1])
        on = np.abs(b).max(0) >= 0.1
        he, me = _h(e)
        ht, mt = _h(b)
        for p in np.flatnonzero(on):
            M.append((me[p], mt[p])); H.append((he[p], ht[p]))
            N.append(abs(int(((np.abs(e[:, p]) >= 0.1) & called[:, p]).sum()) - int((np.abs(b[:, p]) >= 0.1).sum())))
    if len(M) < 3:
        return None
    M, H = np.array(M), np.array(H)
    return dict(n_prog=len(M), table_r=float(np.nanmean(R)), mu_r=float(np.corrcoef(M[:, 0], M[:, 1])[0, 1]),
                mu_rmse=float(np.sqrt(((M[:, 0] - M[:, 1]) ** 2).mean())),
                h_r=float(np.corrcoef(H[:, 0], H[:, 1])[0, 1]) if H[:, 1].std() > 0 else np.nan,
                restricted_acc=float(((H[:, 0] >= 0.5) == (H[:, 1] >= 0.5)).mean()),
                share_restricted=float((H[:, 1] >= 0.5).mean()), nresp_mae=float(np.mean(N)))


def main(splits, names):
    dss = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(D, "*__auto_v2.json"))})
    print("clear programs (max |beta| >= 0.1) pooled over splits; mu = unweighted mean over types")
    print("%-12s %-8s nprog table_r  mu_r  mu_rmse  h_r   restr_acc (share restr)  nresp_mae" % ("dataset", "est"))
    for ds in dss:
        for n in names:
            nn = n + "_pair" if ds in ("kang", "zhang") and n.startswith("v4") else n
            fs = [os.path.join(D, "%s__s%d__%s.json" % (ds, s, nn)) for s in splits]
            fs = [f for f in fs if os.path.exists(f)]
            s = stats(fs) if fs else None
            if s:
                print("%-12s %-8s %3d   %5.2f   %5.2f  %5.3f   %5.2f  %5.2f (%.2f)            %4.2f" % (
                    ds, n, s["n_prog"], s["table_r"], s["mu_r"], s["mu_rmse"], s["h_r"], s["restricted_acc"],
                    s["share_restricted"], s["nresp_mae"]))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
    import causal_loop as cl
    main(cl.db._splits(sys.argv[1] if len(sys.argv) > 1 else "20-29"),
         (sys.argv[2] if len(sys.argv) > 2 else "auto_v2,v4,v4s").split(","))
