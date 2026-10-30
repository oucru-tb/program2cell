"""Shared vs cell-type-specific response (deconvolution.shared_response) on the development splits.

Estimator: auto_m (the default PALM) with unmix_kw shared=True. Truth: beta (C, P) per type from the held-out
counts (causal_loop._setup); the true common effect is the shared-response model fitted to the noise-free
leaked true effects (T_n sum_kp A_ncqkp beta_kp; equal to the common value when the truth is shared), the true
deviation beta - common. Scores: common interval coverage, calls of the wrong sign (or on a true common effect below 0.02) and on clear common effects; deviation
coverage, 'differs' calls on null (|dev| < 0.05) and clear (|dev| >= 0.2) deviations, 'same' labels on
deviations larger than the margin; the pattern of the known program.

Usage: python shared_response.py run DATASETS SPLITS   |   python shared_response.py summary
Writes benchmarks/dev/shared_response/{ds}__s{split}.json
"""
import json, os, sys, time
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl

OUT = os.path.join(ROOT, "benchmarks", "dev", "shared_response")


def true_common(fractions, B, G, T, beta):
    from palm_bulk.core import unmix as um
    A = um.leakage_operator(fractions, B, G, diagonal=False)
    work = np.asarray(T, float)[:, None, None] * np.einsum("ncqkp,kp->ncq", A, beta)
    return um._fit_joint(work, A, T, None, None, shared_only=True)[1]


def score(sr, beta, m_true):
    c, clo, chi = (np.array(sr[k]) for k in ("common", "common_lo", "common_hi"))
    d, dlo, dhi = (np.array(sr[k]) for k in ("deviation", "deviation_lo", "deviation_hi"))
    lab = np.array(sr["type_label"])
    dt = beta - m_true[None]
    call_c = (clo > 0) | (chi < 0)
    diff = lab == "differs"
    nan = lambda m, x: float(x[m].mean()) if m.any() else np.nan
    # common calls: wrong sign (or a true common effect of essentially 0); detected clear common effects
    wrong = call_c & ((np.sign(c) != np.sign(m_true)) | (np.abs(m_true) < 0.02))
    return dict(common_cov=float(((m_true >= clo) & (m_true <= chi)).mean()),
                common_rmse=float(np.sqrt(((c - m_true) ** 2).mean())), common_width=float((chi - clo).mean()),
                common_wrong=nan(call_c, wrong), common_tpr=nan(np.abs(m_true) >= 0.1, call_c),
                dev_cov=float(((dt >= dlo) & (dt <= dhi)).mean()), dev_width=float((dhi - dlo).mean()),
                differs_fpr=nan(np.abs(dt) < 0.05, diff), differs_tpr=nan(np.abs(dt) >= 0.2, diff),
                same_wrong=nan(lab == "same", np.abs(dt) > sr["margin"]),
                n_same=int((lab == "same").sum()), n_unclear=int((lab == "unclear").sum()),
                n_differs=int(diff.sum()))


def run(datasets, splits):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            t0 = time.time()
            kw = dict(cl.ESTIMATORS["auto_m"])
            kw["unmix_kw"] = dict(kw["unmix_kw"], shared=True)
            r = pb.program_deconvolution(inp, cal, seed=split, **src, **kw)
            sr = r.info["shared_response"]
            m_true = true_common(r.fractions, inp.B, inp.Gmat, T, beta)
            s = score(sr, beta, m_true)
            s.update(dataset=dsn, split=split, seconds=time.time() - t0, shared_response=sr,
                     m_true=m_true.tolist(), beta=beta.tolist())
            k = cl.KNOWN.get(dsn)
            if k is not None:
                s["known_pattern"] = sr["pattern"][sr["programs"].index(k["program"])]
            with open(f, "w") as fh:
                json.dump(s, fh, default=float)
            print("[%s] %s s%d common cov %.2f wrong %s tpr %s | dev cov %.2f differs fpr %s tpr %s same_wrong %s "
                  "(same %d unclear %d differs %d) | %s: %s (%.0fs)" % (
                      time.strftime("%H:%M"), dsn, split, s["common_cov"], _f(s["common_wrong"]), _f(s["common_tpr"]),
                      s["dev_cov"], _f(s["differs_fpr"]), _f(s["differs_tpr"]), _f(s["same_wrong"]), s["n_same"],
                      s["n_unclear"], s["n_differs"], k and k["program"], s.get("known_pattern"), s["seconds"]),
                  flush=True)


def _f(x):
    return "  - " if x is None or not np.isfinite(x) else "%.2f" % x


def summary():
    import pandas as pd
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
    cols = ["common_cov", "common_width", "common_wrong", "common_tpr", "dev_cov", "dev_width", "differs_fpr",
            "differs_tpr", "same_wrong", "n_same", "n_unclear", "n_differs"]
    print(df.groupby("dataset")[cols].mean().round(3).to_string())
    print(df.groupby("dataset")["known_pattern"].value_counts().to_string())
    for r in rows:                                               # patterns per program
        print(r["dataset"], r["split"], dict(zip(r["shared_response"]["programs"], r["shared_response"]["pattern"])))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].split(","), cl.db._splits(sys.argv[3]))
    else:
        summary()
