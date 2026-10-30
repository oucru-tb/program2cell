"""Development loop for causal cell-type effects on real perturbation data with a known answer.

Scoreboard of effect estimators (every engine kept; new ones are added to ESTIMATORS) against targets:
  coverage of 90% effect intervals >= 0.85, correlation with the truth >= 0.9, calls on truly null pairs
  (|true effect| < 0.05, all programs; fpr_null) <= 0.1, and on the known program: responder AUROC >= 0.95,
  responders called with the right sign (TPR) >= 0.9 (fpr: calls among the label-based non-responders,
  reported, not a target: weak real responders count there).
Datasets and known answers:
  mcf_ida   MIX-Seq idasanutlin: p53 up in the TP53-wild-type lines (LNCaP, RCC10RGB, SNU1079) only
  mcf_tram  MIX-Seq trametinib: MAPK down in every line
  kang      IFN-beta: JAK-STAT up in every immune type
  zhang     paclitaxel +- atezolizumab, blood pre vs post: no known answer (coverage and correlation only)
Truth: the treatment contrast of the thinned held-out half's activity (dev_bench). Fractions: the PALM
posterior of the PROGENy-7 fit when it exists, else PALM-lite (recorded).
One json per (dataset, split, estimator) in benchmarks/dev/causal_loop/. Restart-safe.
Usage: python causal_loop.py run DATASETS SPLITS [ESTIMATORS]   |   python causal_loop.py summary [heldout]
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db

OUT = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
KNOWN = {"mcf_ida": dict(program="p53", types=["LNCAPCLONEFGC", "RCC10RGB", "SNU1079"], sign=1),
         "mcf_tram": dict(program="MAPK", types=None, sign=-1),
         "kang": dict(program="JAK-STAT", types=None, sign=1)}
TARGETS = dict(cov=(">=", 0.85), r=(">=", 0.9), auroc=(">=", 0.95), tpr_t=(">=", 0.9), fpr_null=("<=", 0.1))
# r is a target only where the truth has effects (max |beta| >= 0.1): with a null truth (zhang) it is noise
R_MIN_EFFECT = 0.1
# name -> program_deconvolution keywords (every estimator stays in the scoreboard)
ESTIMATORS = {"diff": dict(estimator="diff"),
              "ols_pair": dict(estimator="ols", paired=True),
              "unmix": dict(estimator="unmix", unmix_kw=dict(pev=False, misfit=False)),
              "auto": dict(estimator="auto", unmix_kw=dict(pev=False, misfit=False)),                       # since 2026-10-28 18:30 with sim_widen
              "auto_nowiden": dict(estimator="auto", sim_widen=False, unmix_kw=dict(pev=False, misfit=False)),
              "auto_pair": dict(estimator="auto", paired=True, unmix_kw=dict(pev=False, misfit=False)),
              "auto_sim": dict(estimator="auto", sim_widen="replace", unmix_kw=dict(pev=False, misfit=False)),        # simulation-calibrated intervals
              "auto_sim_pair": dict(estimator="auto", sim_widen="replace", paired=True, unmix_kw=dict(pev=False, misfit=False)),
              # 2026-10-28 evening: prediction-error variance + off-program response (misfit) in the intervals
              "unmix_m": dict(estimator="unmix", unmix_kw=dict(pev=True, misfit=True)),
              "auto_m": dict(estimator="auto", unmix_kw=dict(pev=True, misfit=True)),
              "auto_m_pair": dict(estimator="auto", paired=True, unmix_kw=dict(pev=True, misfit=True)),
              "auto_mc": dict(estimator="auto", unmix_kw=dict(pev=True, misfit=True, per_pair=True)),
              # 2026-10-30 (post-freeze development): design check planted like the study's own effects
              "auto_obs": dict(estimator="auto", unmix_kw=dict(pev=True, misfit=True, plant="observed")),
              # 2026-10-30: calibrated programs widened by the absolute leakage error per type (sim_widen="abs")
              "auto_abs": dict(estimator="auto", sim_widen="abs", unmix_kw=dict(pev=True, misfit=True)),
              "direct": dict(estimator="direct", direct_kw=dict(n_boot=40)),
              "direct_nb": dict(estimator="direct", direct_kw=dict(n_boot=40, weights="nb")),
              "direct_pair": dict(estimator="direct", paired=True, direct_kw=dict(n_boot=40))}


def _setup(dsn, split):
    import palm_bulk as pb
    ds = db.DATASETS[dsn]
    table = db._table(ds)
    prog = db._programs(pb, "progeny7")
    b = db._benchmark(table, ds, split)
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    T = np.asarray(inp.T)
    truth = b.truth(inp)
    beta = pb.core.contrast(T)(truth["delta"])
    z7 = os.path.join(ROOT, "benchmarks", "dev", dsn, "s%d__joint_nb.npz" % split)
    if os.path.exists(z7):
        z = np.load(z7)["pi"]
        src = dict(fraction_draws=z[np.linspace(0, len(z) - 1, 50).astype(int)])
        fsrc = "palm"
    else:
        src, fsrc = dict(engine="lite"), "lite"
    cal = pb.calibration_from_table(table, set(b.meta["donor_id"].astype(str)), inp.cell_types, inp.genes, inp.Gmat,
                                    ref_filter=ds.get("ref_filter"),
                                    all_specimens=ds.get("all_test_specimens", False), seed=split)
    donors = b.meta["donor_id"].astype(str).values
    return pb, inp, T, beta, truth, src, fsrc, cal, donors


def score(effects, lo, hi, calls, beta, cell_types, programs, known):
    from palm_bulk.compare import known_answer as ka
    s = dict(cov=float(((beta >= lo) & (beta <= hi)).mean()),
             r=float(np.corrcoef(effects.ravel(), beta.ravel())[0, 1]),
             rmse=float(np.sqrt(((effects - beta) ** 2).mean())), width=float((hi - lo).mean()),
             r_t=float(np.corrcoef(effects.ravel(), beta.ravel())[0, 1]) if np.abs(beta).max() >= R_MIN_EFFECT
             else np.nan,
             calls=int(calls.sum()),
             # calls on pairs whose true effect is essentially zero (all programs): the truth-based false
             # positive rate (the label-based fpr below counts weak real responders, e.g. BICR31 +0.12 p53)
             fpr_null=float(calls[np.abs(beta) < 0.05].mean()) if (np.abs(beta) < 0.05).any() else np.nan)
    if known is not None and known["program"] in programs:
        j = programs.index(known["program"])
        resp = np.array([known["types"] is None or c in known["types"] for c in cell_types])
        e, c = effects[:, j], calls[:, j]
        s["auroc"] = ka.responder_auroc(e, resp, known["sign"]) if (~resp).any() else np.nan
        s["tpr"] = float((c & (np.sign(e) == known["sign"]))[resp].mean())
        s["fpr"] = float(c[~resp].mean()) if (~resp).any() else np.nan
        # truth-based: known-program pairs with a clear true effect (|beta| >= 0.1) called with the right sign
        # (label-based tpr counts e.g. LNCaP under trametinib, true MAPK -0.04: a non-responder in fact)
        clear = np.abs(beta[:, j]) >= R_MIN_EFFECT
        s["tpr_t"] = float((c & (np.sign(e) == np.sign(beta[:, j])))[clear].mean()) if clear.any() else np.nan
        s["cov_known"] = float(((beta[:, j] >= lo[:, j]) & (beta[:, j] <= hi[:, j])).mean())
    return s


def run(datasets, splits, names):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            todo = [n for n in names if not os.path.exists(os.path.join(OUT, "%s__s%d__%s.json" % (dsn, split, n)))]
            if not todo:
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = _setup(dsn, split)
            for name in todo:
                kw = dict(ESTIMATORS[name])
                if kw.pop("paired", False):
                    kw["groups"] = donors
                t0 = time.time()
                r = pb.program_deconvolution(inp, cal, seed=split, **src, **kw)
                lo, hi = r.effects_interval
                s = score(r.effects, lo, hi, r.calls, beta, list(inp.cell_types), list(inp.programs), KNOWN.get(dsn))
                s.update(dataset=dsn, split=split, estimator=name, fractions=fsrc, seconds=time.time() - t0,
                         unmixed=",".join(r.info.get("unmix", {}).get("programs", [])),
                         arrays=dict(effects=r.effects.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                                     calls=r.calls.tolist(), beta=beta.tolist(), cell_types=list(inp.cell_types),
                                     programs=list(inp.programs)))
                with open(os.path.join(OUT, "%s__s%d__%s.json" % (dsn, split, name)), "w") as fh:
                    json.dump(s, fh, default=float)
                print("[%s] %s s%d %-9s cov %.2f r %.3f %s(%.0fs)" % (
                    time.strftime("%H:%M"), dsn, split, name, s["cov"], s["r"],
                    "auroc %.2f tpr %.2f fpr %.2f " % (s.get("auroc", np.nan), s.get("tpr", np.nan),
                                                       s.get("fpr", np.nan)) if "tpr" in s else "", s["seconds"]),
                      flush=True)


def summary(heldout=False):
    """Development splits (< db.HELDOUT_FROM) or, with heldout=True, the held-out splits only."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        s = json.load(open(f))
        a = s.pop("arrays", None)
        if a is not None:                                          # rescore with the current metrics
            s.update(score(*(np.asarray(a[k], float) for k in ("effects", "lo", "hi")), np.asarray(a["calls"], bool),
                           np.asarray(a["beta"], float), a["cell_types"], a["programs"], KNOWN.get(s["dataset"])))
        rows.append(s)
    df = pd.DataFrame(rows)
    df = df[(df.split >= db.HELDOUT_FROM) == heldout]
    print("== %s splits: %s" % ("held-out" if heldout else "development", sorted(df.split.unique())))
    cols = [c for c in ("cov", "r", "r_t", "rmse", "width", "calls", "fpr_null", "auroc", "tpr", "tpr_t", "fpr", "cov_known")
            if c in df]
    t = df.groupby(["dataset", "estimator"], sort=False)[cols].mean()
    t.insert(0, "n", df.groupby(["dataset", "estimator"], sort=False).split.nunique())
    ok = pd.Series(True, index=t.index)
    for m, (op, v) in TARGETS.items():
        m = "r_t" if m == "r" else m
        if m in t:
            good = (t[m] >= v) if op == ">=" else (t[m] <= v)
            ok &= good | t[m].isna()
    t["PASS"] = np.where(ok, "yes", "-")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t.round(3).to_string())
    print("\ntargets:", ", ".join("%s %s %s" % (m, op, v) for m, (op, v) in TARGETS.items()))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(heldout=len(sys.argv) > 2 and sys.argv[2] == "heldout")
    else:
        names = sys.argv[4].split(",") if len(sys.argv) > 4 else list(ESTIMATORS)
        run(sys.argv[2].split(","), db._splits(sys.argv[3]), names)
