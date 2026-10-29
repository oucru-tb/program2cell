"""Method-development benchmark: several donor splits, paired comparisons, a reserved held-out set.

Each (dataset, split, config) is fitted once and written to its own JSON row file, so the run is
restart-safe (existing rows are skipped) and can run unattended. Splits 0-9 are for development;
splits 10-19 are the held-out set and are refused unless --heldout is given (use them once, at the
method freeze). Truth always comes from the held-out half of the counts (truth_split=True).

  python scripts/dev_bench.py run --dataset pbmc --splits 0-9 --configs night
  python scripts/dev_bench.py summary --dataset pbmc --ref "naive|model"

Two truths: delta from palm_bulk.estimate_delta on each cell type's held-out counts (the model's
estimand, all metrics), and a model-free check, decoupler ULM scores on the same counts (rank metrics
only, prefixed ulm_), so a method cannot win just by sharing the first truth's estimator.

Rows per palm config:
  model              posterior delta, the model's beta
  design beta        posterior delta, beta = difference of mean delta between groups (per draw)
  stage-1 mode       each sample's own fit (no pooling; diagnostics, no intervals)
  posterior realloc  bulk counts split by posterior fractions x reference profiles, then the per-type
                     estimator; intervals over fraction draws
  realloc + alloc    the same with each count split at random (multinomial) per draw, so the
                     intervals also carry allocation noise
  EM realloc         one EM step: the split uses each type's program-shifted profile
                     softmax(log B_c + delta_ic G) with the posterior median delta, not B_c
  iter realloc       EM with fractions fixed at the posterior median, started from the plain
                     reallocation (3 steps; converges towards the unpooled per-sample estimate)
Baselines: naive (one bulk shift copied to every type), reallocation (NNLS fractions), oracle realloc
(true fractions: the ceiling for reallocation given perfect fractions).
Published competitors (--competitors cellgeometry,bayesprism) run on the same split, genes and truth.
Their single-cell reference is the reference donors' pseudo-bulk rows (the information PALM's B is
built from), or real cells of the reference donors with --cells (names get a " cells" suffix). Rows:
model (the method's own activity: BayesPrism's per-type expression; cellGeometry has none, so its
wrapper reallocates) and, for BayesPrism, realloc (its fractions with the plain reallocation).
`rescore` recomputes every row's metrics from the saved arrays (after a change to the metrics), keeping
the fit diagnostics; rows without arrays (fits from before 2026-10-27 morning) are left as they are.
Each fit also saves its arrays to s<split>__<config>.npz (and the truth to s<split>__truth.npz), so other
combinations of the estimates can be scored later without refitting.
"""
import argparse, glob, json, os, sys, time, types
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks", "dev")
PROGS = "JAK-STAT,NFkB,TNFa,TGFb,Hypoxia,p53,MAPK".split(",")
# program prior sets (--prior); every set except the default writes to benchmarks/dev/<dataset>__<prior>, since
# the truth (delta on that program matrix) differs. See docs/benchmarking.md, "Program prior sets".
PRIORS = {
    "progeny7": ("progeny_human_top500.csv", PROGS),          # the default and all results before 2026-10-27
    "progeny14": ("progeny_human_top200.csv", None),          # all 14 PROGENy pathways, top 200 genes (as Tripso)
    "collectri": ("collectri_immune.csv", None),              # 26 immune TF regulons, 25-250 targets, signed
    "hallmark": ("hallmark_immune.csv", None),                # 15 MSigDB Hallmark immune/state sets, unsigned
}


def _odir(args):
    return os.path.join(args.out, args.dataset + ("" if args.prior == "progeny7" else "__" + args.prior))


def _programs(pb, prior):
    f, keep = PRIORS[prior]
    prog = pb.load_programs(os.path.join(ROOT, "data", f))
    return prog.loc[keep] if keep else prog
TYPES = "B cell,Monocyte,NK cell,T cell".split(",")
MCF_LINES = sorted(["LNCAPCLONEFGC", "RCC10RGB", "SNU1079", "COLO680N", "BICR31", "TEN"])
DATASETS = {
    "pbmc": dict(path=os.path.join(ROOT, "pbmc_broad_clean.h5ad"), contrast=("disease", "COVID-19", "HC"), n_test=None),
    "o1k": dict(path="C:/Users/duy.pham/Downloads/onek1k_full_broad.h5ad", contrast=("sex", "male", "female"), n_test=150),
    # Kang 2018 IFN-beta stimulation (scripts/experiments/build_kang.py): 8 donors, paired stim/ctrl
    # pseudo-replicates; 4 test donors (all their specimens), reference and calibration from the other
    # donors' control cells only, as with a resting reference atlas
    "kang": dict(path=os.path.join(ROOT, "data", "kang", "kang_pseudobulk.h5ad"), contrast=("label", "stim", "ctrl"),
                 n_test=4, ref_filter=("label", "ctrl"), all_test_specimens=True),
    # Zhang 2021 TNBC blood, pre vs post paclitaxel +/- atezolizumab (scripts/experiments/build_zhang.py): 21
    # patients, 2 pseudo-replicates per patient x time; 10 test patients (all their specimens, paired), reference
    # and calibration from the other patients' pre-treatment blood
    "zhang": dict(path=os.path.join(ROOT, "data", "pert", "zhang_blood_pseudobulk.h5ad"), contrast=("label", "post", "pre"),
                  n_test=10, ref_filter=("label", "pre"), all_test_specimens=True),
    # McFarland 2020 MIX-Seq, semi-synthetic pooled-line mixtures with real drug responses
    # (scripts/experiments/build_mcfarland.py); cell types = 6 cell lines; 20 of 40 paired mixtures per split;
    # reference and calibration from a disjoint pool of control cells. Known answers: idasanutlin -> p53 in the
    # TP53-wild-type lines only; trametinib -> MAPK down in all lines
    "mcf_ida": dict(path=os.path.join(ROOT, "data", "pert", "mcfarland_mix.h5ad"), contrast=("label", "ida", "ctrl"),
                    n_test=20, ref_filter=("label", "refctrl"), all_test_specimens=True, types=MCF_LINES),
    "mcf_tram": dict(path=os.path.join(ROOT, "data", "pert", "mcfarland_mix.h5ad"), contrast=("label", "tram", "ctrl"),
                     n_test=20, ref_filter=("label", "refctrl"), all_test_specimens=True, types=MCF_LINES),
}


def _types(ds):
    """Cell types of a dataset: its own list (e.g. cell lines) or the default broad PBMC types."""
    return ds.get("types", TYPES)


def _table(ds):
    from palm_bulk.core import PseudobulkTable
    return PseudobulkTable.from_h5ad(ds["path"], {}, _types(ds))


def _benchmark(table, ds, split):
    from palm_bulk.core import make_benchmark
    return make_benchmark(table, min_cells=20, seed=split, truth_split=True, contrast=ds["contrast"], n_test=ds["n_test"],
                          ref_filter=ds.get("ref_filter"), all_test_specimens=ds.get("all_test_specimens", False))
HELDOUT_FROM = 10
BASE = dict(activity="common", use_m=False, num_warmup=500, num_samples=500, chains=2, target_accept=0.95, check=False)
PRESETS = {
    # fractions from NB with delta given them (best fractions so far; hosts posterior reallocation),
    # and the current default for reference
    "night": {
        "2step quasi": dict(likelihood="quasi", delta_given_eta=True),
        "joint quasi+sw": dict(likelihood="quasi", sandwich=True),
    },
    # second night: honest delta intervals in the two-step route, the NB joint fractions (lowest
    # fraction error before) as a host for reallocation, and the per-type activity prior
    "night2": {
        "2step quasi+sw": dict(likelihood="quasi", delta_given_eta=True, sandwich=True),
        "joint nb": dict(likelihood="nb"),
        "2step quasi indep": dict(likelihood="quasi", delta_given_eta=True, activity="independent"),
    },
}
BASELINES = ["naive", "reallocation", "oracle realloc"]
COMPETITOR_KW = {"bayesprism": dict(n_cores=8, max_cells_per_type=1000),
                 "cellgeometry": dict(max_cells_per_type=5000),
                 "blade": dict(grid="small", n_jobs=8)}
_CELLS = {}


def _reference_cells(path, test_donors, cell_types, genes):
    """Cells of reference donors only (test donors' cells never used), kept types, genes aligned."""
    import anndata as ad
    import scipy.sparse as sps
    if path not in _CELLS:
        a = ad.read_h5ad(path)
        a.obs["cell_type"] = a.obs["cell_type"].astype(str)
        _CELLS[path] = a
    a = _CELLS[path]
    c = a[(~a.obs["donor_id"].astype(str).isin(test_donors)).values & a.obs["cell_type"].isin(cell_types).values]
    gi = {g: i for i, g in enumerate(c.var_names)}
    X = sps.csr_matrix(c.X)
    cols = [gi.get(g, -1) for g in genes]
    X = sps.hstack([X[:, [j]] if j >= 0 else sps.csr_matrix((X.shape[0], 1)) for j in cols]).tocsr()         if list(c.var_names) != list(genes) else X
    return dict(counts=X, cell_types=c.obs["cell_type"].values, donors=c.obs["donor_id"].astype(str).values)
ACT = ["act_spearman_across_samples", "act_spearman_shared", "act_spearman_specific", "act_spearman_across_types"]
METRICS = (["frac_rmse", "frac_cov90"] + ACT + ["act_gain", "act_leakage", "act_prog_leakage", "act_coupling", "act_mae", "act_cov90",
                                                 "act_iscore", "eff_pearson", "eff_auroc", "eff_localise", "eff_mae",
                                                 "eff_iscore", "eff_fsr", "eff_fdr", "eff_n_called",
                                                 "eff_cov90"] + ["ulm_" + m for m in ACT])
N_DRAWS = 100


def _splits(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s)


def _score(res, truth, truth_ulm):
    from palm_bulk import compare
    s = compare.score(res, truth)
    su = compare.score(res, truth_ulm)
    s.update({"ulm_" + k: su.get(k) for k in ACT + ["act_leakage", "act_prog_leakage", "act_coupling"]})
    return s


def _design(T):
    T = np.asarray(T)
    return lambda d: d[..., T == 1, :, :].mean(-3) - d[..., T == 0, :, :].mean(-3)   # (..., N, C, P) -> (..., C, P)


def _palm_results(name, A, design):
    """[(row, MethodResult)] of a PALM fit from its saved arrays (thinned draws: pi, delta, beta, ...)."""
    from palm_bulk.compare.base import MethodResult
    from palm_bulk.bayes.diagnostics import interval
    pi_med, ivf, delta = np.median(A["pi"], 0), interval(A["pi"]), A["delta"]
    med, iv = np.median(delta, 0), dict(fractions=ivf, activity=interval(delta))
    prd, prn = A["realloc_draws"], A["realloc_alloc"]
    out = [("model", MethodResult(name, fractions=pi_med, activity=med, effects=np.median(A["beta"], 0),
                                  intervals=dict(iv, effects=interval(A["beta"])))),
           ("design beta", MethodResult(name, fractions=pi_med, activity=med, effects=np.median(design(delta), 0),
                                        intervals=dict(iv, effects=interval(design(delta))))),
           ("stage-1 mode", MethodResult(name, fractions=A["mode_pi"], activity=A["mode_delta"],
                                         effects=design(A["mode_delta"]))),
           ("posterior realloc", MethodResult(name, fractions=pi_med, activity=A["realloc"],
                                              effects=design(A["realloc"]),
                                              intervals=dict(fractions=ivf, activity=interval(prd),
                                                             effects=interval(design(prd))))),
           ("realloc + alloc", MethodResult(name, fractions=pi_med, activity=np.median(prn, 0),
                                            effects=np.median(design(prn), 0),
                                            intervals=dict(fractions=ivf, activity=interval(prn),
                                                           effects=interval(design(prn)))))]
    for key, row in (("em_realloc", "EM realloc"), ("iter_realloc", "iter realloc")):
        if key in A:
            out.append((row, MethodResult(name, fractions=pi_med, activity=A[key], effects=design(A[key]))))
    return out


def _save_rows_npz(path, results):
    """Arrays of single-estimate rows as {row}__fractions / __activity / __effects."""
    arrs = {}
    for row, r in results:
        for k in ("fractions", "activity", "effects"):
            v = getattr(r, k)
            if v is not None:
                arrs["%s__%s" % (row, k)] = np.asarray(v, float)
    np.savez_compressed(path, **arrs)


def _rows_from_npz(name, z, design):
    from palm_bulk.compare.base import MethodResult
    if "delta" in z.files and "beta" in z.files:                      # a PALM fit's draws
        return _palm_results(name, {k: z[k] for k in z.files}, design)
    rows = sorted({k.split("__")[0] for k in z.files})
    return [(row, MethodResult(name, **{k: z["%s__%s" % (row, k)] for k in ("fractions", "activity", "effects")
                                        if "%s__%s" % (row, k) in z.files})) for row in rows]


def rescore(args):
    odir = _odir(args)
    done = skipped = 0
    for f in sorted(glob.glob(os.path.join(odir, "s*__*.json"))):
        stem = f[:-5]
        split = int(os.path.basename(f).split("__")[0][1:])
        if not os.path.exists(stem + ".npz"):
            skipped += 1
            continue
        t = np.load(os.path.join(odir, "s%d__truth.npz" % split))
        design = _design(t["T"])
        truth = dict(pi=t["pi"], delta=t["delta"], beta=design(t["delta"]))
        truth_ulm = dict(pi=t["pi"], delta=t["ulm"])
        old = {r["row"]: r for r in json.load(open(f))}
        name = next(iter(old.values()))["config"]
        rows = [{**old.get(row, dict(config=name, row=row, dataset=args.dataset, split=split)),
                 **_score(res, truth, truth_ulm)} for row, res in _rows_from_npz(name, np.load(stem + ".npz"), design)]
        with open(f, "w") as fh:
            json.dump(rows, fh, indent=1, default=float)
        done += 1
    print("rescored %d files; %d without arrays left as they were" % (done, skipped))


def run(args):
    import palm_bulk as pb
    pb.configure_compute("cpu", parallel_chains=2, precision="float64")
    from palm_bulk.core import PseudobulkTable, make_benchmark
    from palm_bulk.core.scores import estimate_delta
    from palm_bulk import compare
    from palm_bulk.compare.base import MethodResult
    from palm_bulk.compare.activity import activity_from_expression
    from palm_bulk.compare.methods import naive_two_step, reallocation
    from palm_bulk.bayes.twostage import run_palm_fast
    from palm_bulk.bayes.diagnostics import interval
    from scipy.special import softmax

    splits = _splits(args.splits)
    if any(s >= HELDOUT_FROM for s in splits) and not args.heldout:
        sys.exit("splits >= %d are the held-out set: pass --heldout (only at the method freeze)" % HELDOUT_FROM)
    configs = PRESETS[args.configs] if args.configs in PRESETS else json.loads(args.configs)
    ds = DATASETS[args.dataset]
    odir = _odir(args)
    os.makedirs(odir, exist_ok=True)
    table = _table(ds)
    prog = _programs(pb, args.prior)
    rng = np.random.default_rng(0)

    for split in splits:
        comp = [c + (" cells" if args.cells else "") for c in args.competitors.split(",") if c]
        todo = [c for c in BASELINES + comp + list(configs)
                if not os.path.exists(os.path.join(odir, "s%d__%s.json" % (split, _slug(c))))]
        if not todo:
            continue
        b = _benchmark(table, ds, split)
        inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
        truth = b.truth(inp)
        gi = [b.reference.genes.index(g) for g in inp.genes]
        ci = [b.cell_types.index(c) for c in inp.cell_types]
        ulm, _ = activity_from_expression(b.truth_counts[:, ci][:, :, gi], inp, "ulm", prog)
        truth_ulm = dict(pi=truth["pi"], delta=ulm)
        data = types.SimpleNamespace(inputs=inp, treatment=b.treatment, programs=prog)
        T = np.asarray(b.treatment)
        Y, Bm, Gm = inp.Y, inp.B, inp.Gmat

        def score(res):
            return _score(res, truth, truth_ulm)

        def save(name, rows):
            for r in rows:
                r.update(dataset=args.dataset, split=split, act_specific_share=score(
                    MethodResult("t", activity=truth["delta"]))["act_specific_share"])
            with open(os.path.join(odir, "s%d__%s.json" % (split, _slug(name))), "w") as f:
                json.dump(rows, f, indent=1, default=float)

        tpath = os.path.join(odir, "s%d__truth.npz" % split)
        if not os.path.exists(tpath):
            np.savez_compressed(tpath, pi=truth["pi"], delta=truth["delta"], ulm=ulm, T=T)

        def realloc(pi, noise=False, delta=None):
            prof = Bm[None] if delta is None else softmax(np.log(Bm)[None] + np.einsum("ncp,pg->ncg", delta, Gm), -1)
            w = pi[:, :, None] * prof
            w = w / w.sum(1, keepdims=True)                                        # (N, C, G) shares
            if noise:
                Z = rng.multinomial(np.rint(Y).astype(np.int64), np.moveaxis(w, 1, -1)).astype(float)
                Z = np.moveaxis(Z, -1, 1)
            else:
                Z = Y[:, None, :] * w
            return estimate_delta(Z, Bm, Gm)

        design = _design(T)

        def stem(name):
            return os.path.join(odir, "s%d__%s" % (split, _slug(name)))

        def compare_data():
            if args.cells:
                sc = _reference_cells(args.cells, set(b.meta["donor_id"].astype(str)), b.cell_types, b.reference.genes)
            else:
                rr = b.reference_rows
                sc = dict(counts=rr["counts"], cell_types=rr["cell_types"], donors=rr["donors"])
            return compare.CompareData(bulk=b.bulk, sc_counts=sc["counts"], sc_genes=b.reference.genes,
                                       sc_cell_types=sc["cell_types"], sc_donors=sc["donors"], inputs=inp,
                                       programs=prog, treatment=b.treatment,
                                       workdir=os.path.join(ROOT, "benchmarks", "dev", "work", args.dataset,
                                                            "s%d" % split))

        for name in todo:
            t0 = time.time()
            print("[%s] split %d: %s" % (time.strftime("%H:%M"), split, name), flush=True)
            try:
                if name == "oracle realloc":
                    a = realloc(truth["pi"])
                    res = MethodResult(name, fractions=truth["pi"], activity=a, effects=design(a))
                    _save_rows_npz(stem(name) + ".npz", [("model", res)])
                    save(name, [dict(config=name, row="model", **score(res), time=time.time() - t0)])
                    continue
                if name in comp:
                    method = name.replace(" cells", "")
                    res, err = compare.run_method(method, compare_data(), **COMPETITOR_KW.get(method, {}))
                    if res is None:
                        raise RuntimeError(err)
                    results = [("model", res)]
                    if method == "bayesprism":
                        a = realloc(res.fractions)
                        results.append(("realloc", MethodResult(name, fractions=res.fractions, activity=a,
                                                                effects=design(a))))
                    _save_rows_npz(stem(name) + ".npz", results)
                    rows = [dict(config=name, row=rw, **score(rr)) for rw, rr in results]
                    rows[0]["time"] = time.time() - t0
                    save(name, rows)
                    continue
                if name in ("naive", "reallocation"):
                    res = (naive_two_step if name == "naive" else reallocation)(data)
                    _save_rows_npz(stem(name) + ".npz", [("model", res)])
                    save(name, [dict(config=name, row="model", **score(res), time=time.time() - t0)])
                    continue
                r = run_palm_fast(inp, **{**BASE, **configs[name]})
                d = r.draws
                delta = r._delta_by_cell_type(slice(None))                       # (S, N, C, P)
                beta = d["beta"] if not r.grouped else d["beta"][:, inp.groups]
                pi_med = np.median(d["pi"], 0)
                fit = dict(rhat=d["rhat_max"], ess=d["ess_min"], div=d["divergences"], time=time.time() - t0,
                           phi_median=d["stage1"].get("phi_median"),
                           sd_delta=float(np.median(delta, 0).std(0).mean()),
                           sd_delta_truth=float(truth["delta"].std(0).mean()))
                med = np.median(delta, 0)
                sm = d["stage1_mode"]
                idx = np.linspace(0, d["pi"].shape[0] - 1, N_DRAWS).astype(int)
                pr = realloc(pi_med)
                it = pr
                for _ in range(3):
                    it = realloc(pi_med, delta=np.nan_to_num(it))
                A = dict(pi=d["pi"][idx], delta=delta[idx], beta=beta[idx], realloc=pr,
                         realloc_draws=np.stack([realloc(d["pi"][s]) for s in idx]),
                         realloc_alloc=np.stack([realloc(d["pi"][s], noise=True) for s in idx]),
                         em_realloc=realloc(pi_med, delta=np.nan_to_num(med)), iter_realloc=it,
                         mode_pi=sm["pi"], mode_delta=sm["delta"])
                np.savez_compressed(stem(name) + ".npz", **A)
                # metrics from the thinned draws, exactly as `rescore` computes them
                rows = [dict(config=name, row=rw, **(fit if rw == "model" else {}), **score(rr))
                        for rw, rr in _palm_results(name, A, design)]
                save(name, rows)
            except Exception as e:                       # keep the queue going; the row is retried next run
                print("  FAILED: %r" % (e,), flush=True)


def summary(args):
    files = glob.glob(os.path.join(_odir(args), "s*__*.json"))
    df = pd.DataFrame([r for f in files for r in json.load(open(f))])
    if df.empty:
        sys.exit("no rows")
    mets = [m for m in METRICS + ["act_specific_share", "rhat", "div", "time"] if m in df]
    g = df.groupby(["config", "row"])
    mean = g[mets].mean()
    se = g[mets].std() / np.sqrt(g.size().values[:, None])
    n = g["split"].nunique().rename("splits")
    pd.set_option("display.width", 250)
    print("== %s: mean over splits" % args.dataset)
    print(pd.concat([n, mean], axis=1).round(3).to_string())
    print("\n== standard error over splits")
    print(se.round(3).to_string())
    if args.ref:
        rc, _, rr = args.ref.partition("|")
        ref = df[(df.config == rc) & (df.row == (rr or "model"))].set_index("split")
        rng = np.random.default_rng(0)
        print("\n== paired difference vs '%s', mean [95%% bootstrap CI over splits]" % args.ref)
        out = {}
        for (c, rw), sub in df.groupby(["config", "row"]):
            if c == rc and rw == (rr or "model"):
                continue
            sub = sub.set_index("split")
            common = sub.index.intersection(ref.index)
            res = {}
            for m in METRICS:
                if m in sub and m in ref:
                    dlt = (sub.loc[common, m] - ref.loc[common, m]).astype(float).dropna().values
                    if len(dlt) >= 2:
                        bs = rng.choice(dlt, (2000, len(dlt))).mean(1)
                        res[m] = "%+.3f [%+.3f, %+.3f]" % (dlt.mean(), *np.percentile(bs, [2.5, 97.5]))
            if res:
                out["%s | %s" % (c, rw)] = res
        print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summary", "rescore"])
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--splits", default="0-9")
    p.add_argument("--configs", default="night", help="preset name or JSON {name: run_palm_fast kwargs}")
    p.add_argument("--heldout", action="store_true", help="allow the held-out splits (only at the method freeze)")
    p.add_argument("--ref", default=None, help="'config|row' to compare against in the summary (row defaults to model)")
    p.add_argument("--out", default=OUT)
    p.add_argument("--prior", default="progeny7", choices=list(PRIORS), help="program prior set (see PRIORS)")
    p.add_argument("--competitors", default="", help="comma-separated compare methods, e.g. cellgeometry,bayesprism")
    p.add_argument("--cells", default=None, help="cell-level h5ad for the competitors' reference (reference donors only)")
    a = p.parse_args()
    dict(run=run, summary=summary, rescore=rescore)[a.cmd](a)
