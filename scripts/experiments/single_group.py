"""Single-group cohorts (e.g. TB patients only): docs/benchmarking.md 0.8. Development splits 0-9.

No comparison group: PALM runs without treatment and gives each patient's calibrated per-cell-type activity
with a 90% interval; program2cell.associate then relates it to a patient-level measure z. Datasets:
  mcf_grad_ida / mcf_grad_tram  McFarland gradient mixtures (build_mcfarland.py MIX_VARIANT=gradient): each
                 "patient" has z ~ U(0, 1), the share of drug-treated cells in every line; 30 of 40 patients per
                 split; reference and calibration from the reference mixtures (control cells)
  combat_covid   COMBAT COVID-19 patients only (paired real bulk + single-cell truth); z = severity
                 (HCW-mild 1, mild 2, severe 3, critical 4); 40 test patients per split; reference and calibration
                 from every other person's single-cell data. Bulks: sc (single-cell pseudobulk), real (raw),
                 real_self (correct_platform on the COVID test bulk alone: the TB-only situation), real_self_hv
                 (offsets estimated on the COVID test bulk + the healthy volunteers' bulk, applied to the COVID
                 samples), real_paired (offsets from paired samples of non-test people)
Methods: palm (calibrated activity, PALM-lite fractions, intervals; association by correlating it with z),
palm_unmix (program2cell.associate_unmix: unmixing with z as a continuous treatment; association metrics only, plus
cov_change = coverage of its change interval for the true change over the z range), instaprism (per-type expression scored with
estimate_delta), reallocation (NNLS fractions), naive (one bulk shift copied to every type).
Scores (truth = each test sample's held-out per-type activity):
  S1 cov       coverage of PALM's 90% activity intervals
  S2 rank      median Spearman across patients, estimated vs true activity, over (type, program) pairs whose true
               between-patient SD >= 0.05
  S5 bias      mean over pairs of |mean over patients (estimated - true)|
  S3 assoc     slope of activity on z (associate): pairs with a true change over the z range >= 0.1 are
               positives, < 0.03 nulls; power = positives called with the right sign; fpr = nulls called;
               perm_fpr = calls with z permuted (all pairs)
  S4 auroc     AUROC of |estimated slope| for positives vs nulls; slope_r = r of estimated vs true slopes
One json per (dataset, split, bulk, method) in benchmarks/dev/single_group/ (arrays in a matching npz).
Usage: python single_group.py run DATASETS SPLITS   |   python single_group.py summary
"""
import glob, json, os, shutil, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import dev_bench as db

OUT = os.path.join(ROOT, "benchmarks", "dev", "single_group")
MCF = os.path.join(ROOT, "data", "pert", "mcfarland_mix_gradient.h5ad")
COMBAT = os.path.join(ROOT, "data", "combat")
COMBAT_TYPES = ["B", "CD14 Mono", "CD16 Mono", "CD4 T", "CD8 T", "DC", "NK"]
SEVERITY = {"COVID_HCW_MILD": 1, "COVID_MILD": 2, "COVID_SEV": 3, "COVID_CRIT": 4}
POS, NULL = 0.1, 0.03
METHODS = ["palm", "palm_unmix", "instaprism", "reallocation", "naive"]


def _score(est, truth, z, lo=None, hi=None, seed=0):
    import program2cell as pb
    N, C, P = truth.shape
    s = {}
    if lo is not None:
        s["cov"] = float(((truth >= lo) & (truth <= hi)).mean())
        s["width"] = float((hi - lo).mean())
    sd = truth.std(0)
    rk = [spearmanr(est[:, c, p], truth[:, c, p])[0] for c in range(C) for p in range(P) if sd[c, p] >= 0.05]
    rk = [x for x in rk if np.isfinite(x)]
    s["rank"] = float(np.median(rk)) if rk else np.nan
    s["n_rank_pairs"] = len(rk)
    s["bias"] = float(np.abs((est - truth).mean(0)).mean())
    a = pb.associate(est, z, n_boot=200, n_perm=499, seed=seed)
    t = pb.associate(truth, z, n_boot=20, n_perm=19, seed=seed)
    dz = np.ptp(z)
    dtrue, dest = t["slope"] * dz, a["slope"] * dz
    pos, nul = np.abs(dtrue) >= POS, np.abs(dtrue) < NULL
    s["n_pos"], s["n_null"] = int(pos.sum()), int(nul.sum())
    s["power"] = float((a["calls"] & (np.sign(dest) == np.sign(dtrue)))[pos].mean()) if pos.any() else np.nan
    s["fpr"] = float(a["calls"][nul].mean()) if nul.any() else np.nan
    rng = np.random.default_rng(seed + 1)
    s["perm_fpr"] = float(pb.associate(est, rng.permutation(z), n_boot=20, n_perm=499, seed=seed)["calls"].mean())
    if pos.any() and nul.any():
        from program2cell.compare import known_answer as ka
        sc = np.abs(dest)[pos | nul]
        s["auroc"] = float(ka._auroc(sc, pos[pos | nul]))
    s["slope_r"] = float(np.corrcoef(dest.ravel(), dtrue.ravel())[0, 1])
    s["calls"] = int(a["calls"].sum())
    return s, dict(slope=a["slope"], p=a["p"], calls=a["calls"], true_slope=t["slope"])


def _score_unmix(inp, pi, truth, z, seed=0):
    import program2cell as pb
    from program2cell.compare import known_answer as ka
    a = pb.associate_unmix(inp, pi, z, seed=seed)
    t = pb.associate(truth, z, n_boot=20, n_perm=19, seed=seed)
    dtrue, dest = t["slope"] * np.ptp(z), a["change"]
    pos, nul = np.abs(dtrue) >= POS, np.abs(dtrue) < NULL
    s = dict(n_pos=int(pos.sum()), n_null=int(nul.sum()), calls=int(a["calls"].sum()),
             cov_change=float(((dtrue >= a["lo"]) & (dtrue <= a["hi"])).mean()), width=float((a["hi"] - a["lo"]).mean()))
    s["power"] = float((a["calls"] & (np.sign(dest) == np.sign(dtrue)))[pos].mean()) if pos.any() else np.nan
    s["fpr"] = float(a["calls"][nul].mean()) if nul.any() else np.nan
    s["perm_fpr"] = float(pb.associate_unmix(inp, pi, np.random.default_rng(seed + 1).permutation(z), seed=seed)["calls"].mean())
    if pos.any() and nul.any():
        s["auroc"] = float(ka._auroc(np.abs(dest)[pos | nul], pos[pos | nul]))
    s["slope_r"] = float(np.corrcoef(dest.ravel(), dtrue.ravel())[0, 1])
    return s, dict(change=dest, lo=a["lo"], hi=a["hi"], calls=a["calls"], true_change=dtrue)


def _compare(b, Y, prog, inp, workdir):
    """Competitor activity (N, C, P) on bulk Y: instaprism, reallocation, naive."""
    from program2cell import compare
    from program2cell.compare import methods as cm
    rr = b.reference_rows
    data = compare.CompareData(bulk=Y, sc_counts=rr["counts"], sc_genes=b.reference.genes, sc_cell_types=rr["cell_types"],
                               sc_donors=rr["donors"], inputs=inp, programs=prog, treatment=None, workdir=workdir)
    out = {}
    res, err = compare.run_method("instaprism", data, n_cores=4, max_cells_per_type=1000)
    shutil.rmtree(workdir, ignore_errors=True)
    if res is not None:
        out["instaprism"] = np.asarray(res.activity)
    else:
        print("  instaprism FAILED: %s" % err, flush=True)
    out["reallocation"] = np.asarray(cm.reallocation(data).activity)
    out["naive"] = np.asarray(cm.naive_two_step(data).activity)
    return out


def _run_one(dsn, split, bulk_name, b, Y, table, prog, z, test_donors, cal_kw):
    import program2cell as pb
    todo = [m for m in METHODS if not os.path.exists(os.path.join(OUT, "%s__s%d__%s__%s.json" % (dsn, split, bulk_name, m)))]
    if not todo:
        return
    inp = pb.prepare_inputs(Y, b.reference, prog, max_genes=1000)
    truth = b.truth(inp)["delta"]
    cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split, **cal_kw)
    t0 = time.time()
    r = pb.program_deconvolution(inp, cal, engine="lite", treatment=np.zeros(len(z)), seed=split)
    ests = {"palm": (r.activity, r.activity_interval[0], r.activity_interval[1], time.time() - t0)}
    if "palm_unmix" in todo:
        t0 = time.time()
        s, arr = _score_unmix(inp, r.fractions, truth, z, seed=split)
        s.update(dataset=dsn, split=split, bulk=bulk_name, method="palm_unmix", n=int(len(z)), seconds=time.time() - t0)
        stem = os.path.join(OUT, "%s__s%d__%s__palm_unmix" % (dsn, split, bulk_name))
        np.savez_compressed(stem + ".npz", z=z, **arr)
        json.dump(s, open(stem + ".json", "w"), default=float)
        print("[%s] %s s%d %-13s palm_unmix   cov_change %.2f power %s fpr %s perm_fpr %.2f auroc %s (%.0fs)" % (
            time.strftime("%H:%M"), dsn, split, bulk_name, s["cov_change"],
            "%.2f" % s["power"] if np.isfinite(s["power"]) else "nan", "%.2f" % s["fpr"] if np.isfinite(s["fpr"]) else "nan",
            s["perm_fpr"], "%.2f" % s["auroc"] if "auroc" in s else "nan", s["seconds"]), flush=True)
    if set(todo) - {"palm", "palm_unmix"}:
        t0 = time.time()
        wd = os.path.join(ROOT, "benchmarks", "dev", "work", "single_group", "%s_s%d_%s" % (dsn, split, bulk_name))
        for m, act in _compare(b, Y, prog, inp, wd).items():
            ests[m] = (act, None, None, time.time() - t0)
    for m in todo:
        if m not in ests:
            continue
        est, lo, hi, sec = ests[m]
        s, arr = _score(np.asarray(est), truth, z, lo, hi, seed=split)
        s.update(dataset=dsn, split=split, bulk=bulk_name, method=m, n=int(len(z)), seconds=sec)
        stem = os.path.join(OUT, "%s__s%d__%s__%s" % (dsn, split, bulk_name, m))
        np.savez_compressed(stem + ".npz", est=est, truth=truth, z=z, **({} if lo is None else dict(lo=lo, hi=hi)), **arr)
        json.dump(s, open(stem + ".json", "w"), default=float)
        print("[%s] %s s%d %-13s %-12s cov %s rank %.2f bias %.3f power %s fpr %s perm_fpr %.2f auroc %s" % (
            time.strftime("%H:%M"), dsn, split, bulk_name, m, "%.2f" % s["cov"] if "cov" in s else "  - ",
            s["rank"], s["bias"], "%.2f" % s["power"] if np.isfinite(s["power"]) else "nan",
            "%.2f" % s["fpr"] if np.isfinite(s["fpr"]) else "nan", s["perm_fpr"],
            "%.2f" % s["auroc"] if "auroc" in s else "nan"), flush=True)


def run_mcf(dsn, splits):
    import program2cell as pb
    from program2cell.core import PseudobulkTable, make_benchmark
    table = PseudobulkTable.from_h5ad(MCF, {}, db.MCF_LINES)
    prog = db._programs(pb, "progeny7")
    label = "grad_" + dsn.split("_")[-1]
    zmap = table.obs.drop_duplicates("specimen").set_index("specimen")["z"]
    for split in splits:
        b = make_benchmark(table, min_cells=20, seed=split, truth_split=True, contrast=("label", label, label),
                           n_test=30, ref_filter=("label", "refctrl"))
        z = zmap.reindex(b.bulk.columns).values.astype(float)
        test_donors = set(b.meta["donor_id"].astype(str))
        _run_one(dsn, split, "assembled", b, b.bulk, table, prog, z, test_donors, dict(ref_filter=("label", "refctrl")))


def run_combat(splits):
    import program2cell as pb
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    import combat_real as cr
    bk = ad.read_h5ad(os.path.join(COMBAT, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    path = os.path.join(COMBAT, "combat_sc_pseudobulk_covid_only.h5ad")
    if not os.path.exists(path):
        sc = ad.read_h5ad(os.path.join(COMBAT, "combat_sc_pseudobulk.h5ad"))
        paired = sc.obs["specimen"].isin(set(bk.obs_names))
        sc.obs["grp"] = np.where(paired & sc.obs["label"].isin(list(SEVERITY)), "COVID", "other")
        sc.write_h5ad(path)
    table = PseudobulkTable.from_h5ad(path, {}, COMBAT_TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    hv = [s for s in bulk_all.columns if bk.obs.loc[s, "label"] == "HV"]
    sev = table.obs.drop_duplicates("specimen").set_index("specimen")["label"].map(SEVERITY)
    prog = db._programs(pb, "progeny7")
    for split in splits:
        b = make_benchmark(table, cell_types=COMBAT_TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("grp", "COVID", "COVID"), n_test=40)
        specs = list(b.bulk.columns)
        z = sev.reindex(specs).values.astype(float)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        test_donors = set(b.meta["donor_id"].astype(str))
        real = bulk_all.loc[genes, specs]
        other = [s for s in bulk_all.columns if s.split("|")[0] not in test_donors]
        off_paired = cr._offsets(bulk_all, table, other, genes)
        off_hv = pb.platform_offsets(bulk_all.loc[genes, specs + [h for h in hv if h not in specs]], b.reference)
        bulks = {"sc": b.bulk.loc[genes], "real": real,
                 "real_self": pb.correct_platform(real, b.reference),
                 "real_self_hv": pb.correct_platform(real, b.reference, offsets=off_hv),
                 "real_paired": pb.correct_platform(real, b.reference, offsets=off_paired.clip(-8, 8))}
        for name, Y in bulks.items():
            _run_one("combat_covid", split, name, b, Y, table, prog, z, test_donors, {})


def summary():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    if not rows:
        sys.exit("no rows")
    df = pd.DataFrame(rows)
    df["set"] = np.where(df.split >= 20, "held-out", "development")
    cols = [c for c in ("cov", "cov_change", "width", "rank", "bias", "power", "fpr", "perm_fpr", "auroc", "slope_r", "calls") if c in df]
    key = ["set", "dataset", "bulk", "method"]
    t = df.groupby(key)[cols].mean()
    t.insert(0, "n", df.groupby(key).size())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t.round(3).to_string())
    print("\ntargets: cov >= 0.85 (PALM), fpr and perm_fpr <= 0.1, auroc >= 0.9; rank and power vs comparators")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        os.makedirs(OUT, exist_ok=True)
        splits = db._splits(sys.argv[3])
        for dsn in sys.argv[2].split(","):
            run_combat(splits) if dsn == "combat_covid" else run_mcf(dsn, splits)
