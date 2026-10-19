"""Collect benchmark outputs into benchmarks/palm_benchmarks.json (read by the tutorial notebook).

Usage: python collect_benchmarks.py OUT.json --calibration calib_*.json --gene-budget gene_budget.json
                                     [--extra extra.json]
`--extra` merges sections measured by hand (e.g. precision, sampler) given as
{"section": {"note": ..., "table": [rows]}}.
"""
import sys, json
import numpy as np


def calibration(files):
    order = ["baseline", "reparam", "svi", "float64", "shared_on_overdisp", "gene_disp", "no_covariates", "covariates",
             "defaults"]
    data = {d["config"]: d for d in (json.load(open(f)) for f in files)}
    metrics = [("pi_cov90", "fractions: 90% coverage"), ("beta_cov90", "cell-type response: 90% coverage"),
               ("I_cov90", "intrinsic effect: 90% coverage"), ("K_cov90", "composition effect: 90% coverage"),
               ("pi_mae", "fractions: error"), ("I_mae", "intrinsic effect: error"), ("K_mae", "composition effect: error"),
               ("ess_pi_med", "efficiency (ESS/1000 grads), fractions"), ("ess_IK_min", "efficiency, worst effect"),
               ("mean_steps", "steps per iteration"), ("time", "seconds per fit"), ("divergences", "divergences")]
    rows = []
    for key, label in metrics:
        row = {"metric": label}
        for name in order:
            if name in data:
                row[name] = round(float(np.mean([r[key] for r in data[name]["rows"]])), 4)
        rows.append(row)
    reps = max(len(d["rows"]) for d in data.values())
    note = ("%d replicate datasets per option (24 samples, 4 cell types, 3 programs, 300 genes, 2 chains). Coverage "
            "should be near 0.90. Pairs share data: baseline/reparam/svi/float64 use standard data; "
            "shared_on_overdisp vs gene_disp use data with gene-specific noise; no_covariates vs covariates use data "
            "with a batch confounded with treatment." % reps)
    return dict(note=note, table=rows)


def gene_budget(path):
    rows = json.load(open(path))
    rows = rows["rows"] if isinstance(rows, dict) else rows          # older files: a bare list
    keys = ["pi_rmse", "beta_mae", "kappa_mae", "beta_cov90", "kappa_cov90", "mean_steps", "time"]
    out = []
    for m in sorted({r["max_genes"] for r in rows}):
        rs = [r for r in rows if r["max_genes"] == m]
        out.append(dict(max_genes=m, **{k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys}))
    note = ("Accuracy and cost versus MAX_GENES (3000-gene transcriptome, 40 samples, 5 cell types, 4 programs of "
            "150 genes, %d replicates). beta/kappa: per-cell-type program activity; lower error and coverage near "
            "0.90 are better." % len({r["rep"] for r in rows}))
    return dict(note=note, table=out)


if __name__ == "__main__":
    args = sys.argv[1:]
    out_path = args.pop(0)
    sections, mode, files = {}, None, {"--calibration": [], "--gene-budget": [], "--extra": []}
    for a in args:
        if a in files:
            mode = a
        else:
            files[mode].append(a)
    if files["--calibration"]:
        sections["calibration"] = calibration(files["--calibration"])
    if files["--gene-budget"]:
        sections["gene_budget"] = gene_budget(files["--gene-budget"][0])
    for f in files["--extra"]:
        sections.update(json.load(open(f)))
    json.dump(sections, open(out_path, "w"), indent=1)
    print("wrote", out_path, list(sections))
