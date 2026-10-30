"""Unattended experiment queue for method development (simulation, PBMCpedia pseudo-bulk, OneK1K).

Runs the jobs below a few at a time and is safe to restart: a job whose output file exists is skipped,
so re-running after a crash or reboot continues where it stopped. Progress goes to
<outdir>/status.json and <outdir>/SUMMARY.md (updated after every job); each job's log is in
<outdir>/logs/<name>.log.

Usage:  python scripts/campaign.py [--outdir benchmarks/campaign] [--slots 3] [--only name1,name2]
                                   [--wait file1,file2 --wait-hours 6] [--dry-run]
Set PALM_RSCRIPT for InstaPrism/TOAST/TCA (see docs/user_guide.md).
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--outdir", default="benchmarks/campaign")
p.add_argument("--slots", type=int, default=3, help="jobs running at once")
p.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per job")
p.add_argument("--only", default=None, help="comma-separated job names")
p.add_argument("--wait", default="", help="comma-separated files to wait for before starting (other runs)")
p.add_argument("--wait-hours", type=float, default=6.0)
p.add_argument("--input-hours", type=float, default=48.0,
               help="give up on jobs whose input files are still missing after this long")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

PY = sys.executable
OUT = args.outdir
PBMC = "pbmc_broad_clean.h5ad"
O1K = "data/onek1k_50k_%s.h5ad"
O1K_FULL = os.environ.get("ONEK1K_FULL", "C:/Users/duy.pham/Downloads/onek1k_full_%s.h5ad")
PROGS7 = "JAK-STAT,NFkB,TNFa,TGFb,Hypoxia,p53,MAPK"
COMP = "naive_two_step,reallocation,toast,tca,instaprism"
FINE_MAP = json.dumps({"B naive": "B cell", "B memory": "B cell", "B intermediate": "B cell",
                       "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte", "NK": "NK cell",
                       "NK_CD56bright": "NK cell", "NK Proliferating": "NK cell",
                       "CD4 Naive": "CD4 T", "CD4 TCM": "CD4 T", "CD4 TEM": "CD4 T", "CD4 CTL": "CD4 T",
                       "Treg": "CD4 T", "CD4 Proliferating": "CD4 T",
                       "CD8 Naive": "CD8 T", "CD8 TCM": "CD8 T", "CD8 TEM": "CD8 T", "CD8 Proliferating": "CD8 T"})


def compare(name, methods, reps, kw=None, est_h=1.0, **opt):
    """One scripts/run_compare.py job. opt: extra CLI options (pseudobulk=..., N=..., ...)."""
    out = os.path.join(OUT, name + ".json")
    cmd = [PY, "scripts/run_compare.py", "--methods", methods, "--reps", str(reps), "--warmup", "1000",
           "--samples", "1000", "--precision", "float64", "--out", out,
           "--workdir", os.path.join(OUT, "work", name)]
    if kw:
        cmd += ["--method-kw", json.dumps(kw)]
    for k, v in opt.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return dict(name=name, out=out, cmd=cmd, est_h=est_h)


def pbmc(name, methods, reps, kw=None, est_h=1.0, **opt):
    return compare(name, methods, reps, kw, est_h, pseudobulk=PBMC, max_genes=opt.pop("max_genes", 1000),
                   program_list=opt.pop("program_list", PROGS7), **opt)


def onek1k(name, methods, reps, kw=None, est_h=1.0, level="broad", **opt):
    return compare(name, methods, reps, kw, est_h, pseudobulk=O1K % level, max_genes=1000,
                   program_list=PROGS7, min_cells=10, **opt)


def onek1k_full(name, methods, reps, kw=None, est_h=1.0, level="broad", **opt):
    """Full OneK1K (981 donors). Competitors use donor pseudo-bulk rows as their reference (the
    517k-cell file is too large to hand to them on a 20 GB machine). Waits until the file exists."""
    j = compare(name, methods, reps, kw, est_h, pseudobulk=O1K_FULL % level, max_genes=1000,
                program_list=PROGS7, min_cells=opt.pop("min_cells", 20), **opt)
    j["needs"] = [O1K_FULL % level]
    return j


fast = lambda act: {"palm_fast": {"activity": act}}
full = lambda act: {"palm": {"activity": act}}

# activity: 'independent' is the default; on PBMC (activity test, 3 splits) 'shared' ranked samples
# better (rho 0.56 vs 0.46) but hurt effects (r 0.36 vs 0.45, cov90 0.39 vs 0.66), so it is only compared in A/B/C4.
# Priority order. est_h: rough hours for the job alone on a 6-core desktop (3 at once take ~2-3x longer).
JOBS = [
    # A. does the two-stage route match the full model? (PBMC, the 3 splits of the activity test)
    pbmc("A1_pbmc_fast_independent", "palm_fast", 3, fast("independent"), 0.5),
    pbmc("A2_pbmc_fast_shared", "palm_fast", 3, fast("shared"), 0.5),
    pbmc("A3_pbmc_fast_shared_only", "palm_fast", 3, fast("shared_only"), 0.5),
    # B. PBMC benchmark, 5 donor splits, final settings
    pbmc("B1_pbmc5_competitors", COMP, 5, None, 2.5),
    pbmc("B2_pbmc5_fast_shared", "palm_fast", 5, fast("shared"), 0.7),
    pbmc("B3_pbmc5_fast_independent", "palm_fast", 5, fast("independent"), 0.7),
    pbmc("B4_pbmc5_palm_independent", "palm", 5, full("independent"), 4.0),
    pbmc("B5_pbmc5_palm_shared", "palm", 5, full("shared"), 5.0),
    # D. OneK1K 50k subset (39 donors; small): second dataset, no disease contrast
    onek1k("D1_onek1k_broad", "naive_two_step,reallocation,instaprism,palm_fast,palm", 3, fast("independent"), 3.0,
           cells=O1K % "cells", contrast="none"),
    onek1k("D2_onek1k_broad_sex", "naive_two_step,reallocation,toast,tca,palm_fast", 3, fast("independent"), 1.0,
           cells=O1K % "cells", contrast="sex,male,female"),
    onek1k("D3_onek1k_5types", "naive_two_step,reallocation,palm_fast", 3, fast("independent"), 1.0, level="fine",
           label_map=FINE_MAP, keep_types="B cell,Monocyte,NK cell,CD4 T,CD8 T", contrast="none"),
    # C. simulation with exact truth: sample size (independent per-type variation, the simulator's truth)
    compare("C1_sim_N24", COMP + ",palm,palm_fast", 5, None, 3.0, N=24, genes=1500, max_genes=1000),
    compare("C2_sim_N48", COMP + ",palm,palm_fast", 5, None, 5.0, N=48, genes=1500, max_genes=1000),
    compare("C3_sim_N96", "naive_two_step,reallocation,palm,palm_fast", 5, None, 8.0, N=96, genes=1500,
            max_genes=1000),
    compare("C4_sim_N48_shared", "palm_fast", 5, fast("shared"), 0.5, N=48, genes=1500, max_genes=1000),
    # E. sensitivity of the fast route
    pbmc("E1_pbmc_fast_genes500", "palm_fast", 3, fast("independent"), 0.3, max_genes=500),
    pbmc("E2_pbmc_fast_genes2000", "palm_fast", 3, fast("independent"), 0.8, max_genes=2000),
    pbmc("E3_pbmc_fast_all_programs", "naive_two_step,palm_fast", 3, fast("independent"), 1.0,
         program_list="Androgen,EGFR,Estrogen,Hypoxia,JAK-STAT,MAPK,NFkB,PI3K,TGFb,TNFa,Trail,VEGF,WNT,p53"),
    # F. full OneK1K (981 donors; 616 with >= 20 cells of each broad type), after the plan above
    onek1k_full("F1_onek1k_full_broad", "naive_two_step,reallocation,instaprism,palm_fast", 3, fast("independent"),
                4.0, contrast="none", n_test=150),
    onek1k_full("F2_onek1k_full_sex", "naive_two_step,reallocation,toast,tca,palm_fast", 3, fast("independent"),
                3.0, contrast="sex,male,female", n_test=300),
    onek1k_full("F3_onek1k_full_5types", "naive_two_step,reallocation,palm_fast", 3, fast("independent"), 2.0,
                level="fine", label_map=FINE_MAP, keep_types="B cell,Monocyte,NK cell,CD4 T,CD8 T",
                contrast="none", n_test=150),
    onek1k_full("F4_onek1k_full_palm", "palm", 2, full("independent"), 8.0, contrast="sex,male,female",
                n_test=150),
]


def write_status(state, t0):
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(updated=datetime.datetime.now().isoformat(timespec="seconds"),
                   hours=round((time.time() - t0) / 3600, 2), jobs=state), open(os.path.join(OUT, "status.json"), "w"),
              indent=1)
    lines = ["# Campaign summary", "", "Updated %s. Job table: status.json; logs: logs/." % datetime.datetime.now()
             .strftime("%Y-%m-%d %H:%M"), ""]
    for j in JOBS:
        s = state[j["name"]]
        lines.append("## %s: %s%s" % (j["name"], s["status"], " (%.1f h)" % s["hours"] if s.get("hours") else ""))
        if os.path.exists(j["out"]):
            res = json.load(open(j["out"]))
            cols = ["frac_rmse", "frac_cov90", "act_spearman_across_samples", "act_spearman_across_types",
                    "act_cov90", "eff_pearson", "eff_auroc", "eff_mae", "eff_cov90", "runtime"]
            lines.append("")
            lines.append("| method | " + " | ".join(cols) + " |")
            lines.append("|---" * (len(cols) + 1) + "|")
            for r in res["summary"]:
                lines.append("| %s | " % r["method"] + " | ".join(
                    "%.3f" % r[c] if isinstance(r.get(c), (int, float)) and r[c] == r[c] else "" for c in cols) + " |")
            if res.get("skipped"):
                lines.append("")
                lines.append("skipped: %s" % res["skipped"])
        lines.append("")
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(lines))


_seen = {}


def inputs_ready(j):
    """All `needs` files exist and their size has not changed for 5 minutes (download finished)."""
    for f in j.get("needs", []):
        if not os.path.exists(f):
            return False
        size, now = os.path.getsize(f), time.time()
        if _seen.get(f, (None,))[0] != size:
            _seen[f] = (size, now)
            return False
        if now - _seen[f][1] < 300:
            return False
    return True


def main():
    jobs = JOBS if not args.only else [j for j in JOBS if j["name"] in args.only.split(",")]
    if args.dry_run:
        for j in jobs:
            print("%-28s ~%.1fh  %s" % (j["name"], j["est_h"], " ".join(j["cmd"][1:])))
        print("total ~%.0f h alone; with %d slots ~%.0f-%.0f h wall" % (
            sum(j["est_h"] for j in jobs), args.slots, sum(j["est_h"] for j in jobs) / args.slots * 2,
            sum(j["est_h"] for j in jobs) / args.slots * 3))
        return
    waits = [w for w in args.wait.split(",") if w]
    t_wait = time.time()
    while waits and not all(os.path.exists(w) for w in waits) and time.time() - t_wait < args.wait_hours * 3600:
        time.sleep(60)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    state = {j["name"]: dict(status="done (earlier)" if os.path.exists(j["out"]) else "queued") for j in JOBS}
    t0, running = time.time(), {}
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    write_status(state, t0)
    while todo or running:
        for name, (proc, j, ts, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                ok = proc.returncode == 0 and os.path.exists(j["out"])
                state[name].update(status="done" if ok else "failed (exit %s)" % proc.returncode,
                                   hours=round((time.time() - ts) / 3600, 2),
                                   finished=datetime.datetime.now().isoformat(timespec="minutes"))
                del running[name]
                write_status(state, t0)
        while todo and len(running) < args.slots:
            ready = [j for j in todo if inputs_ready(j)]
            if not ready:
                if not running and time.time() - t0 > args.input_hours * 3600:
                    for j in todo:
                        state[j["name"]]["status"] = "skipped (input missing: %s)" % j.get("needs")
                    todo = []
                    write_status(state, t0)
                break
            j = ready[0]
            todo.remove(j)
            log = open(os.path.join(OUT, "logs", j["name"] + ".log"), "w")
            log.write("$ %s\n" % " ".join(j["cmd"]))
            log.flush()
            proc = subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env)
            running[j["name"]] = (proc, j, time.time(), log)
            state[j["name"]].update(status="running", started=datetime.datetime.now().isoformat(timespec="minutes"))
            write_status(state, t0)
        time.sleep(30)
    write_status(state, t0)
    print("campaign finished", datetime.datetime.now())


if __name__ == "__main__":
    main()
