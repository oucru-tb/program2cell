# palm-bulk user guide

Bayesian deconvolution of bulk RNA-seq that infers, jointly and with uncertainty:

1. **cell fractions**,
2. **program activity inside each cell type** (e.g. IFN in CD8 T cells vs monocytes),
3. **attribution**: what share of each program's signal each cell type carries,
4. for treated vs control cohorts, **intrinsic vs composition-driven treatment effects**.

NumPyro/NUTS on JAX. Runs on CPU; an NVIDIA GPU is optional (`pip install "palm-bulk[gpu]"`, Linux/WSL2) and recommended for hundreds of samples. Status: research prototype, validated on simulations only. Not yet benchmarked on real data.

## Install

```bash
git clone <your-repo-url> palm-bulk && cd palm-bulk
pip install -e .                    # core
pip install -e ".[sc]"              # + anndata for .h5ad references
pip install -e ".[progeny]"         # + decoupler for PROGENy programs
pip install -e ".[dev]"             # + pytest, jupyter tooling
pip install -e ".[gpu]"             # + CUDA JAX (NVIDIA GPU, Linux/WSL2)
pytest -q -m "not slow"            # quick checks, under a minute
pytest -q                           # full suite with NUTS fits, about 5-10 minutes on CPU
```
Python 3.10 or newer.

## Quick start (notebook)

`notebooks/palm_bulk_real_data.ipynb` is a full tutorial: data loading, gene selection, a run-time estimate before fitting, the fit, every output, a recovery check against a hidden truth, and the benchmarks. With `USE_DEMO = True` it writes a realistic synthetic dataset in the formats below (5 immune cell types, 4 programs, 24 samples with a treatment and a confounded batch; about 10 minutes on a laptop CPU). Put your files in `data/`, set `USE_DEMO = False`, and edit the config cell.

| Input | Format |
|---|---|
| Reference scRNA-seq | `.h5ad`, raw counts, `obs` columns for cell type and (ideally) donor |
| Bulk counts | CSV, genes x samples, raw integer counts |
| Bulk metadata | optional CSV, index = sample IDs, a condition column for treatment effects, covariates (batch, sex, age, ...) |
| Programs | optional CSV `program,gene,weight` (e.g. PROGENy); if omitted the notebook fetches PROGENy via decoupler (this path was not tested offline) |

## Quick start (Python)

```python
import palm_bulk as pb
pb.configure_compute("auto", precision="float64")              # first, before any fit: device "cpu" | "gpu" | "auto"
ref  = pb.from_anndata(adata, "cell_type", "donor")             # donor-aware baseline + HVG scores
prog = pb.load_programs("programs.csv")                         # programs x genes
inp  = pb.prepare_inputs(bulk, ref, prog, treatment=T,          # bulk: genes x samples DataFrame; T: 0/1 or None
                         covariates=meta[["batch", "sex"]])     # optional confounders to adjust for
print(pb.diagnose_inputs(inp).describe())                       # reference/bulk compatibility check
print(pb.estimate_runtime(inp))                                 # predicted fit time on this machine
res  = pb.run_palm(inp)                                         # defaults: 2 chains, VI start, gene-wise dispersion
print(res.diagnostics()["problems"])                             # [] when R-hat, ESS and divergences pass
res.fractions()["median"]; res.cell_type_activity(); res.attribution(); res.program_activity()
res.effect_totals(); res.effects(); res.intrinsic_shares()      # if T is given
res.save("results")
```
### Choosing the annotation level

| Reference annotation | Cell types | Setup |
|---|---|---|
| Level 1 (lineages / broad types) | up to ~20 | full model (above) |
| Level 2 / cell states | more than 20 | **two stages (recommended):** fine fractions with `prepare_inputs(bulk, ref_fine, None)` (composition only, fast), then the full model on a level-1 reference for programs and effects |
| | | **joint alternative:** `from_anndata(..., lineage_key="level1")` and `prepare_inputs(..., program_groups="lineage")`: fine fractions plus program activity shared within lineages in one fit |

Bulk data resolve many cell types for fractions, but program activity per cell type needs enough signal;
with more than 20 types `prepare_inputs` warns. `res.lineage_fractions()` sums fine fractions by lineage.

### Defaults, chosen from the calibration benchmarks
`run_palm` starts NUTS from a short variational fit (`init="svi"`, about 2.5x more efficient, warmup 200) and fits
a dispersion per gene (`dispersion="gene"`, better interval coverage on noisy data). Pass known confounders as
`covariates`: with a batch linked to treatment, coverage of treatment effects rose from 0.25-0.44 to 0.81-1.00.
Use `precision="float64"` for real datasets: at realistic size float32 rounding shrank the NUTS step size
330-fold and raised fraction error 70% (`benchmarks/precision.json`). `run_palm` warns when float32 is used on a
large problem.

**Convergence.** `run_palm` runs 2 chains by default and reports split R-hat and effective sample size
for fractions, program activity, effects and dispersion (`res.convergence()`). It warns when R-hat > 1.01,
ESS < 100 per chain, or NUTS diverged. With one chain R-hat cannot see chains stuck in different modes.

All outputs list cell types and programs in **alphabetical order** (`res.inputs.cell_types`, `res.inputs.programs`). Align with other tables by name, never by position.

## Model in one paragraph

Each sample is a mixture of cell-type profiles. A cell type's profile is its reference baseline `B_cg` multiplied by `exp(sum_p delta_cp G_pg)`, where `delta` is how strongly program `p` is active in that cell type relative to its reference state and `G` is the program dictionary. Counts are negative binomial with a dispersion per gene. With a treatment `T`, `T` shifts composition (`gamma`) and activity (`beta`) with sparse priors, and the bulk program change is split exactly (Shapley average) into intrinsic and composition parts per cell type. Covariates shift both composition and activity. A gene-wise platform scale `m_g` absorbs scRNA-vs-bulk bias. The modelled genes are a union of top program genes, cell-type markers, highly variable genes and stable anchor genes (`select_genes`). See `src/palm_bulk/core/equations.py` (forward model) and `src/palm_bulk/bayes/model.py` (priors).

## What the simulations show

Calibration of the current options (`scripts/run_calibration.py`; results in `benchmarks/calibration/` and the tutorial's section 11) checks interval coverage, accuracy and sampler efficiency for each option. The older studies below used 3 to 4 replicates of 24 samples, 4 cell types, 3 programs, 300 genes. They are small, and the simulator matches the model except where stated. Raw numbers are in `benchmarks/`.

| Setting | intrinsic-effect MAE | 90% interval coverage (intrinsic / composition) | cell-type share L1 error |
|---|---|---|---|
| well specified | 0.0059 | 0.89 / 0.89 | 0.17 |
| hidden platform bias, model without `m_g` | 0.0107 | 0.67 / 0.67 | 0.83 |
| hidden platform bias, model with `m_g` | 0.0040 | 1.00 / 0.89 | 0.20 |
| **cell type missing from the reference** | 0.0251 | **0.00** / 0.56 | n/a |

Against a two-step baseline (NNLS fractions, then reference program scores), the joint model had about 4x lower error on intrinsic and composition effects in the well-specified causal simulation (`benchmarks/causal_summary.json`). That run predates the package refactor; the refactored `scripts/run_causal.py` gave consistent results on a 1-replicate check.

## Limits you should know before using it on real data

* **A missing cell type gives confidently wrong answers.** Its signal is absorbed by the cell types that are present, errors grow about 4x, and the intervals do not widen (0% coverage above). The built-in compatibility check (`diagnose_inputs`, an NNLS fit of each bulk profile on the reference) is only a weak detector: median R^2 was 0.83 with a missing type vs 0.91 without, and the ranges overlapped across replicates. Do not rely on it alone. Make sure the reference covers the tissue. A posterior predictive check is the next planned addition.
* Platform bias matters: keep `use_m=True` for real data.
* The causal reading of treatment effects needs randomised or ignorable treatment, a control-state reference, and a correct program dictionary. Cell-to-cell causal drivers (which cell type induces a response in another) are not identifiable from bulk data.
* Programs whose genes are expressed alike across cell types give wide attribution intervals by design.
* Not validated: real data, PROGENy/Hallmark programs, large cohorts. Run time grows with samples x genes x cell types x programs, not with reference cells. For 500 samples x 10 cell types x 14 programs x 1500 genes expect about 1-2 days per chain on a 12-core CPU in float64; use a data-centre GPU for hundreds of samples, and check `estimate_runtime` first.
* The reference is treated as known (donor-to-donor variation of the baseline profiles is not propagated).

## Comparing with other methods

`palm_bulk.compare` runs PALM and competing methods on the same inputs and scores them against truth
(`scripts/run_compare.py` on a simulated study; real pseudo-bulk next). Methods that output per-sample
per-cell-type expression get program activity on the model's own scale (`pb.estimate_delta`), or with
decoupler's ULM (`activity="ulm"`, the usual "deconvolve, then PROGENy" route; needs `pip install "palm-bulk[progeny]"`).

| Method | Runs via | Needs |
|---|---|---|
| `naive_two_step` (NNLS fractions + bulk-level program shift) | Python | nothing |
| `reallocation` (NNLS + BayesPrism-style read split with a fixed reference) | Python | nothing |
| `bayesprism` | `Rscript` + files | R with BayesPrism (`devtools::install_github("Danko-Lab/BayesPrism/BayesPrism")`) |
| `cibersortx` (fractions + high resolution) | Docker | images `cibersortx/fractions`, `cibersortx/hires`; token in `CIBERSORTX_USER`/`CIBERSORTX_TOKEN` |
| `blade` | another Python env | `pip install BLADE_Deconvolution` there; its interpreter in `PALM_BLADE_PYTHON` |
| `toast`, `tca` (cell-type-specific DE -> program effects) | `Rscript` + files | Bioconductor TOAST / CRAN TCA |

Set `PALM_RSCRIPT` if `Rscript` is not on PATH. Missing software is reported and the method skipped.
Executables inside a conda env are run with that env's folders on PATH, so no activation is needed.

Setting up the external tools in their own conda envs (Windows; Linux is the same with `bioconductor-*`
packages from bioconda):

```bash
mamba create -n palm-r -c conda-forge r-base=4.4 r-devtools r-biocmanager r-matrix r-remotes
# in palm-r's R:
#   options(repos = BiocManager::repositories())        # keeps the annotation repo (GenomeInfoDbData)
#   BiocManager::install(c("GenomeInfoDbData", "TOAST")); install.packages("TCA")
#   remotes::install_github("Danko-Lab/BayesPrism/BayesPrism", upgrade = "never", dependencies = TRUE)
mamba create -n palm-blade -c conda-forge python=3.8 pip numpy scipy scikit-learn numba joblib
#   then in palm-blade: pip install BLADE_Deconvolution      (0.0.7; run_blade.py patches its import bug)
export PALM_RSCRIPT=<miniforge>/envs/palm-r/Lib/R/bin/x64/Rscript.exe
export PALM_BLADE_PYTHON=<miniforge>/envs/palm-blade/python.exe
```

BayesPrism, TOAST and TCA have been run through these wrappers on simulated data; CIBERSORTx has not
(it needs Docker and a token).

## Layout

```
src/palm_bulk/
  core/        shared by both papers: equations.py (forward model; numpy, JAX or torch arrays),
               scores.py (estimate_delta: program activity of any observed profile on the model's scale),
               data.py (reference building, gene selection, covariates), sim.py (simulators, demo data)
  bayes/       Bayesian inference: model.py (NumPyro priors + likelihood), fit.py (VI warm start, NUTS),
               diagnostics.py (intervals, R-hat, ESS), pipeline.py (run_palm, PalmResult, estimate_runtime)
  compare/     competing methods with one input/output format, and scoring against truth:
               methods.py, activity.py, scoring.py; R/ (BayesPrism, TOAST, TCA), external/ (BLADE)
  compute.py   CPU/GPU, precision, parallel chains;  plots.py;  provenance.py  commit + versions saved with results
notebooks/     palm_bulk_real_data.ipynb (executed demo)
scripts/       run_calibration.py, run_gene_budget.py, run_compare.py, collect_benchmarks.py;
               run_s1.py, run_causal.py, run_misspec.py (earlier studies)
benchmarks/    saved results with provenance (palm_benchmarks.json is read by the tutorial)
tests/         unit and end-to-end tests (`pytest -m "not slow"` skips the NUTS fits)
```
