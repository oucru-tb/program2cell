# palm-bulk – program activity in cell types from bulk RNA-seq

palm-bulk estimates, from bulk RNA-seq and a single-cell reference, how active known gene programs
(e.g. [PROGENy][] pathways) are inside each cell type, together with cell-type fractions.
For treated vs control samples it splits a program's change into the part that comes from cells responding
and the part that comes from the cell mix changing.
It is a Bayesian model fitted with NUTS in [NumPyro][], so every estimate comes with a credible interval.

This is research code. It has been checked on simulated data; benchmarks on real pseudo-bulk data are under way.

## Installation

```bash
git clone https://github.com/oucru-tb/program2cell
cd program2cell
pip install -e ".[sc,progeny]"
```

Python 3.10 or newer. For an NVIDIA GPU add `pip install "jax[cuda12]"`.

## Usage

```python
import palm_bulk as pb

pb.configure_compute("cpu", precision="float64")
ref = pb.from_anndata(adata, "cell_type", "donor_id")           # raw counts
programs = pb.load_programs("progeny.csv")                       # columns: program, gene, #weight
inputs = pb.prepare_inputs(bulk, ref, programs, treatment=condition)     # bulk: genes x samples, #raw counts
res = pb.run_palm(inputs)

res.fractions()
res.cell_type_activity()
res.effect_totals()
```

See the [user guide][] for inputs, options and known limits, [the method in plain words][method] for how it
works, and `notebooks/palm_bulk_real_data.ipynb` for a full example.

### Program activity and treatment effects per cell type

`program_deconvolution` turns fractions (from PALM, PALM-lite or any other method) into calibrated program
activity per sample and cell type and, when a treatment is given, effects per cell type and program with
intervals and calls. It needs a calibration set: pseudo-bulk of reference donors who are not in the bulk.

```python
table = pb.core.PseudobulkTable.from_h5ad("ref_broad.h5ad")      # scripts/build_pseudobulk_from_cells.py
cal = pb.calibration_from_table(table, exclude_donors=bulk_donors, cell_types=inputs.cell_types,
                                genes=inputs.genes, G=inputs.Gmat)
fit = pb.run_palm_fast(inputs, likelihood="nb", activity="common", use_m=False)   # fractions
res = pb.program_deconvolution(inputs, cal, fraction_draws=fit.draws["pi"][::20])
res.table()          # activity / effects, intervals, calls, robustness value per cell type and program
```

### Settings

| Goal | Settings | Time (50 samples) | Use when |
|---|---|---|---|
| **Causal (default)** | `estimator="auto"`, `unmix_kw=dict(pev=True, misfit=True)` (both on by default) | minutes | Effects you will report. Each cell type × program pair is estimated by unmixing or by calibrated reallocation, whichever the design check finds more accurate for this study. Intervals include the treatment's response outside the programs. It was the only setting with valid intervals on all perturbation benchmarks. |
| **Fast** | `estimator="diff"` (or `"ols"` with `covariates` / `groups`), `engine="lite"` | seconds | Screening and exploring. Activity is calibrated, but effects can leak between cell types under a strong shared response (effect-interval coverage 0.63–0.67 on McFarland, against 0.92–0.97 for causal). |
| **Full Bayesian** | `pb.run_palm(inputs, dispersion="gene")`, then its draws as `fraction_draws` | tens of minutes to hours | Final fractions with the best intervals (`dispersion="fixed"`, the default, is 15–25× faster, but effect coverage drops from 0.89 to 0.82). |
| **Paired / repeated samples** | `groups=donor_ids` | — | Before/after or several samples per donor. Gives within-donor effects and resamples whole donors for intervals. |
| **Observational (e.g. disease vs healthy)** | `covariates=X`, optionally `estimator="aipw"` | — | Effects are adjusted associations, not causal. Report `robustness`: how strong an unmeasured confounder would have to be to remove a call. |
| **Composition vs cell response** | `decompose=True` | +seconds | Splits each bulk program effect into a part from changed proportions and a part from changed cell state, per cell type. |
| **Other methods' fractions** | `fractions=...` (convert cell to read fractions with `compare.methods.to_modelled_read_fractions`) or `frac_fn=method` | — | CIBERSORTx, InstaPrism, etc. A `frac_fn` that can be rerun has its own fraction error calibrated. Otherwise PALM-lite stands in for that error. |
| **Many cell types (> 20)** | fine fractions with `prepare_inputs(bulk, ref_fine, None)`, programs on a level-1 reference | — | Program activity needs enough signal per type. |

Always:
- Use `pb.configure_compute(..., precision="float64")` for real data.
- Pass known confounders (batch, sex, age) as `covariates`.
- Keep the bulk donors out of the calibration set.

**Other options:**
- `level` (default 0.9) sets the interval level.
- `n_sim` (default 10) sets how many cohorts the design check simulates.
- Experimental engines are kept for comparison. On the benchmarks, `estimator="unmix"` (unmix every pair) and `unmix_kw=dict(per_pair=True)` passed fewer targets. `estimator="direct"` (likelihood on the bulk counts; `direct_kw=dict(weights="nb")`) made the most calls, but its intervals were not valid.

**What the effects mean:**
- With a randomised treatment, an effect is the causal effect of the treatment on that program in that cell type.
- Otherwise it is an association within the cell type, adjusted for the covariates.
- In either case, the method has checked that the shift sits in that cell type, not in the cell mix or another type.
- Pairs that the bulk cannot resolve get wide intervals and no call. Kang's B and NK cells are an example: they are 4% of the reads and their fractions barely vary across samples.

## Benchmarks

`scripts/run_compare.py` compares palm-bulk with InstaPrism, BLADE, TOAST, TCA and simple two-step
baselines on simulated or pseudo-bulk data with known truth. `hpc/` has scripts to set up the
environments and run the benchmark on a cluster.

## Tests

```bash
pytest -m "not slow"   # quick
pytest                 # everything, a few minutes
```

## Citation

A manuscript is in preparation.

[PROGENy]: https://saezlab.github.io/progeny/
[NumPyro]: https://num.pyro.ai/
[user guide]: docs/user_guide.md
[method]: docs/palm_method_explained.md
