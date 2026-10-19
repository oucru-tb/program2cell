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

## Benchmarks

`scripts/run_compare.py` compares palm-bulk with BayesPrism, BLADE, TOAST, TCA and simple two-step
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
