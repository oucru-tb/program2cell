"""palm-bulk: program attribution deconvolution for bulk RNA-seq (Bayesian, CPU or GPU).

Layers: `core` (equations, data preparation, simulators; shared with the fast model of Paper 2),
`bayes` (NumPyro model and NUTS), `compare` (competing methods and scoring)."""
from .compute import configure_compute, compute_info
from .core import (cell_profiles, mix_profiles, expected_fractions, expected_counts, attribution, shares,
                   effect_decomposition, estimate_delta, bulk_delta)
from .core import (Reference, Inputs, build_reference, from_anndata, load_programs, select_genes,
                   prepare_inputs, diagnose_inputs)
from .core import data, sim
from .bayes import (fit, make_model, run_palm, PalmResult, estimate_runtime, interval, coverage, mean_width)
from .provenance import provenance
from . import plots

__version__ = "0.1.0"
