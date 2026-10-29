"""palm-bulk: program attribution deconvolution for bulk RNA-seq (Bayesian, CPU or GPU).

Layers: `core` (equations, data preparation, simulators, reallocation, calibration; shared with the fast
model of Paper 2), `bayes` (NumPyro model and NUTS), `compare` (competing methods and scoring).
`program_deconvolution` runs the frozen pipeline on fractions from PALM, PALM-lite or any method."""
from .compute import configure_compute, compute_info
from .core import (cell_profiles, mix_profiles, expected_fractions, expected_counts, attribution, shares,
                   effect_decomposition, estimate_delta, bulk_delta)
from .core import (Reference, Inputs, build_reference, from_anndata, load_programs, select_genes,
                   prepare_inputs, diagnose_inputs)
from .core import data, sim
from .core import (reallocate, realloc_counts, em_fractions, CalibrationSet, Calibrator, build_calibration_set,
                   calibration_from_table)
from .bayes import (fit, make_model, run_palm, run_palm_fast, PalmResult, estimate_runtime, interval, coverage,
                     mean_width)
from .provenance import provenance
from .deconvolution import program_deconvolution, DeconvolutionResult
from . import causal
from . import plots

__version__ = "0.1.0"
