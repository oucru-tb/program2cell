"""Bayesian inference layer (Paper 1; gold standard for Paper 2): NumPyro model, VI warm start + NUTS,
convergence diagnostics, result tables."""
from .model import make_model
from .fit import fit, svi_warm_start, model_args
from .diagnostics import interval, coverage, mean_width, convergence, convergence_problems
from .pipeline import run_palm, PalmResult, estimate_runtime
from .twostage import run_palm_fast, fit_fast
