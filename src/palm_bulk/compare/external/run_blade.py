"""Run BLADE in its own Python environment (BLADE_Deconvolution 0.0.7 pins old numpy/numba).

Usage: python run_blade.py IN.npz OUT.npz
IN.npz:  X, stdX (genes x cell types; mean and sd of log(1 + CPM) over reference cells),
         Y (genes x samples; bulk 1 + CPM, linear scale: BLADE models Y = sum_c F_c exp(Nu_c)),
         marker (genes,) bool for the fraction fit, n_jobs, n_rep, grid ('default' or 'small').
OUT.npz: fractions (samples x cell types), nu (samples x genes x cell types; log(1 + CPM) per type),
         setting (the hyperparameters BLADE selected, as text).
This file only imports numpy and BLADE so that it runs in that environment.
"""
import sys
import numpy as np
from joblib import parallel_backend
import BLADE_Deconvolution.BLADE as _blade
import BLADE_Deconvolution.BLADE_run as _run
import BLADE_Deconvolution.BLADE_wrapper as _wrapper
import BLADE_Deconvolution.BLADE_framework as _framework

# BLADE_Deconvolution 0.0.7 does `from . import X` and then calls the module X as a function, at
# three levels. Point those names at the functions. Worker processes would re-import the broken
# modules, so the parallel jobs run on threads (parallel_backend below) instead.
_run.BLADE = _blade.BLADE
_wrapper.BLADE_run = _run.BLADE_run
_framework.BLADE_wrapper = _wrapper.BLADE_wrapper
BLADE_framework = _framework.BLADE_framework

inp = np.load(sys.argv[1])
X, stdX, Y, marker = inp["X"], inp["stdX"], inp["Y"], inp["marker"].astype(bool)
n_jobs, n_rep, grid = int(inp["n_jobs"]), int(inp["n_rep"]), str(inp["grid"])

kw = {}
if grid == "small":            # 1 x 2 x 2 x 2 settings instead of 2 x 3 x 3 x 3 (for quick tests)
    kw = dict(Alphas=[1], Alpha0s=[0.1, 1], Kappa0s=[1, 0.1], SYs=[1, 0.3])
with parallel_backend("threading", n_jobs=n_jobs):
    final_obs, best_obs, best_set, _ = BLADE_framework(X, stdX, Y, Ind_Marker=list(marker), Nrep=n_rep,
                                                       Njob=n_jobs, Nrepfinal=n_rep, **kw)
np.savez(sys.argv[2], fractions=np.asarray(final_obs.ExpF(final_obs.Beta)), nu=np.asarray(final_obs.Nu),
         setting=str(best_set))
