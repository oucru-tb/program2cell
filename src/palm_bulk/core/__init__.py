"""Core layer, shared by the Bayesian method (Paper 1) and the fast model (Paper 2): model equations,
program scores on the model's scale, reference building and gene selection, simulators."""
from .equations import (cell_profiles, mix_profiles, expected_fractions, expected_counts, attribution,
                        shares, effect_decomposition)
from .scores import estimate_delta, bulk_delta
from .realloc import realloc_counts, reallocate, em_fractions
from .calibrate import (CalibrationSet, Calibrator, build_calibration_set, calibration_from_table,
                        contrast, plant_shifts)
from .unmix import leakage_operator, unmix_effects, design_check, simulate_cohort
from .data import (Reference, Inputs, build_reference, from_anndata, load_programs, select_genes,
                   encode_covariates, prepare_inputs, diagnose_inputs)
from .pseudobulk import PseudobulkTable, PseudobulkBenchmark, make_benchmark
from . import sim
