"""Comparison with competing methods (Paper 1, item 7): wrappers with one input and output format,
conversion of their outputs to program activity, and scoring against ground truth."""
from .base import CompareData, MethodResult, MethodUnavailable
from .methods import (METHODS, run_method, palm, naive_two_step, reallocation, instaprism, cellgeometry, cibersortx, blade,
                      toast, tca, nnls_fractions)
from .activity import activity_from_expression, effects_from_activity, effects_from_logfc
from .scoring import score
