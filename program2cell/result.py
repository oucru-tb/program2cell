"""Structured analysis output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Program2CellResult:
    """Tables and intermediate representations from a Program2Cell analysis."""

    modules: pd.DataFrame
    attribution: pd.DataFrame
    ci: pd.DataFrame
    metrics: pd.DataFrame
    dictionary: pd.DataFrame = field(repr=False)
    gene_weights: pd.DataFrame = field(repr=False)

