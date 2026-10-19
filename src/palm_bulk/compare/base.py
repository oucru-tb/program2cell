"""Shared containers for the method comparison.

Every method receives one `CompareData` and returns one `MethodResult` aligned to PALM's prepared
inputs (`data.inputs`: modelled genes, cell types, programs, samples, all in that order), so all
methods are scored the same way (palm_bulk.compare.scoring).
"""
from dataclasses import dataclass, field
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class CompareData:
    """Inputs for all methods.

    bulk: genes x samples raw counts (all genes, not only the modelled ones).
    sc_counts: cells x genes raw counts of the single-cell reference (dense or scipy sparse);
        sc_genes, sc_cell_types (per cell), sc_donors (per cell, optional) describe it.
    inputs: palm_bulk prepare_inputs(...) output built from the same bulk and reference; defines
        the modelled genes, the baseline B and the program matrix used for scoring.
    programs: programs x genes weights (as given to prepare_inputs), for decoupler scoring.
    treatment: 0/1 per sample (aligned with bulk columns) or None.
    workdir: where file-based methods (R, Docker, other Python environments) write their files."""
    bulk: pd.DataFrame
    sc_counts: object
    sc_genes: list
    sc_cell_types: np.ndarray
    inputs: object
    programs: pd.DataFrame = None
    treatment: np.ndarray = None
    sc_donors: np.ndarray = None
    workdir: str = "benchmarks/compare/work"

    def __post_init__(self):
        self.sc_cell_types = np.asarray(self.sc_cell_types).astype(str)
        if self.treatment is not None:
            self.treatment = np.asarray(self.treatment, float)
        missing = set(self.inputs.cell_types) - set(self.sc_cell_types)
        if missing:
            raise ValueError("cell types in inputs but not in the single-cell reference: %s" % sorted(missing))

    # ---- views on the modelled genes ----------------------------------------
    @property
    def Y(self):
        """(N, G) bulk counts on the modelled genes."""
        return self.inputs.Y

    def method_dir(self, name):
        d = os.path.abspath(os.path.join(self.workdir, name))
        os.makedirs(d, exist_ok=True)
        return d

    def sc_csr(self):
        X = self.sc_counts
        return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))

    def subsample_cells(self, max_per_type=None, seed=0):
        """Indices of reference cells, at most `max_per_type` per cell type (all if None)."""
        idx = np.arange(len(self.sc_cell_types))
        if max_per_type is None:
            return idx
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(self.sc_cell_types):
            ic = idx[self.sc_cell_types == c]
            keep.append(ic if len(ic) <= max_per_type else rng.choice(ic, max_per_type, replace=False))
        return np.sort(np.concatenate(keep))


@dataclass
class MethodResult:
    """One method's output, aligned to data.inputs (samples, cell_types, programs, genes).

    fractions: (N, C) read fractions (share of the sample's reads from each cell type).
    activity: (N, C, P) program activity per sample and cell type, or None.
    activity_scale: 'delta' (the model's log-scale shift, comparable in value) or 'score'
        (another scale, e.g. decoupler ULM t-values: compare by rank only).
    effects: (C, P) treatment shift of per-cell-type activity (comparable to the model's beta).
    effects_scale: as activity_scale.
    type_expression: (N, C, G) per-sample per-cell-type expression on the modelled genes, or None.
    intervals: optional {"fractions"|"activity"|"effects": (lo, hi)} 90% intervals.
    """
    method: str
    fractions: np.ndarray = None
    activity: np.ndarray = None
    activity_scale: str = "delta"
    effects: np.ndarray = None
    effects_scale: str = "delta"
    type_expression: np.ndarray = None
    intervals: dict = field(default_factory=dict)
    runtime: float = float("nan")
    notes: str = ""


# ---- running external tools ----------------------------------------------------
class MethodUnavailable(RuntimeError):
    """The method's software (R package, Docker image, other Python environment) is not set up."""


def find_rscript():
    """Rscript from $PALM_RSCRIPT, else PATH; None if absent."""
    return os.environ.get("PALM_RSCRIPT") or shutil.which("Rscript")


def conda_env_of(exe):
    """Prefix of the conda environment that contains executable `exe` (looks a few levels up for
    conda-meta), or None."""
    path = shutil.which(str(exe)) or str(exe)
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "conda-meta")):
            return d
        d = os.path.dirname(d)
    return None


def _env_for(exe):
    """Environment for running `exe`: if it lives in a conda env, put that env's binary folders first
    on PATH (what `conda activate` does; on Windows its DLLs are not found otherwise)."""
    env = dict(os.environ)
    prefix = conda_env_of(exe)
    if prefix:
        dirs = [prefix] + [os.path.join(prefix, *p) for p in
                           (("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"),
                            ("Scripts",), ("bin",))]
        env["PATH"] = os.pathsep.join([d for d in dirs if os.path.isdir(d)] + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = prefix
    return env


def run_command(cmd, workdir, name, timeout=None):
    """Run an external command, logging stdout/stderr to <workdir>/<name>.log; raise on failure."""
    log = os.path.join(workdir, name + ".log")
    with open(log, "w") as fh:
        p = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                           env=_env_for(cmd[0]))
    if p.returncode != 0:
        tail = open(log).read()[-3000:]
        raise RuntimeError("%s failed (exit %d); log %s:\n%s" % (name, p.returncode, log, tail))
    return log


def safe_names(names, prefix):
    """Map arbitrary labels to R/Docker-safe names (prefix0, prefix1, ...) and back."""
    fwd = {n: "%s%d" % (prefix, i) for i, n in enumerate(names)}
    return fwd, {v: k for k, v in fwd.items()}
