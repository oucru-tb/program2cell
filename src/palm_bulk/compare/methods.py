"""Competing methods, each returning a MethodResult aligned to data.inputs.

  palm            this package (reference point)
  naive_two_step  NNLS fractions + one bulk-level program shift per sample (no cell-type resolution)
  reallocation    NNLS fractions + bulk reads split across cell types in proportion to pi_c B_cg
                  (the expectation step of BayesPrism with a fixed reference), then programs
                  per cell type; pure Python, no extra software
  bayesprism      BayesPrism (R)                        -> per-type expression -> programs
  cibersortx      CIBERSORTx fractions + high-resolution (Docker, needs a token) -> programs
  blade           BLADE (Python, run in its own environment)                     -> programs
  toast, tca      cell-type-specific differential expression (R) -> program effects only

Programs from per-type expression: palm_bulk.compare.activity (`activity="delta"` for the model's
scale, `"ulm"` for decoupler). R, Docker and other-environment methods exchange files in
data.method_dir(name); their scripts live in palm_bulk/compare/R and palm_bulk/compare/external.
"""
import glob
import os
import shutil
import time
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.optimize import nnls

from ..core.scores import bulk_delta
from .base import MethodResult, MethodUnavailable, find_rscript, run_command, safe_names
from .activity import activity_from_expression, effects_from_activity, effects_from_logfc

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- helpers ------------------------------------------------------------------------
def _as_cmd(exe):
    """An executable path, or a list (an interpreter plus a script), as a command prefix."""
    return list(exe) if isinstance(exe, (list, tuple)) else [exe]


def nnls_fractions(Y, B):
    """Read fractions by non-negative least squares of each sample's gene shares on B (C, G)."""
    frac = Y / Y.sum(1, keepdims=True)
    pi = np.array([nnls(B.T, f)[0] for f in frac])
    return pi / pi.sum(1, keepdims=True)


def _reference_mass(data, genes):
    """(C,) share of each cell type's reference reads that fall on `genes`; converts read fractions
    over one gene set to read fractions over another."""
    X = data.sc_csr()
    gi = {g: i for i, g in enumerate(data.sc_genes)}
    cols = [gi[g] for g in genes if g in gi]
    out = []
    for c in data.inputs.cell_types:
        rows = data.sc_cell_types == c
        tot = X[rows].sum()
        out.append(X[rows][:, cols].sum() / tot if tot > 0 else np.nan)
    return np.array(out)


def to_modelled_read_fractions(frac, data, method_genes=None):
    """Read fractions over the genes a method used (all reference genes if None) -> read fractions
    over the modelled genes, which is what PALM and the truth tables report."""
    method_genes = data.sc_genes if method_genes is None else method_genes
    w = frac * _reference_mass(data, data.inputs.genes) / _reference_mass(data, method_genes)
    return w / w.sum(1, keepdims=True)


def _align_type_expression(per_type, data):
    """{cell type: DataFrame samples x genes} -> (N, C, G) on data.inputs; nan where missing."""
    inp = data.inputs
    out = np.full((len(inp.samples), len(inp.cell_types), len(inp.genes)), np.nan)
    for c, name in enumerate(inp.cell_types):
        if name in per_type:
            out[:, c] = per_type[name].reindex(index=inp.samples, columns=inp.genes).values
    return out


def _finish(name, data, frac, type_expr, activity, t0, notes="", **kw):
    act, scale = (None, "delta")
    if type_expr is not None:
        act, scale = activity_from_expression(type_expr, data.inputs, activity, data.programs)
    eff = None
    if act is not None and data.treatment is not None:
        eff = effects_from_activity(act, data.treatment)
    return MethodResult(method=name, fractions=frac, activity=act, activity_scale=scale, effects=eff,
                        effects_scale=scale, type_expression=type_expr, runtime=time.time() - t0, notes=notes,
                        **kw)


def _write_bulk(data, path, sep=",", genes=None, cpm=False):
    """Bulk counts (genes x samples) to text; cpm=True scales each sample to counts per million."""
    b = data.bulk if genes is None else data.bulk.reindex(genes)
    if cpm:
        b = b / b.sum(axis=0) * 1e6
    b.to_csv(path, sep=sep, index_label="gene")


def _write_sc_mtx(data, d, cells, genes_mask=None):
    """Reference cells (rows `cells`) as Matrix Market genes x cells, plus gene and label files."""
    X = data.sc_csr()[cells]
    genes = np.asarray(data.sc_genes)
    if genes_mask is not None:
        X, genes = X[:, genes_mask], genes[genes_mask]
    scipy.io.mmwrite(os.path.join(d, "sc_counts.mtx"), sp.csr_matrix(X).T.tocoo())
    pd.Series(genes).to_csv(os.path.join(d, "sc_genes.txt"), index=False, header=False)
    return genes


# ---- PALM ---------------------------------------------------------------------------
def palm(data, **run_kw):
    """PALM-bulk on data.inputs (keyword arguments go to run_palm)."""
    from ..bayes.pipeline import run_palm
    from ..bayes.diagnostics import interval
    t0 = time.time()
    res = run_palm(data.inputs, **run_kw)
    d = res.draws
    iv = dict(fractions=interval(d["pi"]))
    act = eff = None
    if "delta" in d:
        delta = res._delta_by_cell_type(slice(None))
        act = np.median(delta, 0)
        iv["activity"] = interval(delta)
        if res.has_treatment:
            beta = d["beta"] if not res.grouped else d["beta"][:, data.inputs.groups]
            eff, iv["effects"] = np.median(beta, 0), interval(beta)
    return MethodResult("palm", fractions=np.median(d["pi"], 0), activity=act, effects=eff, intervals=iv,
                        runtime=time.time() - t0, notes="diagnostics: %s" % res.diagnostics())


# ---- pure-Python baselines --------------------------------------------------------------
def naive_two_step(data, activity="delta"):
    """NNLS read fractions on the modelled genes; one program shift per bulk sample relative to the
    reference mixture, copied to every cell type (bulk activity cannot be split by cell type).
    `activity` is accepted for a uniform interface; the bulk shift is always on the model's scale."""
    t0 = time.time()
    inp = data.inputs
    pi = nnls_fractions(inp.Y, inp.B)
    d = bulk_delta(inp.Y, pi, inp.B, inp.Gmat)                                  # (N, P)
    act = np.repeat(d[:, None, :], len(inp.cell_types), axis=1)
    eff = effects_from_activity(act, data.treatment) if data.treatment is not None else None
    return MethodResult("naive_two_step", fractions=pi, activity=act, effects=eff, runtime=time.time() - t0,
                        notes="bulk-level program shift, identical for all cell types")


def reallocation(data, activity="delta"):
    """NNLS read fractions, then Z_icg = Y_ig pi_ic B_cg / sum_c pi_ic B_cg (reads split in proportion
    to each cell type's expected share), then programs per cell type from Z."""
    t0 = time.time()
    inp = data.inputs
    pi = nnls_fractions(inp.Y, inp.B)
    w = pi[:, :, None] * inp.B[None]                                            # (N, C, G)
    Z = inp.Y[:, None, :] * w / w.sum(1, keepdims=True)
    return _finish("reallocation", data, pi, Z, activity, t0,
                   notes="expectation step of BayesPrism with a fixed reference; NNLS fractions")


# ---- BayesPrism (R) -----------------------------------------------------------------------
def bayesprism(data, n_cores=4, max_cells_per_type=1000, activity="delta", species="hs", cleanup=True,
               rscript=None, timeout=None):
    """BayesPrism (Chu et al. 2022). Needs R with the BayesPrism package
    (devtools::install_github("Danko-Lab/BayesPrism/BayesPrism")). Fractions: final theta; per-type
    expression: get.exp() (posterior mean reads per sample, cell type and gene).
    cleanup: drop ribosomal, mitochondrial and sex-chromosome genes, as the BayesPrism tutorial
    advises (needs gene symbols for `species` 'hs' or 'mm').
    max_cells_per_type: BayesPrism holds the reference as a dense cells x genes matrix in R, so
    cells are subsampled per type (None keeps all; ~8 bytes x cells x genes of memory)."""
    rs = rscript or find_rscript()
    if not rs:
        raise MethodUnavailable("Rscript not found (set PALM_RSCRIPT or add R to PATH)")
    t0 = time.time()
    d = data.method_dir("bayesprism")
    cells = data.subsample_cells(max_cells_per_type)
    genes = _write_sc_mtx(data, d, cells)
    fwd, back = safe_names(data.inputs.cell_types, "ct")
    pd.Series([fwd[c] for c in data.sc_cell_types[cells]]).to_csv(os.path.join(d, "sc_labels.txt"),
                                                                  index=False, header=False)
    _write_bulk(data, os.path.join(d, "bulk_counts.csv"))
    run_command(_as_cmd(rs) + [os.path.join(HERE, "R", "run_bayesprism.R"), d, n_cores, species, int(cleanup)],
                d, "bayesprism", timeout)
    theta = pd.read_csv(os.path.join(d, "theta.csv"), index_col=0).rename(columns=back)
    used = pd.read_csv(os.path.join(d, "genes_used.txt"), header=None)[0].tolist()
    frac = theta.reindex(index=data.inputs.samples, columns=data.inputs.cell_types).values
    frac = to_modelled_read_fractions(frac, data, used)
    per_type = {name: pd.read_csv(os.path.join(d, "Z_%s.csv" % code), index_col=0)     # samples x genes
                for code, name in back.items() if os.path.exists(os.path.join(d, "Z_%s.csv" % code))}
    Z = _align_type_expression(per_type, data)
    return _finish("bayesprism", data, frac, Z, activity, t0,
                   notes="%d reference cells; %d genes after cleanup" % (len(cells), len(used)))


# ---- CIBERSORTx (Docker) --------------------------------------------------------------------
def cibersortx(data, username=None, token=None, max_cells_per_type=300, activity="delta", docker="docker",
               window=None, timeout=None):
    """CIBERSORTx (Newman et al. 2019) through the official Docker images cibersortx/fractions and
    cibersortx/hires. A token is free for academic use from https://cibersortx.stanford.edu;
    pass username/token or set CIBERSORTX_USER / CIBERSORTX_TOKEN. The single-cell reference is
    written as dense text, so cells are subsampled (max_cells_per_type).
    Step 1 builds a signature from the reference and estimates fractions (S-mode batch correction);
    step 2 estimates per-sample per-type expression (high-resolution mode)."""
    username = username or os.environ.get("CIBERSORTX_USER")
    token = token or os.environ.get("CIBERSORTX_TOKEN")
    if not (username and token):
        raise MethodUnavailable("CIBERSORTx needs username and token (CIBERSORTX_USER / CIBERSORTX_TOKEN)")
    if not shutil.which(docker):
        raise MethodUnavailable("docker not found")
    t0 = time.time()
    d = data.method_dir("cibersortx")
    out = os.path.join(d, "out")
    os.makedirs(out, exist_ok=True)
    cells = data.subsample_cells(max_cells_per_type)
    fwd, back = safe_names(data.inputs.cell_types, "ct")
    X = data.sc_csr()[cells]
    cpm = sp.diags(1e6 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X           # per-cell CPM
    ref = pd.DataFrame(cpm.T.toarray(), index=data.sc_genes,
                       columns=[fwd[c] for c in data.sc_cell_types[cells]])
    ref.to_csv(os.path.join(d, "sc_reference.txt"), sep="\t", index_label="GeneSymbol")
    _write_bulk(data, os.path.join(d, "mixture.txt"), sep="\t", cpm=True)
    mounts = ["-v", "%s:/src/data" % d, "-v", "%s:/src/outdir" % out]
    auth = ["--username", username, "--token", token]
    run_command([docker, "run", "--rm", *mounts, "cibersortx/fractions", *auth, "--single_cell", "TRUE",
                 "--refsample", "sc_reference.txt", "--mixture", "mixture.txt", "--rmbatchSmode", "TRUE",
                 "--perm", "0", "--verbose", "TRUE"], d, "cibersortx_fractions", timeout)
    sig = sorted(glob.glob(os.path.join(out, "*inferred_refsample.bm.K999.txt")))
    res_file = sorted(glob.glob(os.path.join(out, "CIBERSORTx_Adjusted.txt")) +
                      glob.glob(os.path.join(out, "CIBERSORTx_Results.txt")))
    if not sig or not res_file:
        raise RuntimeError("CIBERSORTx outputs not found in %s" % out)
    shutil.copy(sig[0], os.path.join(d, "signature.txt"))
    shutil.copy(res_file[0], os.path.join(d, "fractions.txt"))
    hires = [docker, "run", "--rm", *mounts, "cibersortx/hires", *auth, "--mixture", "mixture.txt",
             "--sigmatrix", "signature.txt", "--cibresults", "fractions.txt", "--verbose", "TRUE"]
    if window is not None:
        hires += ["--window", str(window)]
    run_command(hires, d, "cibersortx_hires", timeout)
    fr = pd.read_csv(os.path.join(d, "fractions.txt"), sep="\t", index_col=0).rename(columns=back)
    frac = fr.reindex(index=data.inputs.samples, columns=data.inputs.cell_types).values
    frac = to_modelled_read_fractions(frac / frac.sum(1, keepdims=True), data)
    per_type = {}
    for code, name in back.items():
        f = sorted(glob.glob(os.path.join(out, "CIBERSORTxHiRes_*_%s_Window*.txt" % code)))
        if f:
            e = pd.read_csv(f[0], sep="\t", index_col=0).T                       # samples x genes
            per_type[name] = 2.0 ** e.clip(upper=60) if _looks_log2(e.values) else e
    Z = _align_type_expression(per_type, data)
    return _finish("cibersortx", data, frac, Z, activity, t0,
                   notes="%d reference cells; high-resolution expression for %d cell types"
                   % (len(cells), len(per_type)))


def _looks_log2(v):
    """High-resolution output is linear in recent CIBERSORTx versions and log2 in some older ones."""
    v = v[np.isfinite(v)]
    return v.size > 0 and v.max() < 30


# ---- BLADE (separate Python environment) ----------------------------------------------------
def _blade_prior_cells(data):
    """BLADE's prior as its authors describe it: mean and sd across single cells of log(1 + CPM)."""
    inp = data.inputs
    gi = {g: i for i, g in enumerate(data.sc_genes)}
    cols = [gi[g] for g in inp.genes]
    X = data.sc_csr()
    lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    Xg = (sp.diags(1e6 / lib) @ X)[:, cols].tocsr()
    mean, sd = [], []
    for c in inp.cell_types:
        lx = np.log1p(Xg[data.sc_cell_types == c].toarray())
        mean.append(lx.mean(0)); sd.append(lx.std(0) + 1e-3)
    return mean, sd


def _blade_prior(data, min_sd=0.1):
    """Alternative prior: mean and sd over donors of log(1 + CPM) of each cell type's per-donor
    pseudo-bulk (closer to bulk-scale profiles, but a tight sd when donors are few; on a simulated
    study with 4 donors and platform bias it gave worse fractions than the cell-level prior).
    With one donor the sd is `min_sd`."""
    inp = data.inputs
    gi = {g: i for i, g in enumerate(data.sc_genes)}
    cols = [gi[g] for g in inp.genes]
    X = data.sc_csr()
    donors = data.sc_donors if data.sc_donors is not None else np.zeros(len(data.sc_cell_types))
    donors = np.asarray(donors).astype(str)
    mean, sd = [], []
    for c in inp.cell_types:
        in_c = data.sc_cell_types == c
        prof = []
        for dn in np.unique(donors[in_c]):
            s = np.asarray(X[in_c & (donors == dn)].sum(0)).ravel()
            if s.sum() > 0:
                prof.append(np.log1p(s[cols] / s.sum() * 1e6))
        prof = np.array(prof)
        mean.append(prof.mean(0))
        sd.append(np.maximum(prof.std(0, ddof=1) if len(prof) > 1 else 0.0, min_sd))
    return mean, sd


def blade(data, python=None, activity="delta", n_jobs=4, n_rep=3, grid="default", marker_genes=None,
          prior="cells", min_sd=0.1, timeout=None):
    """BLADE (Andrade Barbosa et al. 2021). Its package pins old dependencies, so it runs in its own
    environment: pip install BLADE_Deconvolution there and pass that interpreter as `python` (or
    set PALM_BLADE_PYTHON). Inputs on the modelled genes: the prior of each cell type's expression
    (prior='cells': mean and sd across cells of log(1 + CPM), the authors' recipe; prior='donors':
    per-donor pseudo-bulk, see _blade_prior) and bulk 1 + CPM (BLADE models bulk =
    sum_c F_c exp(Nu_c) on the linear scale); `marker_genes` restricts the fraction fit.
    grid='small' searches 8 hyperparameter settings instead of 54."""
    py = python or os.environ.get("PALM_BLADE_PYTHON")
    if not py:
        raise MethodUnavailable("BLADE needs its own Python environment (python=... or PALM_BLADE_PYTHON)")
    t0 = time.time()
    d = data.method_dir("blade")
    inp = data.inputs
    if prior not in ("cells", "donors"):
        raise ValueError("prior must be 'cells' or 'donors'")
    mean, sd = _blade_prior_cells(data) if prior == "cells" else _blade_prior(data, min_sd)
    bulk = data.bulk.reindex(inp.genes).fillna(0).values
    bulk_lin = 1.0 + bulk / data.bulk.sum(axis=0).values * 1e6
    marker = np.isin(inp.genes, marker_genes) if marker_genes is not None else np.ones(len(inp.genes), bool)
    np.savez(os.path.join(d, "blade_in.npz"), X=np.array(mean).T, stdX=np.array(sd).T, Y=bulk_lin,
             marker=marker, n_jobs=n_jobs, n_rep=n_rep, grid=grid)
    run_command(_as_cmd(py) + [os.path.join(HERE, "external", "run_blade.py"), os.path.join(d, "blade_in.npz"),
                 os.path.join(d, "blade_out.npz")], d, "blade", timeout)
    o = np.load(os.path.join(d, "blade_out.npz"))
    frac = o["fractions"]                                                       # (N, C)
    Z = np.expm1(np.clip(o["nu"], None, 30))                                    # (N, G, C) log1p CPM
    return _finish("blade", data, frac / frac.sum(1, keepdims=True), np.transpose(Z, (0, 2, 1)), activity, t0,
                   notes="fractions from linear mixing of expression (read-fraction-like); setting %s"
                   % o["setting"])


# ---- TOAST / TCA (R): cell-type-specific differential expression -----------------------------
def _csde(method, data, fractions=None, rscript=None, timeout=None):
    if data.treatment is None:
        raise ValueError("%s needs a treatment" % method)
    rs = rscript or find_rscript()
    if not rs:
        raise MethodUnavailable("Rscript not found (set PALM_RSCRIPT or add R to PATH)")
    t0 = time.time()
    inp = data.inputs
    d = data.method_dir(method)
    pi = nnls_fractions(inp.Y, inp.B) if fractions is None else np.asarray(fractions)
    fwd, back = safe_names(inp.cell_types, "ct")
    pd.DataFrame(pi, index=inp.samples, columns=[fwd[c] for c in inp.cell_types]).to_csv(
        os.path.join(d, "fractions.csv"))
    bulk = data.bulk[inp.samples]
    cpm = bulk / bulk.sum(axis=0) * 1e6
    cpm.reindex(inp.genes).to_csv(os.path.join(d, "bulk_cpm.csv"), index_label="gene")
    pd.Series(data.treatment, index=inp.samples, name="T").to_csv(os.path.join(d, "design.csv"))
    run_command(_as_cmd(rs) + [os.path.join(HERE, "R", "run_csde.R"), d, method], d, method, timeout)
    lfc = pd.read_csv(os.path.join(d, "logfc.csv"), index_col=0).rename(columns=back)    # genes x types
    lfc = lfc.reindex(index=inp.genes, columns=inp.cell_types).values.T
    eff = effects_from_logfc(lfc, inp)
    return MethodResult(method, fractions=pi, effects=eff, runtime=time.time() - t0,
                        notes="per-gene cell-type-specific log fold changes projected on programs; "
                              "fractions from %s" % ("NNLS" if fractions is None else "the caller"))


def toast(data, fractions=None, **kw):
    """TOAST csTest (Li & Wu 2019; Bioconductor 'TOAST') on bulk CPM, linear scale; effect_size
    converted to log fold change. Fractions: NNLS on the modelled genes unless given."""
    return _csde("toast", data, fractions, **kw)


def tca(data, fractions=None, **kw):
    """TCA (Rahmani et al. 2019; CRAN 'TCA') on log2(1 + CPM) with the treatment as a cell-type-level
    covariate (C1). Fractions: NNLS on the modelled genes unless given."""
    return _csde("tca", data, fractions, **kw)


METHODS = dict(palm=palm, naive_two_step=naive_two_step, reallocation=reallocation, bayesprism=bayesprism,
               cibersortx=cibersortx, blade=blade, toast=toast, tca=tca)


def run_method(name, data, **kw):
    """Run one method; returns (MethodResult or None, error message or '')."""
    try:
        return METHODS[name](data, **kw), ""
    except MethodUnavailable as e:
        return None, "unavailable: %s" % e
