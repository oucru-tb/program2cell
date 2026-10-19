"""Real-data utilities: turn scRNA-seq reference + bulk counts + program weights into model inputs.

Conventions
-----------
* Bulk counts: pandas DataFrame, genes x samples (raw integer counts, not TPM/FPKM).
* Reference scRNA-seq: raw counts cells x genes with a cell-type label per cell and, ideally,
  a donor label per cell (donor-aware pseudobulks make the baseline less donor-driven).
* Program dictionary: long table with columns [program, gene, weight] (e.g. PROGENy), or a
  wide DataFrame (programs x genes). Weights act on the log scale: delta = 1 means a gene with
  weight 1 is multiplied by e in that cell type.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls


@dataclass
class Reference:
    B: np.ndarray                 # (C, G) share of reads per gene in each cell type
    genes: list
    cell_types: list
    rna_content: np.ndarray       # (C,) mean UMI per cell, used to convert read to cell fractions
    n_cells: np.ndarray           # (C,)
    n_donors: np.ndarray = None   # (C,) donors contributing to each cell type
    hvg_score: np.ndarray = None  # (G,) cell-level normalised dispersion (higher = more variable)
    lineage: list = None          # (C,) lineage / level-1 label of each cell type, if known


@dataclass
class Inputs:
    Y: np.ndarray                 # (N, G) bulk counts of modelled genes
    L: np.ndarray                 # (N,) library size over the modelled genes
    B: np.ndarray                 # (C, G) reference baseline renormalised on modelled genes
    Gmat: np.ndarray              # (P, G) program weights
    genes: list
    samples: list
    cell_types: list
    programs: list
    T: np.ndarray = None          # (N,) 0/1 treatment indicator or None
    rna_content: np.ndarray = None
    info: dict = field(default_factory=dict)
    X: np.ndarray = None          # (N, K) sample covariates, numeric, centred (one-hot for categories)
    covariates: list = None       # names of the K covariate columns
    lineages: list = None         # (C,) lineage of each cell type (from the reference), if known
    groups: np.ndarray = None     # (C,) index into group_names when program shifts are shared by lineage
    group_names: list = None


# --------------------------------------------------------------------------
def hvg_scores(X, n_bins=20, chunk=20000):
    """Highly-variable-gene score per gene (Seurat-style normalised dispersion): counts scaled to
    1e4 per cell, dispersion = var / mean, log, then z-scored within 20 bins of mean expression.
    Processed in row chunks so large references (1e5+ cells) fit in memory. Unexpressed genes: -inf."""
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    n, Gn = X.shape
    s1, s2 = np.zeros(Gn), np.zeros(Gn)
    for k in range(0, n, chunk):
        Xk = X[k:k + chunk].astype(np.float64)
        lib = np.asarray(Xk.sum(axis=1)).ravel()
        Xk = sp.diags(1e4 / np.maximum(lib, 1)) @ Xk
        s1 += np.asarray(Xk.sum(axis=0)).ravel()
        s2 += np.asarray(Xk.multiply(Xk).sum(axis=0)).ravel()
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0) * n / max(n - 1, 1)
    score = np.full(Gn, -np.inf)
    ok = mean > 0
    disp = np.log(np.maximum(var[ok], 1e-12) / mean[ok])
    lm = np.log1p(mean[ok])
    bins = np.digitize(lm, np.quantile(lm, np.linspace(0, 1, n_bins + 1)[1:-1]))
    z = np.zeros_like(disp)
    for b in np.unique(bins):
        m = bins == b
        sd = disp[m].std()
        z[m] = (disp[m] - disp[m].mean()) / sd if sd > 0 else 0.0
    score[ok] = z
    return score


def build_reference(counts, genes, cell_types, donors=None, min_cells=10, floor=1e-6, compute_hvg=True,
                    lineages=None):
    """Donor-aware pseudobulk baseline. counts: cells x genes (dense or scipy sparse).
    compute_hvg: also score highly variable genes across cells (used by select_genes).
    lineages: optional per-cell coarse label (e.g. annotation level 1) for fine cell types; each
    cell type must belong to exactly one lineage. Enables lineage summaries and shared programs."""
    X = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(np.asarray(counts))
    ct = np.asarray(cell_types).astype(str)
    dn = np.zeros(len(ct), dtype=str) if donors is None else np.asarray(donors).astype(str)
    types = [str(t) for t in sorted(set(ct))]
    Gn = X.shape[1]
    B = np.zeros((len(types), Gn))
    rna, ncell, ndon = np.zeros(len(types)), np.zeros(len(types), int), np.zeros(len(types), int)
    for k, c in enumerate(types):
        in_c = ct == c
        tot = np.asarray(X[in_c].sum(axis=1)).ravel()
        rna[k], ncell[k] = tot.mean(), int(in_c.sum())
        shares = []
        for d in sorted(set(dn[in_c])):
            sel = in_c & (dn == d)
            if sel.sum() >= min_cells:
                s = np.asarray(X[sel].sum(axis=0)).ravel()
                shares.append(s / s.sum())
        if not shares:                                   # too few cells per donor: pool
            s = np.asarray(X[in_c].sum(axis=0)).ravel()
            shares = [s / s.sum()]
        B[k] = np.mean(shares, axis=0)
        ndon[k] = len(shares)
    B = (B + floor) / (B + floor).sum(axis=1, keepdims=True)
    hvg = hvg_scores(X) if compute_hvg else None
    lin = None
    if lineages is not None:
        pairs = pd.DataFrame({"ct": ct, "lin": np.asarray(lineages).astype(str)}).drop_duplicates()
        bad = pairs.ct[pairs.ct.duplicated()].unique()
        if len(bad):
            raise ValueError("cell types assigned to more than one lineage: %s" % list(bad))
        lin = list(pairs.set_index("ct").loc[types, "lin"])
    return Reference(B=B, genes=list(genes), cell_types=types, rna_content=rna, n_cells=ncell, n_donors=ndon,
                     hvg_score=hvg, lineage=lin)


def from_anndata(adata, celltype_key, donor_key=None, layer=None, lineage_key=None, **kw):
    """Build a Reference from an AnnData with raw counts in .X (or `layer`). lineage_key: optional
    obs column with the coarse label (annotation level 1) of each fine cell type."""
    X = adata.layers[layer] if layer else adata.X
    donors = adata.obs[donor_key].values if donor_key else None
    lin = adata.obs[lineage_key].values if lineage_key else None
    return build_reference(X, adata.var_names, adata.obs[celltype_key].values, donors, lineages=lin, **kw)


def load_programs(source, scale="median_abs"):
    """Return a programs x genes DataFrame. `source`: path to CSV, long DataFrame
    [program, gene, weight], or wide DataFrame. scale: 'median_abs' rescales each program so the
    median |weight| of its genes is 1 (recommended; keeps delta ~ O(1)); None leaves weights as is."""
    df = pd.read_csv(source) if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__") else source
    if {"program", "gene", "weight"} <= set(df.columns):
        df = df.pivot_table(index="program", columns="gene", values="weight", aggfunc="mean").fillna(0.0)
    df = df.astype(float)
    if scale == "median_abs":
        for p in df.index:
            nz = np.abs(df.loc[p].values[df.loc[p].values != 0])
            if len(nz):
                df.loc[p] = df.loc[p] / np.median(nz)
    return df


GENE_BUDGET = dict(programs=0.45, markers=0.30, hvg=0.10, anchors=0.15)


def _round_robin(ranked_lists, budget, taken):
    """Take the best remaining gene from each list in turn until `budget` new genes are taken."""
    out, pos = [], [0] * len(ranked_lists)
    while len(out) < budget:
        moved = False
        for j, lst in enumerate(ranked_lists):
            while pos[j] < len(lst) and lst[pos[j]] in taken:
                pos[j] += 1
            if pos[j] < len(lst) and len(out) < budget:
                g = lst[pos[j]]
                out.append(g); taken.add(g); pos[j] += 1; moved = True
        if not moved:
            break
    return out


def select_genes(ref, bulk, programs=None, max_genes=1500, budget=None, min_ref_cpm=2.0,
                 min_bulk_detect=0.2, return_info=False):
    """Choose the modelled genes as a union of four sets, each serving one part of the model:

    * programs (45%): top genes of each program by |weight|, taken in turn so every program is
      represented -> program activity and attribution;
    * markers (30%): genes specific to each cell type, taken in turn per type -> cell fractions,
      so program signal is not confused with composition;
    * hvg (10%): highly variable genes across reference cells (ref.hvg_score) -> cell-state
      variation that markers alone miss;
    * anchors (15%): well-expressed genes with similar expression in all cell types and no program
      weight -> profiles are renormalised over the modelled genes, so stable genes keep induced
      programs from making every other gene look repressed.

    Eligible genes are in both datasets, expressed in the reference (>= min_ref_cpm reads per
    million in at least one cell type; the model cannot induce a gene absent from the baseline) and,
    if `bulk` is a genes x samples DataFrame, detected in >= min_bulk_detect of samples (lenient, so
    genes induced in one arm only are kept). Budget left unused by a set passes to the others.
    `bulk` may also be a list of gene names (no detection filter)."""
    share = dict(GENE_BUDGET, **(budget or {}))
    bulk_genes = bulk.index if isinstance(bulk, pd.DataFrame) else bulk
    bset = set(bulk_genes)
    elig = np.array([g in bset for g in ref.genes])
    elig &= ref.B.max(0) * 1e6 >= min_ref_cpm
    if isinstance(bulk, pd.DataFrame):
        det = (bulk.reindex(ref.genes).fillna(0).values > 0).mean(1)
        elig &= det >= min_bulk_detect
    genes_e = [g for g, e in zip(ref.genes, elig) if e]
    Be = ref.B[:, elig]
    taken, sets = set(), {}

    # programs: per program, eligible genes by |weight|
    prog_lists = []
    if programs is not None:
        Pm = programs.reindex(columns=genes_e).fillna(0.0)
        for p in Pm.index:
            w = Pm.loc[p].abs()
            prog_lists.append(list(w[w > 0].sort_values(ascending=False, kind="stable").index))
    n_prog_avail = len(set().union(*map(set, prog_lists))) if prog_lists else 0
    quota = {k: int(round(v * max_genes)) for k, v in share.items()}
    sets["programs"] = _round_robin(prog_lists, min(quota["programs"], n_prog_avail), taken)
    program_genes = set().union(*map(set, prog_lists)) if prog_lists else set()

    # markers: per cell type, specificity among reasonably expressed genes
    C = Be.shape[0]
    lB = np.log(Be + 1e-9)
    expr_min = np.quantile(Be.max(0), 0.5) if Be.size else 0.0
    marker_lists = []
    for c in range(C):
        score = lB[c] - np.log(np.delete(Be, c, axis=0).mean(0) + 1e-9) if C > 1 else lB[c]
        score = np.where(Be[c] >= expr_min, score, -np.inf)
        order = np.argsort(-score, kind="stable")
        marker_lists.append([genes_e[i] for i in order if np.isfinite(score[i])])

    # hvg: reference cell-level variability
    hvg_list = []
    if ref.hvg_score is not None:
        hs = ref.hvg_score[elig]
        hvg_list = [genes_e[i] for i in np.argsort(-hs, kind="stable") if np.isfinite(hs[i])]

    # anchors: top half by expression, lowest spread across cell types, not in any program
    mean_e = Be.mean(0)
    cv = Be.std(0) / np.maximum(mean_e, 1e-12)
    ok = (mean_e >= np.quantile(mean_e, 0.5)) & np.array([g not in program_genes for g in genes_e])
    anchor_list = [genes_e[i] for i in np.argsort(cv, kind="stable") if ok[i]]

    # fill markers, hvg, anchors; leftover (from short sets) goes round the sets again
    remaining = max_genes - len(sets["programs"])
    sets["markers"] = _round_robin(marker_lists, min(quota["markers"], remaining), taken)
    remaining = max_genes - len(taken)
    sets["hvg"] = _round_robin([hvg_list], min(quota["hvg"], remaining), taken)
    remaining = max_genes - len(taken)
    sets["anchors"] = _round_robin([anchor_list], min(quota["anchors"], remaining), taken)
    for name, lists in [("programs", prog_lists), ("markers", marker_lists), ("hvg", [hvg_list]),
                        ("anchors", [anchor_list])]:
        sets[name] += _round_robin(lists, max_genes - len(taken), taken)
    genes = sets["programs"] + sets["markers"] + sets["hvg"] + sets["anchors"]
    if not return_info:
        return genes
    top_type = np.argmax(Be, axis=0)
    ge = {g: i for i, g in enumerate(genes_e)}
    per_type = np.bincount([top_type[ge[g]] for g in sets["markers"]], minlength=C)
    info = dict(n_eligible=int(elig.sum()), sizes={k: len(v) for k, v in sets.items()},
                program_genes_available=n_prog_avail,
                markers_per_type=dict(zip(map(str, ref.cell_types), per_type.tolist())))
    return genes, info


def encode_covariates(cov, samples):
    """DataFrame (samples x covariates) -> centred numeric matrix. Categorical columns are one-hot
    coded (first level dropped); numeric columns are standardised."""
    cov = cov.loc[samples]
    if cov.isna().any().any():
        raise ValueError("covariates contain missing values in columns %s; impute or drop them first"
                         % list(cov.columns[cov.isna().any()]))
    num = cov.select_dtypes("number")
    cat = cov.drop(columns=num.columns)
    parts = [(num - num.mean()) / num.std(ddof=0).replace(0, 1)]
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat.astype(str), drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X = X - X.mean()
    return X.values.astype(float), list(X.columns)


MAX_TYPES_FULL_MODEL = 20


def prepare_inputs(bulk, ref, programs, treatment=None, max_genes=1500, min_program_genes=5,
                   gene_budget=None, covariates=None, program_groups=None, **select_kw):
    """Align bulk counts, reference and programs on a common gene set (see select_genes).
    bulk: genes x samples DataFrame of raw counts. treatment: Series/array of 0/1 per sample, optional.
    gene_budget: override the share of programs / markers / hvg / anchors, e.g. {"programs": 0.5}.
    covariates: optional DataFrame indexed by sample (e.g. batch, sex, age) adjusted for in both
    composition and program activity; treatment effects are then for a sample at the covariate mean.
    programs=None: composition-only inputs (fractions only; fast even for many cell types).
    program_groups='lineage': program activity is estimated per lineage (ref.lineage) and shared by
    its cell types, while fractions stay at cell-type level.

    Annotation level guidance: up to ~20 cell types (level 1), use the full model. With more
    (level 2 / cell states), either run two stages (programs=None at level 2 for fractions, then
    the full model on a level-1 reference) or use program_groups='lineage'."""
    if not isinstance(bulk, pd.DataFrame):
        raise TypeError("bulk must be a pandas DataFrame (genes x samples)")
    if programs is None:
        programs = pd.DataFrame(np.zeros((0, len(ref.genes))), columns=ref.genes)
    C = len(ref.cell_types)
    if programs.shape[0] and program_groups is None and C > MAX_TYPES_FULL_MODEL:
        warnings.warn("%d cell types with programs: fitting a program shift per cell type is slow and weakly "
                      "identified. Consider two stages (programs=None at this level for fractions, then a "
                      "level-1 reference for programs) or program_groups='lineage'." % C)
    groups, group_names = None, None
    if program_groups == "lineage":
        if ref.lineage is None:
            raise ValueError("program_groups='lineage' needs a reference built with lineage labels "
                             "(from_anndata(..., lineage_key=...))")
        group_names = sorted(set(ref.lineage))
        groups = np.array([group_names.index(l) for l in ref.lineage])
    elif program_groups is not None:
        raise ValueError("program_groups must be None or 'lineage'")
    n_prog_in = programs.shape[0]
    genes, gene_info = select_genes(ref, bulk, programs, max_genes, budget=gene_budget,
                                    return_info=True, **select_kw)
    gi = {g: i for i, g in enumerate(ref.genes)}
    B = ref.B[:, [gi[g] for g in genes]]
    B = B / B.sum(axis=1, keepdims=True)
    Gm = programs.reindex(columns=genes).fillna(0.0)
    n_per = (Gm.abs() > 0).sum(axis=1)
    keep = n_per >= min_program_genes
    if (~keep).any():
        warnings.warn("dropping programs with < %d genes present: %s" % (min_program_genes, list(Gm.index[~keep])))
    Gm = Gm[keep]
    Y = bulk.reindex(genes).T.values.astype(float)
    L = Y.sum(1)
    T = None
    if treatment is not None:
        T = np.asarray(pd.Series(treatment).reindex(bulk.columns).values if isinstance(treatment, pd.Series)
                       else treatment, float)
        if set(np.unique(T)) - {0.0, 1.0}:
            raise ValueError("treatment must be coded 0/1")
    info = dict(n_genes=len(genes), n_programs_in=n_prog_in, n_programs_kept=int(keep.sum()),
                program_genes_present=n_per[keep].to_dict(), n_samples=Y.shape[0],
                min_library=float(L.min()), gene_sets=gene_info)
    X, cov_names = (None, None) if covariates is None else encode_covariates(covariates, bulk.columns)
    return Inputs(Y=Y, L=L, B=B, Gmat=Gm.values, genes=genes, samples=list(bulk.columns),
                  cell_types=list(ref.cell_types), programs=list(Gm.index), T=T,
                  rna_content=ref.rna_content, info=info, X=X, covariates=cov_names,
                  lineages=ref.lineage, groups=groups, group_names=group_names)


def diagnose_inputs(inp):
    """Reference/bulk compatibility check: non-negative least-squares fit of each bulk profile onto
    the reference baselines. Low R^2 or a large unexplained fraction suggests missing cell types
    or strong platform bias (consider use_m=True)."""
    frac = inp.Y / inp.Y.sum(1, keepdims=True)
    rows = []
    for name, f in zip(inp.samples, frac):
        w, _ = nnls(inp.B.T, f)
        fit = inp.B.T @ w
        ss_res, ss_tot = ((f - fit) ** 2).sum(), ((f - f.mean()) ** 2).sum()
        lc = np.corrcoef(np.log(f + 1e-6), np.log(fit + 1e-6))[0, 1]
        rows.append(dict(sample=name, r2=1 - ss_res / ss_tot, log_corr=lc, mass_explained=w.sum()))
    return pd.DataFrame(rows).set_index("sample")
