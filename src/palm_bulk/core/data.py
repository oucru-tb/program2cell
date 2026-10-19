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


PRIOR_SD_ETA, PRIOR_SD_DELTA = 1.5, 0.5     # prior scales of the composition logits and program shifts


def design_matrix(B, G, pi, groups=None):
    """Sensitivity of each gene's expected share to the per-sample unknowns, at programs off.

    B (C, Gn) profiles (rows normalised here), G (P, Gn) program weights, pi (C,) fractions.
    Unknowns: composition logits eta (C) and program shifts delta (C x P, or groups x P).
    With E = rows of B, f = pi @ E (expected share of each gene):
      d f_g / d eta_k     = pi_k (E_kg - f_g)
      d f_g / d delta_kp  = pi_k E_kg (G_pg - sum_h E_kh G_ph)      (the second term: renormalisation)
    Returns (A (Gn, D) derivatives, f (Gn,), column labels)."""
    E = B / B.sum(1, keepdims=True)
    C = E.shape[0]
    f = pi @ E
    cols = [pi[k] * (E[k] - f) for k in range(C)]
    labels = ["eta:%d" % k for k in range(C)]
    if G is not None and G.shape[0]:
        Gbar = E @ G.T                                                       # (C, P)
        dd = pi[:, None, None] * E[:, None, :] * (G[None] - Gbar[:, :, None])  # (C, P, Gn)
        if groups is not None:
            groups = np.asarray(groups, int)
            dd = np.stack([dd[groups == j].sum(0) for j in range(groups.max() + 1)])
        for k in range(dd.shape[0]):
            for p in range(dd.shape[1]):
                cols.append(dd[k, p])
                labels.append("delta:%d,%d" % (k, p))
    return np.column_stack(cols), f, labels


def gene_dispersion(Y, default=30.0, shrink=0.5):
    """Negative-binomial dispersion theta per gene from bulk counts (samples x genes), by moments on
    library-size-scaled counts, shrunk towards the median on the log scale; `default` if < 4 samples."""
    if Y is None or Y.shape[0] < 4:
        return np.full(Y.shape[1] if Y is not None else 1, default)
    lib = Y.sum(1, keepdims=True)
    Z = Y / lib * np.median(lib)
    m, v = Z.mean(0), Z.var(0, ddof=1)
    phi = np.clip((v - m) / np.maximum(m, 1e-9) ** 2, 1e-3, 10.0)
    lphi = np.log(phi)
    lphi = shrink * lphi + (1 - shrink) * np.median(lphi)
    return 1.0 / np.exp(lphi)


def select_genes_fisher(B, G, Y=None, max_genes=1500, groups=None, min_gain=1e-3, pi=None, depth=None,
                        theta=None):
    """Greedy Bayesian D-optimal gene selection.

    Each gene g adds information J_g = w_g a_g a_g^T about the per-sample unknowns (a_g from
    design_matrix), with w_g = L / (f_g (1 + L f_g / theta_g)) the inverse negative-binomial variance
    of its share at library size L. Starting from the prior precision, genes are added one at a time
    to maximise log det(prior + sum J_g); because log det is submodular, the greedy choice is within
    (1 - 1/e) of the best set of that size. Sherman-Morrison keeps each step O(genes x unknowns).
    Program genes (for program shifts), markers (for fractions) and stable genes that separate the
    two (the renormalisation term) are all valued by the same criterion, and noisy genes are
    down-weighted through theta, so no group shares need to be fixed in advance.

    Y (samples x genes) bulk counts set the design point: mean NNLS fractions (floored at 2% so
    rare types still count), median library size and gene-wise dispersion; pi, depth, theta override.
    Without Y: equal fractions, depth 1e7, theta 30. Stops at max_genes or when the best gain is
    below min_gain. Returns (indices in selection order, info)."""
    C, Gn = B.shape
    if Y is not None:
        from scipy.optimize import nnls
        frac = Y / np.maximum(Y.sum(1, keepdims=True), 1)
        Bn = B / B.sum(1, keepdims=True)
        est = np.array([nnls(Bn.T, fr)[0] for fr in frac])
        est = est / np.maximum(est.sum(1, keepdims=True), 1e-12)
        pi = est.mean(0) if pi is None else pi
        depth = float(np.median(Y.sum(1))) if depth is None else depth
        theta = gene_dispersion(Y) if theta is None else theta
    pi = np.full(C, 1.0 / C) if pi is None else np.asarray(pi, float)
    pi = np.maximum(pi, 0.02)
    pi = pi / pi.sum()
    depth = 1e7 if depth is None else depth
    theta = np.broadcast_to(30.0 if theta is None else theta, (Gn,)).astype(float)
    A, f, labels = design_matrix(B, G, pi, groups)
    f = np.maximum(f, 1e-12)
    w = depth / (f * (1 + depth * f / theta))
    D = A.shape[1]
    prior = np.array([1 / PRIOR_SD_ETA ** 2 if l.startswith("eta") else 1 / PRIOR_SD_DELTA ** 2 for l in labels])
    Minv = np.diag(1 / prior)
    s = np.einsum("gd,de,ge->g", A, Minv, A)                                 # a_g^T M^-1 a_g
    gain = np.log1p(w * s)
    chosen, gains = [], []
    avail = np.ones(Gn, bool)
    for _ in range(min(max_genes, Gn)):
        cand = np.where(avail, gain, -np.inf)
        j = int(np.argmax(cand))
        if not np.isfinite(cand[j]) or cand[j] < min_gain:
            break
        chosen.append(j)
        gains.append(float(cand[j]))
        avail[j] = False
        u = Minv @ A[j]
        c = w[j] / (1 + w[j] * A[j] @ u)
        Minv -= c * np.outer(u, u)
        s -= c * (A @ u) ** 2
        gain = np.log1p(w * np.maximum(s, 0))
    stop = "max_genes" if len(chosen) == min(max_genes, Gn) else "min_gain"
    info = dict(n_selected=len(chosen), stop=stop, logdet_gain=float(np.sum(gains)),
                gain_first=gains[0] if gains else None, gain_last=gains[-1] if gains else None, gains=gains,
                design=dict(pi=pi.round(3).tolist(), depth=depth, theta_median=float(np.median(theta)),
                            unknowns=D))
    return np.array(chosen, int), info


def select_genes(ref, bulk, programs=None, max_genes=1500, budget=None, min_ref_cpm=2.0,
                 min_bulk_detect=0.2, return_info=False, method="groups", groups=None, min_gain=1e-3):
    """Choose the modelled genes. Two methods share the eligibility filters below:

    method='fisher': Bayesian D-optimal design, see select_genes_fisher. Genes are added one at a
    time by how much they increase the information about the fractions and program shifts, given
    each gene's noise; stops at `max_genes` or when a gene adds less than `min_gain` (log-det units).
    `groups` (C,) shares program shifts within lineages, as in prepare_inputs.

    method='groups' (default for now): a union of four sets, each serving one part of the model:

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
    if method == "fisher":
        Y = bulk.reindex(genes_e).fillna(0).values.T if isinstance(bulk, pd.DataFrame) else None
        Gm = None if programs is None else programs.reindex(columns=genes_e).fillna(0.0).values
        idx, info = select_genes_fisher(Be, Gm, Y, max_genes=max_genes, groups=groups, min_gain=min_gain)
        genes = [genes_e[i] for i in idx]
        if not return_info:
            return genes
        info.update(n_eligible=int(elig.sum()), method="fisher",
                    sizes=dict(program_genes=int((np.abs(Gm[:, idx]) > 0).any(0).sum()) if Gm is not None else 0,
                               other=len(idx) - (int((np.abs(Gm[:, idx]) > 0).any(0).sum()) if Gm is not None else 0)))
        return genes, info
    if method != "groups":
        raise ValueError("method must be 'groups' or 'fisher'")
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
                   gene_budget=None, covariates=None, program_groups=None, gene_selection="groups", **select_kw):
    """Align bulk counts, reference and programs on a common gene set (see select_genes).
    bulk: genes x samples DataFrame of raw counts. treatment: Series/array of 0/1 per sample, optional.
    gene_selection: 'groups' (four fixed shares, see select_genes) or 'fisher' (D-optimal design:
    genes chosen by the information they add about fractions and program shifts; max_genes is then a cap).
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
    genes, gene_info = select_genes(ref, bulk, programs, max_genes, budget=gene_budget, return_info=True,
                                    method=gene_selection, groups=groups, **select_kw)
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
