"""Benchmarks with exact truth from real single-cell data aggregated per specimen and cell type.

Input: a table of raw counts with one row per (specimen, cell type) and, per row, the specimen, donor,
cell type and number of cells (e.g. scripts/build_pbmc_pseudobulk.py output). From it:

  * test specimens: one per test donor, every kept cell type present with >= min_cells cells;
    bulk = sum of the specimen's rows, so read fractions are known exactly;
  * reference: built only from the other donors (cross-fitting; a donor is never on both sides),
    each cell type's profile averaged over reference donors, as build_reference does for cells;
  * truth per test specimen and cell type: read fractions on the modelled genes, cell fractions,
    and program activity delta = estimate_delta(that specimen's rows, reference profiles), i.e. the
    model's own definition; with a two-group contrast, beta = difference in mean delta.

With truth_split=True each test count is split at random into two independent halves (binomial
thinning; exact for Poisson counts, close for overdispersed ones): the bulk and the read fractions come
from one half and delta/beta from the other. Otherwise the truth's own sampling noise is also in the
bulk, which rewards methods that copy the bulk's noise (e.g. a bulk shift copied to every cell type) and
penalises shrinkage, and interval coverage is scored against a noisy target.

The same tables serve as Paper 2's Teacher 1 labels.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import Reference
from .scores import estimate_delta


@dataclass
class PseudobulkTable:
    """Rows = (specimen, cell type). counts: (rows, genes) raw counts (sparse or dense)."""
    counts: object
    genes: list
    obs: pd.DataFrame          # columns: specimen, donor_id, cell_type, n_cells (+ any metadata)

    @classmethod
    def from_h5ad(cls, path, label_map=None, keep_types=None):
        """Read a pseudo-bulk h5ad. label_map: rename/merge labels, e.g. {"DC": "Monocyte"};
        keep_types: cell types to keep after renaming (others dropped; None keeps all)."""
        import anndata as ad
        a = ad.read_h5ad(path)
        t = cls(a.X, list(a.var_names), a.obs.copy())
        return t.relabel(label_map, keep_types)

    def relabel(self, label_map=None, keep_types=None):
        """Merge or rename cell-type labels (rows of one specimen that end up with the same label are
        summed) and keep only `keep_types`."""
        obs = self.obs.copy()
        obs["cell_type"] = obs["cell_type"].astype(str)
        if label_map:
            obs["cell_type"] = obs["cell_type"].replace(label_map)
        keep = np.ones(len(obs), bool) if keep_types is None else obs["cell_type"].isin(keep_types).values
        obs, X = obs[keep], _csr(self.counts)[np.flatnonzero(keep)]
        key = obs["specimen"].astype(str) + "::" + obs["cell_type"]
        codes, uniq = pd.factorize(key)
        M = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        agg = obs.groupby(codes, sort=True)
        new = agg.first()
        new["n_cells"] = agg["n_cells"].sum().values
        new.index = uniq
        X = (M @ X).tocsr()
        new["n_counts"] = np.asarray(X.sum(1)).ravel()
        return PseudobulkTable(X, self.genes, new)

    def cells_wide(self):
        """specimens x cell types table of cell counts."""
        return self.obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum",
                                    fill_value=0)


def _csr(X):
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))


@dataclass
class PseudobulkBenchmark:
    """A cross-fitted benchmark built by make_benchmark."""
    reference: Reference
    bulk: pd.DataFrame                    # genes x test specimens, raw counts
    type_counts: np.ndarray               # (N, C, G) counts of each test specimen's cell types, all genes
    cell_counts: np.ndarray               # (N, C) cells per test specimen and cell type
    meta: pd.DataFrame                    # one row per test specimen
    treatment: np.ndarray = None          # 0/1 per test specimen for the contrast, or None
    reference_rows: dict = field(default_factory=dict)   # counts, cell_types, donors of the reference rows
    truth_counts: np.ndarray = None       # (N, C, G) held-out half for delta (truth_split=True), else None

    @property
    def cell_types(self):
        return self.reference.cell_types

    def truth(self, inputs):
        """Truth aligned to prepared inputs (their genes, cell types and programs): pi (read fractions on
        the modelled genes), cell_fractions, delta (N, C, P) on the model's scale, beta (C, P) if a
        contrast was set."""
        gi = [self.reference.genes.index(g) for g in inputs.genes]
        ci = [self.cell_types.index(c) for c in inputs.cell_types]
        Z = self.type_counts[:, ci][:, :, gi]
        tot = Z.sum(-1)
        out = dict(pi=tot / tot.sum(1, keepdims=True),
                   cell_fractions=self.cell_counts[:, ci] / self.cell_counts[:, ci].sum(1, keepdims=True))
        if inputs.Gmat.shape[0]:
            Zd = Z if self.truth_counts is None else self.truth_counts[:, ci][:, :, gi]
            out["delta"] = estimate_delta(Zd, inputs.B, inputs.Gmat)
            if self.treatment is not None:
                T = self.treatment
                out["beta"] = out["delta"][T == 1].mean(0) - out["delta"][T == 0].mean(0)
        return out


def make_benchmark(table, cell_types=None, min_cells=30, n_test=None, test_frac=0.5, contrast=None,
                   ref_min_cells=10, projects=None, seed=0, truth_split=False, ref_filter=None,
                   all_test_specimens=False):
    """Split donors into test and reference and build bulk, reference and truth.

    cell_types: types to use (default: all in the table). A test specimen needs >= min_cells cells in
        every one of them; one specimen per test donor (chosen at random) keeps samples independent.
    n_test / test_frac: number (or share) of eligible donors used for testing; every other donor,
        eligible or not, may serve as reference.
    contrast: optional (column, treated_value, control_value), e.g. ("disease", "COVID-19", "HC"):
        test specimens are restricted to the two groups and `treatment` is set (1 = treated).
    ref_min_cells: reference rows with fewer cells are ignored (too noisy as profiles).
    projects: optional list restricting test and reference to these projects.
    truth_split: build bulk and truth from independent halves of each test count (see module doc).
    ref_filter: optional (column, value): reference rows must have obs[column] == value, e.g.
        ("label", "ctrl") for a resting reference in a perturbation study.
    all_test_specimens: use every eligible specimen of a test donor (e.g. paired stim / ctrl samples,
        or pseudo-replicates) instead of one at random."""
    rng = np.random.default_rng(seed)
    obs = table.obs.copy()
    if projects is not None:
        obs = obs[obs["project"].isin(projects)]
    types = sorted(cell_types or obs["cell_type"].unique().tolist())
    obs = obs[obs["cell_type"].isin(types)]
    wide = obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum",
                           fill_value=0).reindex(columns=types, fill_value=0)
    spec = obs.drop_duplicates("specimen").set_index("specimen")
    ok = (wide >= min_cells).all(axis=1)
    if contrast is not None:
        col, tv, cv = contrast
        ok &= spec.loc[wide.index, col].isin([tv, cv]).values
    elig = spec.loc[ok[ok].index]
    donors = rng.permutation(np.array(sorted(map(str, elig["donor_id"].unique())), dtype=object))
    n = n_test if n_test is not None else int(round(test_frac * len(donors)))
    if n < 2 or n >= len(obs["donor_id"].unique()):
        raise ValueError("cannot split: %d eligible donors, %d requested for test" % (len(donors), n))
    test_donors = set(donors[:n])
    test_spec = [rng.choice(np.array(elig.index[elig["donor_id"].astype(str) == d], dtype=object))
                 for d in sorted(test_donors)]
    if all_test_specimens:
        test_spec = sorted(s_ for s_ in elig.index if str(elig.loc[s_, "donor_id"]) in test_donors)
    ref_mask = ~obs["donor_id"].astype(str).isin(test_donors) & (obs["n_cells"] >= ref_min_cells)
    if ref_filter is not None:
        ref_mask &= (obs[ref_filter[0]] == ref_filter[1]).values
    X = _csr(table.counts)
    pos = {k: i for i, k in enumerate(table.obs.index)}

    # reference: per donor and cell type sum the donor's rows, normalise, then average over donors
    ref = obs[ref_mask]
    Gn = X.shape[1]
    B = np.zeros((len(types), Gn))
    rna, ncell, ndon = np.zeros(len(types)), np.zeros(len(types), int), np.zeros(len(types), int)
    for k, c in enumerate(types):
        rc = ref[ref["cell_type"] == c]
        if rc.empty:
            raise ValueError("no reference rows for cell type %s" % c)
        prof = []
        for _, rd in rc.groupby("donor_id"):
            s = np.asarray(X[[pos[i] for i in rd.index]].sum(0)).ravel()
            prof.append(s / s.sum())
        B[k] = np.mean(prof, 0)
        tot = X[[pos[i] for i in rc.index]].sum()
        rna[k], ncell[k], ndon[k] = tot / rc["n_cells"].sum(), int(rc["n_cells"].sum()), len(prof)
    B = (B + 1e-6) / (B + 1e-6).sum(1, keepdims=True)
    reference = Reference(B=B, genes=list(table.genes), cell_types=types, rna_content=rna, n_cells=ncell,
                          n_donors=ndon)

    # test specimens
    tc = np.zeros((len(test_spec), len(types), Gn), np.float32)
    cells = np.zeros((len(test_spec), len(types)))
    for i, s in enumerate(test_spec):
        rows = obs[obs["specimen"] == s]
        for k, c in enumerate(types):
            r = rows[rows["cell_type"] == c]
            tc[i, k] = np.asarray(X[[pos[j] for j in r.index]].sum(0)).ravel()
            cells[i, k] = r["n_cells"].sum()
    held = None
    if truth_split:                 # own generator: the donor split above does not change with the flag
        half = np.random.default_rng(seed + 1).binomial(tc.astype(np.int64), 0.5).astype(np.float32)
        tc, held = half, tc - half
    names = ["%s" % s for s in test_spec]
    bulk = pd.DataFrame(tc.sum(1).T, index=pd.Index(table.genes, name="gene"), columns=names)
    meta = spec.loc[test_spec].copy()
    meta.index = names
    meta["specimen"] = test_spec
    for k, c in enumerate(types):
        meta["cells_" + c] = cells[:, k].astype(int)
    T = None
    if contrast is not None:
        T = (meta[contrast[0]] == contrast[1]).astype(float).values
    ref_rows = dict(counts=X[[pos[i] for i in ref.index]], cell_types=ref["cell_type"].values,
                    donors=ref["donor_id"].astype(str).values)
    return PseudobulkBenchmark(reference, bulk, tc, cells, meta, T, ref_rows, held)
