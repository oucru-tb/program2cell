"""CyTOF-anchored fractions on COMBAT real bulk (plan item 3, 2026-11-05; docs/v4_plan.md sections 7, 10).

Real whole-blood bulk: PALM-lite drives B and CD8 T to ~0 (log bias -7). Here the composition is anchored on the
same specimens' whole-blood CyTOF: coarse groups B / myeloid / T / NK from CyTOF (renormalised over the PBMC
types: granulocytes, pDC, plasmablasts, unclassified dropped), split within group (CD14 / CD16 Mono / DC;
CD4 / CD8 T) by the PALM-lite proportions, converted from cell to read fractions with each type's reads per cell
in the single-cell table. Specimens without CyTOF keep PALM-lite.
Per split (sepsis vs healthy) and fractions (lite / cytof): fraction r vs the single-cell truth; effects by v2
(auto) and v4 stacked (v4k: frac_sd 1.0 for lite on real bulk, "estimate" for CyTOF fractions); scores as
fraction_error.score plus calls.
Usage: python combat_cytof.py SPLITS -> benchmarks/dev/combat_cytof/sepsis__s{split}.json
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.realloc import em_fractions
from fraction_error import score
from missing_type_score import CYTOF

OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_cytof")
GROUP = {"B": "B.cell", "CD14 Mono": "Myeloid", "CD16 Mono": "Myeloid", "DC": "Myeloid", "CD4 T": "T.cell",
         "CD8 T": "T.cell", "NK": "NK"}


def cytof_table():
    x = pd.ExcelFile(CYTOF)
    d = pd.concat([x.parse(s) for s in x.sheet_names if s.endswith("CellFreqs")])
    key = d["CyTOF_sample_ID"].str.extract(r"^([A-Z]\d+)-([A-Za-z]+\d+)")
    d.index = key[0] + "|" + key[1]
    return d[~d.index.duplicated()]


def anchored(pi_lite, specs, types, cyt, reads_per_cell):
    pi = pi_lite.copy()
    n = 0
    for i, s in enumerate(specs):
        if s not in cyt.index:
            continue
        n += 1
        cells = np.zeros(len(types))
        for g in set(GROUP.values()):
            m = np.array([GROUP[t] == g for t in types])
            within = pi_lite[i, m] / max(pi_lite[i, m].sum(), 1e-9) if pi_lite[i, m].sum() > 0 else np.full(m.sum(), 1 / m.sum())
            cells[m] = float(cyt.loc[s, g]) * within
        reads = cells * reads_per_cell
        pi[i] = reads / reads.sum()
    return np.clip(pi, 1e-4, None) / np.clip(pi, 1e-4, None).sum(1, keepdims=True), n


def run(splits):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_sepsis.h5ad"), {}, cr.TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    cyt = cytof_table()
    o = table.obs
    rpc = {t: float(o.loc[o["cell_type"] == t, "n_counts"].sum() / max(o.loc[o["cell_type"] == t, "n_cells"].sum(), 1))
           for t in cr.TYPES}
    for split in splits:
        f = os.path.join(OUT, "sepsis__s%d.json" % split)
        if os.path.exists(f):
            continue
        b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=20)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        real = bulk_all.loc[genes, list(b.bulk.columns)]
        test_donors = set(b.meta["donor_id"].astype(str))
        inp = pb.prepare_inputs(real, b.reference, prog, treatment=b.treatment, max_genes=1000)
        T = np.asarray(inp.T)
        types = list(inp.cell_types)
        tr = b.truth(inp)
        beta = np.asarray(pb.core.contrast(T)(tr["delta"]))
        pt = np.asarray(tr["pi"], float)
        cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split)
        lite = em_fractions(np.asarray(inp.Y, float), np.asarray(inp.B, float))
        cy, n_cy = anchored(lite, list(real.columns), types, cyt, np.array([rpc[t] for t in types]))
        out = dict(split=split, types=types, n=int(len(T)), n_cytof=n_cy, beta=beta.tolist())
        for name, pi, fsd in (("lite", lite, 1.0), ("cytof", cy, "estimate")):
            fr = [np.corrcoef(pi[:, c], pt[:, c])[0, 1] for c in range(len(types))]
            out[name] = dict(frac_r=fr, frac_mean_r=float(np.nanmean(fr)),
                             frac_bias=(np.log(pi) - np.log(np.clip(pt, 1e-4, None))).mean(0).tolist())
            for est, kw in (("v2", dict(estimator="auto")),
                            ("v4k", dict(estimator="v4", v4_kw=dict(stack=True, noise_anchor=True, frac_sd=fsd)))):
                t0 = time.time()
                r = pb.program_deconvolution(inp, cal, fractions=pi, treatment=T, seed=split, **kw)
                lo, hi = (np.asarray(x) for x in r.effects_interval)
                s = score(beta, np.asarray(r.effects), lo, hi)
                s.update(calls=int(np.asarray(r.calls).sum()), seconds=time.time() - t0,
                         effects=np.asarray(r.effects).tolist(), lo=lo.tolist(), hi=hi.tolist(),
                         frac_sd=r.info.get("v4", {}).get("frac_sd"))
                out[name][est] = s
                print("[%s] s%d %-5s frac r %.2f  %-3s cov %.2f r %.2f calls %d null %.2f power %.2f width %.2f%s" % (
                    time.strftime("%H:%M"), split, name, out[name]["frac_mean_r"], est, s["cov"], s["r"], s["calls"],
                    s["null_calls"], s["power"], s["width"],
                    "  frac_sd %.2f" % s["frac_sd"] if s["frac_sd"] is not None else ""), flush=True)
        json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(cl.db._splits(sys.argv[1]))
