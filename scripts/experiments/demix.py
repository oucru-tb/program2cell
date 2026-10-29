"""De-mixed reallocation test (post hoc on saved dev_bench fits).

Reallocation r(Y, pi) = estimate_delta(Y * pi B / sum_c pi B, B, G). Under the model, expected counts are
mu_i(delta) = n_i sum_c pi_ic softmax(log B_c + delta_ic G), so E r ~ f_i(delta_i), with
f_i(delta) ~ f_i(0) + A_i delta (A_i: (CP x CP) leakage/attenuation matrix, by central differences).
De-mixed estimate: delta~ = (A'A + lam I)^-1 A' (r - f(0)); lam -> inf with rescale is not realloc, so
realloc itself is reported as a separate row. Optional Gauss-Newton refinement (nonlinear f).
"""
import json, os, sys, types
import numpy as np, pandas as pd
from scipy.special import softmax

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db
import palm_bulk as pb
from palm_bulk.core import PseudobulkTable, make_benchmark
from palm_bulk.core.scores import estimate_delta
from palm_bulk.compare.activity import activity_from_expression
from palm_bulk.compare.base import MethodResult
from palm_bulk import compare

ds_name, config = sys.argv[1], sys.argv[2]
splits = range(10)
LAMS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
ds = db.DATASETS[ds_name]
table = PseudobulkTable.from_h5ad(ds["path"], {}, db.TYPES)
prog = pb.load_programs(os.path.join(ROOT, "data", "progeny_human_top500.csv")).loc[db.PROGS]
rows = []


def expected_realloc(pi, n, Bm, Gm, delta, w):
    """r(mu(delta)) for all samples; delta (N, C, P)."""
    prof = softmax(np.log(Bm)[None] + np.einsum("ncp,pg->ncg", delta, Gm), -1)
    mu = n[:, None] * np.einsum("nc,ncg->ng", pi, prof)
    return estimate_delta(mu[:, None, :] * w, Bm, Gm)


def jacobian(pi, n, Bm, Gm, at, w, eps=1e-3):
    N, C, P = at.shape
    A = np.zeros((N, C * P, C * P))
    for k in range(C * P):
        c, p = divmod(k, P)
        d = np.zeros_like(at); d[:, c, p] = eps
        A[:, :, k] = ((expected_realloc(pi, n, Bm, Gm, at + d, w) - expected_realloc(pi, n, Bm, Gm, at - d, w))
                      / (2 * eps)).reshape(N, -1)
    return A


def demix(r, pi, n, Bm, Gm, w, lam, gn_steps=0):
    N, C, P = r.shape
    z = np.zeros_like(r)
    f0 = expected_realloc(pi, n, Bm, Gm, z, w)
    A = jacobian(pi, n, Bm, Gm, z, w)
    I = np.eye(C * P)
    rhs = np.einsum("nji,nj->ni", A, (r - f0).reshape(N, -1))
    d = np.linalg.solve(np.einsum("nji,njk->nik", A, A) + lam * I, rhs[..., None])[..., 0].reshape(N, C, P)
    for _ in range(gn_steps):              # Gauss-Newton on ||r - f(d)||^2 + lam ||d||^2
        fd = expected_realloc(pi, n, Bm, Gm, d, w)
        J = jacobian(pi, n, Bm, Gm, d, w)
        g = np.einsum("nji,nj->ni", J, (r - fd).reshape(N, -1)) - lam * d.reshape(N, -1)
        step = np.linalg.solve(np.einsum("nji,njk->nik", J, J) + lam * I, g[..., None])[..., 0]
        d = d + step.reshape(N, C, P)
    return d, A


for split in splits:
    f = os.path.join(ROOT, "benchmarks", "dev", ds_name, "s%d__%s.npz" % (split, db._slug(config)))
    if not os.path.exists(f):
        continue
    z = np.load(f)
    b = make_benchmark(table, min_cells=20, seed=split, truth_split=True, contrast=ds["contrast"], n_test=ds["n_test"])
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    truth = b.truth(inp)
    gi = [b.reference.genes.index(g) for g in inp.genes]
    ci = [b.cell_types.index(c) for c in inp.cell_types]
    ulm, _ = activity_from_expression(b.truth_counts[:, ci][:, :, gi], inp, "ulm", prog)
    T = np.asarray(b.treatment)
    Y, Bm, Gm = inp.Y, inp.B, inp.Gmat
    n = Y.sum(1)

    def design(d):
        return d[T == 1].mean(0) - d[T == 0].mean(0)

    def score(name, act, pi):
        s = compare.score(MethodResult(name, fractions=pi, activity=act, effects=design(act)), truth)
        su = compare.score(MethodResult(name, fractions=pi, activity=act), dict(pi=truth["pi"], delta=ulm))
        out = dict(split=split, method=name)
        for k in db.ACT + ["act_mae", "eff_pearson"]:
            out[k] = s.get(k)
        for k in db.ACT:
            out["ulm_" + k] = su.get(k)
        rows.append(out)

    for src, pi in [("post", np.median(z["pi"], 0)), ("oracle", truth["pi"])]:
        w = pi[:, :, None] * Bm[None]
        w = w / w.sum(1, keepdims=True)
        r = estimate_delta(Y[:, None, :] * w, Bm, Gm)
        score("%s realloc" % src, r, pi)
        for lam in LAMS:
            d, A = demix(r, pi, n, Bm, Gm, w, lam)
            score("%s demix lam=%g" % (src, lam), d, pi)
        for lam in [0.01, 0.1]:
            d, _ = demix(r, pi, n, Bm, Gm, w, lam, gn_steps=2)
            score("%s demix+GN lam=%g" % (src, lam), d, pi)
        if src == "post" and split == 0:
            C, P = r.shape[1:]
            Am = A.mean(0)
            diag = np.array([Am[c * P + p, c * P + p] for c in range(C) for p in range(P)]).reshape(C, P)
            off = np.abs(Am).sum(1).reshape(C, P) - np.abs(diag)
            print("split 0 leakage: mean diag (own-type gain) per type", diag.mean(1).round(2),
                  "| sum |off-diag| per type", off.mean(1).round(2), flush=True)
    print("split %d done" % split, flush=True)

df = pd.DataFrame(rows)
out = os.path.join(ROOT, "benchmarks", "dev", "demix_%s_%s.json" % (ds_name, db._slug(config)))
df.to_json(out, orient="records", indent=1)
cols = ["act_spearman_across_samples", "act_spearman_shared", "act_spearman_specific", "act_spearman_across_types",
        "act_mae", "eff_pearson", "ulm_act_spearman_across_samples", "ulm_act_spearman_specific"]
t = df.groupby("method", sort=False)[cols].mean()
t.columns = ["rho", "shared", "spec", "types", "mae", "eff_r", "ulm_rho", "ulm_spec"]
ref = df[df.method == "post realloc"].set_index("split")["act_spearman_across_samples"]
t["d_rho_vs_realloc"] = df.assign(d=df.act_spearman_across_samples - df.split.map(ref)).groupby("method", sort=False)["d"].mean()
pd.set_option("display.width", 200)
print("== %s %s, %d splits" % (ds_name, config, df.split.nunique()))
print(t.round(3).to_string())
