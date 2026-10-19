"""Simulators: synthetic reference/bulk data with known ground truth."""
import numpy as np
import jax.numpy as jnp
from .equations import cell_profiles, attribution, shares, effect_decomposition


def make_reference(rng, C=4, P=3, Gn=300, genes_per_prog=30, contrast=1.0):
    """Baseline B (C,Gn) and program dictionary G (P,Gn).
    contrast=1: program genes have cell-type-specific baselines (identifiable attribution);
    contrast=0: identical across cell types (shared program, attribution unidentifiable)."""
    base = rng.gamma(0.7, 1.0, size=Gn)
    B = np.tile(base, (C, 1))
    for c in range(C):
        idx = rng.choice(Gn, size=40, replace=False)
        B[c, idx] *= rng.uniform(4, 10, size=40)
    G = np.zeros((P, Gn))
    prog_genes = rng.choice(Gn, size=P * genes_per_prog, replace=False).reshape(P, -1)
    for p in range(P):
        pg = prog_genes[p]
        G[p, pg] = rng.uniform(0.5, 1.5, size=genes_per_prog)
        specific = base[pg] * np.exp(rng.normal(0, 1.2, size=(C, len(pg))))
        B[:, pg] = (1 - contrast) * base[pg] + contrast * specific
    return B / B.sum(axis=1, keepdims=True), G


def simulate_reference_cells(rng, B, n_per_type=400, n_donors=4, depth=3000, donor_sd=0.15):
    """Single cells drawn from B (with donor effects) -> (counts, cell_types, donors), used to
    exercise the real-data path (palm_bulk.data.build_reference)."""
    C, Gn = B.shape
    per = n_per_type // n_donors
    counts, types, donors = [], [], []
    for d in range(n_donors):
        shift = np.exp(rng.normal(0, donor_sd, size=Gn))
        for c in range(C):
            p = B[c] * shift
            p /= p.sum()
            for _ in range(per):
                counts.append(rng.multinomial(rng.poisson(depth), p))
            types += [c] * per
            donors += [d] * per
    return np.array(counts), np.array(types), np.array(donors)


def simulate(rng, B, G, N=10, tau=0.7, depth=2e5, theta=40.0, active=None, dirichlet=2.0):
    """Non-causal bulk simulator (used by scripts/run_s1.py)."""
    C, Gn = B.shape
    P = G.shape[0]
    if active is None:
        active = np.zeros((C, P), bool)
        for p in range(P):
            active[rng.choice(C, size=2, replace=False), p] = True
    pi = rng.dirichlet(np.full(C, dirichlet), size=N)
    delta = np.zeros((N, C, P))
    on = rng.random((N, P)) < 0.8
    for p in range(P):
        for c in range(C):
            if active[c, p]:
                delta[:, c, p] = on[:, p] * rng.uniform(0.6, 1.6, size=N)
    E = np.array(cell_profiles(jnp.array(B), jnp.array(G), jnp.array(delta)))
    mu = depth * np.einsum("ic,icg->ig", pi, E)
    Y = rng.negative_binomial(theta, theta / (theta + mu))
    Cicp, S = attribution(jnp.array(B), jnp.array(G), jnp.array(pi), jnp.array(delta))
    return dict(Y=Y, L=np.full(N, depth), pi=pi, delta=delta, Cicp=np.array(Cicp), S=np.array(S),
                A=np.array(shares(Cicp)), active=active, theta=theta, tau=tau)


def simulate_causal(rng, B, G, N=40, depth=2e5, theta=40.0, s_eta=0.4, sigma_d=0.15, platform_sd=0.0,
                    theta_sd=0.0, confounding=0.0, batch_size=0.8, balanced=False):
    """Randomised treatment T shifts composition (gamma) and per-cell-type activity (beta).
    platform_sd > 0 multiplies each gene's mean by a hidden lognormal factor (platform bias
    between scRNA reference and bulk) that the model is not told about.
    theta_sd > 0: gene-wise dispersion, log theta_g ~ Normal(log theta, theta_sd).
    confounding > 0: a binary batch x_i with P(x=1) = 0.5 +/- confounding/2 by arm, shifting
    composition and program activity (effect scale batch_size); returned as "X".
    balanced: exactly half the samples treated (otherwise each is a coin flip)."""
    C, Gn = B.shape
    P = G.shape[0]
    T = rng.permutation(np.arange(N) % 2).astype(float) if balanced else (rng.random(N) < 0.5).astype(float)
    T[:2] = [0.0, 1.0]
    a = rng.normal(0, 0.5, C)
    gamma = rng.normal(0, 0.8, C)
    gamma -= gamma.mean()
    beta = np.zeros((C, P))
    kappa = np.zeros((C, P))
    for p in range(P):
        for c in rng.choice(C, size=2, replace=False):
            beta[c, p] = rng.uniform(0.8, 1.6)
        kappa[rng.integers(C), p] = 0.3
    eta = a + gamma * T[:, None] + s_eta * rng.normal(size=(N, C))
    x, Ax, Dx = np.zeros(N), np.zeros(C), np.zeros((C, P))
    if confounding > 0:
        x = (rng.random(N) < 0.5 + confounding * (T - 0.5)).astype(float)
        Ax = rng.normal(0, batch_size, C)
        eta = eta + x[:, None] * Ax
    pi = np.exp(eta - eta.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    delta = kappa + beta * T[:, None, None] + sigma_d * rng.normal(size=(N, C, P))
    if confounding > 0:
        Dx = batch_size * 0.6 * rng.normal(0, 1, (C, P))
        delta = delta + x[:, None, None] * Dx
    E = np.array(cell_profiles(jnp.array(B), jnp.array(G), jnp.array(delta)))
    raw = np.einsum("ic,icg->ig", pi, E)
    if platform_sd > 0:
        raw = raw * np.exp(platform_sd * rng.normal(size=Gn))
        raw /= raw.sum(1, keepdims=True)
    mu = depth * raw
    th = theta * np.exp(theta_sd * rng.normal(size=Gn)) if theta_sd > 0 else theta
    Y = rng.negative_binomial(th, th / (th + mu))
    xb = x.mean()      # effects are defined for a sample at the covariate mean (as the model reports)
    truth = effect_decomposition(jnp.array(B), jnp.array(G), jnp.array(a + xb * Ax), jnp.array(gamma),
                                 jnp.array(kappa + xb * Dx), jnp.array(beta))
    return dict(Y=Y, L=np.full(N, depth), T=T, pi=pi, a=a, gamma=gamma, beta=beta, kappa=kappa,
                truth={k: np.array(v) for k, v in truth.items()}, theta=theta, X=x[:, None], delta=delta,
                a_ref=a + xb * Ax, kappa_ref=kappa + xb * Dx)


DEMO_CELL_TYPES = ["CD4_T", "CD8_T", "B_cell", "Monocyte", "NK"]
DEMO_PROGRAMS = ["IFN_response", "TNFa_NFkB", "Hypoxia", "TGFb"]
DEMO_LINEAGES = ["T_cell", "T_cell", "B_cell", "Myeloid", "NK"]     # level-1 label of each demo cell type


def write_demo(outdir, seed=7, n_genes=2000, n_samples=24, cells_per_type=300, genes_per_prog=60):
    """Write a realistic demo dataset in the formats the notebook expects and return the truth.

    Files in `outdir`: reference.h5ad (raw counts, obs: cell_type, lineage, donor), bulk_counts.csv
    (genes x samples), bulk_metadata.csv (condition, batch, age), programs.csv (program, gene,
    weight). The bulk data contain everything the full model handles: a randomised treatment
    that changes composition and per-cell-type program activity, a sequencing batch that is
    partly confounded with treatment and shifts both, gene-wise dispersion, and platform bias
    between the scRNA reference and bulk. Age is recorded but has no effect."""
    import os
    import pandas as pd
    import anndata as ad
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    C, P = len(DEMO_CELL_TYPES), len(DEMO_PROGRAMS)
    B, G = make_reference(rng, C=C, P=P, Gn=n_genes, genes_per_prog=genes_per_prog)
    genes = ["GENE%04d" % i for i in range(n_genes)]
    counts, ct, dn = simulate_reference_cells(rng, B, n_per_type=cells_per_type, n_donors=4, depth=3000)
    adata = ad.AnnData(X=counts.astype("float32"),
                       obs=pd.DataFrame({"cell_type": np.array(DEMO_CELL_TYPES)[ct],
                                         "lineage": np.array(DEMO_LINEAGES)[ct],
                                         "donor": ["D%d" % d for d in dn]}),
                       var=pd.DataFrame(index=genes))
    adata.obs_names = ["cell%05d" % i for i in range(adata.n_obs)]
    adata.write_h5ad(os.path.join(outdir, "reference.h5ad"))
    s = simulate_causal(rng, B, G, N=n_samples, depth=2e6, platform_sd=0.2, theta_sd=0.4,
                        confounding=0.5, batch_size=0.6, balanced=True)
    samples = ["S%02d" % i for i in range(n_samples)]
    pd.DataFrame(s["Y"].T, index=pd.Index(genes, name="gene"), columns=samples).to_csv(
        os.path.join(outdir, "bulk_counts.csv"))
    pd.DataFrame({"condition": np.where(s["T"] == 1, "treated", "control"),
                  "batch": np.where(s["X"][:, 0] == 1, "run2", "run1"),
                  "age": rng.integers(25, 70, n_samples)}, index=pd.Index(samples, name="sample")).to_csv(
        os.path.join(outdir, "bulk_metadata.csv"))
    long = [(DEMO_PROGRAMS[p], genes[g], G[p, g]) for p in range(P) for g in np.nonzero(G[p])[0]]
    pd.DataFrame(long, columns=["program", "gene", "weight"]).to_csv(os.path.join(outdir, "programs.csv"), index=False)
    return dict(B=B, G=G, genes=genes, cell_types=list(DEMO_CELL_TYPES), programs=list(DEMO_PROGRAMS),
                samples=samples, **{k: s[k] for k in ["pi", "delta", "T", "beta", "kappa", "gamma",
                                                      "a_ref", "kappa_ref", "X"]})


def simulate_study(rng, C=5, P=4, Gn=1500, genes_per_prog=60, N=24, cells_per_type=200, n_donors=4,
                   depth=2e6, platform_sd=0.2, theta_sd=0.3, **causal_kw):
    """A full synthetic study in real-data formats, for method comparisons: single-cell reference
    counts (cells x genes, with cell type and donor labels), bulk counts (genes x samples DataFrame)
    from a balanced randomised treatment, a program table, and the simulator output `sim` (truth).
    Bulk carries hidden platform bias and gene-wise dispersion by default. Use `truth_on_genes`
    for truth tables aligned to prepared inputs."""
    import pandas as pd
    B, G = make_reference(rng, C=C, P=P, Gn=Gn, genes_per_prog=genes_per_prog)
    counts, ct, dn = simulate_reference_cells(rng, B, n_per_type=cells_per_type, n_donors=n_donors, depth=3000)
    s = simulate_causal(rng, B, G, N=N, depth=depth, platform_sd=platform_sd, theta_sd=theta_sd, balanced=True,
                        **causal_kw)
    genes = ["g%04d" % g for g in range(Gn)]
    types = ["type%02d" % c for c in range(C)]
    programs = ["prog%02d" % p for p in range(P)]
    samples = ["s%03d" % i for i in range(N)]
    return dict(B=B, G=G, genes=genes, cell_types=types, programs=programs, samples=samples,
                sc_counts=counts, sc_cell_types=np.array(types)[ct], sc_donors=np.array(["d%d" % d for d in dn]),
                bulk=pd.DataFrame(s["Y"].T, index=genes, columns=samples),
                program_table=pd.DataFrame(G, index=programs, columns=genes),
                treatment=s["T"], sim=s)


def truth_on_genes(study, genes, cell_types, programs):
    """Truth tables for `simulate_study`, aligned to prepared inputs (their genes, cell types and
    programs, in that order): read fractions on those genes (noise-free, no platform bias), delta
    (N, C, P) and beta (C, P)."""
    s = study["sim"]
    E = np.asarray(cell_profiles(study["B"], study["G"], s["delta"]))                      # (N, C, G)
    gi = [study["genes"].index(g) for g in genes]
    ci = [study["cell_types"].index(c) for c in cell_types]
    pj = [study["programs"].index(p) for p in programs]
    w = s["pi"][:, :, None] * E[:, :, gi]
    pi_sub = w.sum(-1) / w.sum((-1, -2))[:, None]
    return dict(pi=pi_sub[:, ci], delta=s["delta"][:, ci][:, :, pj], beta=s["beta"][np.ix_(ci, pj)])
