"""High-level API: `run_palm(inputs)` -> PalmResult with tidy tables, plots and export."""
import os
import time
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from ..core.equations import attribution, shares
from .diagnostics import interval, convergence_problems
from .fit import fit, model_args
from .model import make_model


class PalmResult:
    def __init__(self, inputs, draws, level=0.9, max_draws=200):
        self.inputs, self.draws, self.level = inputs, draws, level
        self.has_treatment = inputs.T is not None and len(np.unique(inputs.T)) == 2
        self.has_programs = "delta" in draws
        self.grouped = getattr(inputs, "groups", None) is not None
        # level at which program activity is estimated: cell types, or lineages when grouped
        self.activity_levels = inputs.group_names if self.grouped else inputs.cell_types
        n = draws["pi"].shape[0]
        self._idx = np.linspace(0, n - 1, min(n, max_draws)).astype(int)
        self._attr = None

    # ---- compositions ------------------------------------------------------
    def fractions(self):
        """Posterior median and interval of read fractions pi (samples x cell types)."""
        return self._wide(self.draws["pi"])

    def cell_fractions(self):
        """Cell fractions: read fractions divided by reference RNA content, renormalised per draw."""
        r = self.inputs.rna_content
        if r is None:
            raise ValueError("rna_content missing from inputs")
        w = self.draws["pi"] / r
        return self._wide(w / w.sum(-1, keepdims=True))

    def lineage_fractions(self, cells=False):
        """Fractions summed within lineages (reference built with lineage_key), with intervals.
        cells=True sums cell fractions instead of read fractions. Useful to compare a level-2 fit with
        a level-1 fit, and because sums over similar subtypes are better determined than each subtype."""
        lin = self.inputs.lineages
        if lin is None:
            raise ValueError("no lineage labels: build the reference with from_anndata(..., lineage_key=...)")
        w = self.draws["pi"]
        if cells:
            w = w / self.inputs.rna_content
            w = w / w.sum(-1, keepdims=True)
        names = sorted(set(lin))
        M = np.array([[l == n for n in names] for l in lin], float)          # (C, lineages)
        return self._wide(w @ M, names)

    def _wide(self, x, columns=None):
        lo, hi = interval(x, self.level)
        cols = self.inputs.cell_types if columns is None else columns
        mk = lambda a: pd.DataFrame(a, index=self.inputs.samples, columns=cols)
        return dict(median=mk(np.median(x, 0)), lo=mk(lo), hi=mk(hi))

    def _need_programs(self):
        if not self.has_programs:
            raise ValueError("this fit has no programs (composition only)")

    def _delta_by_cell_type(self, idx):
        d = self.draws["delta"][idx]
        return d[:, :, self.inputs.groups, :] if self.grouped else d

    # ---- per-sample attribution -----------------------------------------------
    def _attribution_draws(self, max_elements=2.5e8):
        # Draws are processed in batches so the (draws, N, C, G) profile tensors stay ~1 GB
        # (all 200 draws at once would need tens of GB for hundreds of samples).
        # float32 is enough here: this is a deterministic function of each draw, not the NUTS
        # energy, so rounding stays ~1e-7 relative instead of accumulating over a large likelihood.
        self._need_programs()
        if self._attr is None:
            B, G = jnp.array(self.inputs.B, jnp.float32), jnp.array(self.inputs.Gmat, jnp.float32)
            pi, de = self.draws["pi"][self._idx], self._delta_by_cell_type(self._idx)
            batch = max(1, int(max_elements // np.prod(pi.shape[1:] + B.shape[1:])))
            f = jax.jit(jax.vmap(lambda p, d: attribution(B, G, p, d)))
            parts = [f(jnp.array(pi[k:k + batch]), jnp.array(de[k:k + batch]))
                     for k in range(0, len(pi), batch)]
            Cicp = np.concatenate([np.array(c) for c, _ in parts])
            S = np.concatenate([np.array(s) for _, s in parts])
            self._attr = (Cicp, S, np.array(shares(Cicp, 1e-9)))
        return self._attr

    def cell_type_activity(self):
        """Program activity inside each cell type, per sample: posterior median and interval of
        delta (log-scale shift of the program's genes relative to the reference state). With
        program_groups='lineage' the `cell_type` column holds lineages (the level estimated)."""
        self._need_programs()
        d = self.draws["delta"]
        lo, hi = interval(d, self.level)
        med = np.median(d, 0)
        return pd.DataFrame([dict(sample=s, cell_type=cn, program=pn, activity=med[i, c, p],
                                  lo=lo[i, c, p], hi=hi[i, c, p])
                             for i, s in enumerate(self.inputs.samples)
                             for c, cn in enumerate(self.activity_levels)
                             for p, pn in enumerate(self.inputs.programs)])

    def program_activity(self):
        """Per-sample induced program signal S_ip (sum over cell types of contributions)."""
        _, S, _ = self._attribution_draws()
        lo, hi = interval(S, self.level)
        rows = []
        for i, s in enumerate(self.inputs.samples):
            for p, name in enumerate(self.inputs.programs):
                rows.append(dict(sample=s, program=name, activity=np.median(S[:, i, p]),
                                 lo=lo[i, p], hi=hi[i, p]))
        return pd.DataFrame(rows)

    def attribution(self):
        """Long table: share of each program's induced signal carried by each cell type, per sample."""
        Cicp, S, A = self._attribution_draws()
        lo, hi = interval(A, self.level)
        Am, Cm = np.nanmedian(A, 0), np.median(Cicp, 0)
        rows = []
        for i, s in enumerate(self.inputs.samples):
            for p, pn in enumerate(self.inputs.programs):
                for c, cn in enumerate(self.inputs.cell_types):
                    rows.append(dict(sample=s, program=pn, cell_type=cn, contribution=Cm[i, c, p],
                                     share=Am[i, c, p], share_lo=lo[i, c, p], share_hi=hi[i, c, p]))
        return pd.DataFrame(rows)

    # ---- treatment effects ---------------------------------------------------------
    def effects(self):
        """Intrinsic (within-cell-type) vs composition-driven effect of treatment on each program,
        per cell type, with credible intervals. Requires a 0/1 treatment with both arms present."""
        self._need_programs()
        if not self.has_treatment:
            raise ValueError("no two-arm treatment was supplied")
        d = self.draws["dec"]
        out = []
        for key, name in [("I", "intrinsic"), ("K", "composition")]:
            lo, hi = interval(d[key], self.level)
            med = np.median(d[key], 0)
            for p, pn in enumerate(self.inputs.programs):
                for c, cn in enumerate(self.inputs.cell_types):
                    out.append(dict(program=pn, cell_type=cn, component=name,
                                    effect=med[c, p], lo=lo[c, p], hi=hi[c, p]))
        return pd.DataFrame(out)

    def effect_totals(self):
        """Per program: total bulk effect and its intrinsic / composition parts (summed over cell types)."""
        self._need_programs()
        if not self.has_treatment:
            raise ValueError("no two-arm treatment was supplied")
        d = self.draws["dec"]
        parts = dict(total=d["total"], intrinsic=d["I"].sum(1), composition=d["K"].sum(1))
        rows = []
        for p, pn in enumerate(self.inputs.programs):
            row = dict(program=pn)
            for k, v in parts.items():
                lo, hi = interval(v[:, p], self.level)
                row.update({k: np.median(v[:, p]), k + "_lo": lo, k + "_hi": hi})
            rows.append(row)
        return pd.DataFrame(rows)

    def intrinsic_shares(self):
        """Share of each program's intrinsic treatment effect carried by each cell type."""
        self._need_programs()
        if not self.has_treatment:
            raise ValueError("no two-arm treatment was supplied")
        I = self.draws["dec"]["I"]
        A = I / I.sum(1, keepdims=True)
        lo, hi = interval(A, self.level)
        med = np.nanmedian(A, 0)
        return pd.DataFrame([dict(program=pn, cell_type=cn, share=med[c, p], lo=lo[c, p], hi=hi[c, p])
                             for p, pn in enumerate(self.inputs.programs)
                             for c, cn in enumerate(self.inputs.cell_types)])

    # ---- diagnostics / export ----------------------------------------------------------
    def diagnostics(self):
        """Sampler summary: chains, divergences, worst split R-hat and effective sample size, and
        `problems`, the list of failed checks (empty when the fit looks converged)."""
        dr = self.draws
        chains = dr.get("chains", 1)
        return dict(chains=chains, n_draws=int(dr["pi"].shape[0]), divergences=dr["divergences"],
                    rhat_max=dr.get("rhat_max"), ess_min=dr.get("ess_min"), mean_steps=dr.get("mean_steps"),
                    platform=dr.get("platform"), chain_method=dr.get("chain_method"), precision=dr.get("precision"),
                    problems=convergence_problems(dr.get("convergence", {}), chains, dr["divergences"]))

    def convergence(self):
        """Split R-hat and effective sample size per quantity (one row each)."""
        return pd.DataFrame(self.draws.get("convergence", {})).T

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        f = self.fractions()
        f["median"].to_csv(os.path.join(outdir, "read_fractions.csv"))
        f["lo"].to_csv(os.path.join(outdir, "read_fractions_lo.csv"))
        f["hi"].to_csv(os.path.join(outdir, "read_fractions_hi.csv"))
        if self.inputs.lineages is not None:
            self.lineage_fractions()["median"].to_csv(os.path.join(outdir, "lineage_read_fractions.csv"))
        if not self.has_programs:
            return outdir
        self.program_activity().to_csv(os.path.join(outdir, "program_activity.csv"), index=False)
        self.cell_type_activity().to_csv(os.path.join(outdir, "cell_type_activity.csv"), index=False)
        self.attribution().to_csv(os.path.join(outdir, "attribution.csv"), index=False)
        if self.has_treatment:
            self.effects().to_csv(os.path.join(outdir, "treatment_effects.csv"), index=False)
            self.effect_totals().to_csv(os.path.join(outdir, "treatment_effect_totals.csv"), index=False)
            self.intrinsic_shares().to_csv(os.path.join(outdir, "intrinsic_shares.csv"), index=False)
        return outdir


def estimate_runtime(inputs, use_m=True, dispersion="gene", num_warmup=200, num_samples=400,
                     chains=2, n_eval=10):
    """Predict fit time before fitting: times one gradient of the model on these inputs and
    multiplies by the gradients NUTS needs. NUTS uses 2^k - 1 gradients per iteration; small
    problems typically settle near 255-511, large ones near the 1023 cap, so a range is given.
    The total assumes chains run one after another; divide by the number of chains running at
    once (configure_compute(parallel_chains=...) or a GPU)."""
    from numpyro.infer.util import initialize_model
    args, n_prog, _ = model_args(inputs.Y, inputs.L, inputs.T, inputs.B, inputs.Gmat, inputs.X)
    model = make_model(use_m, dispersion=dispersion, n_prog=n_prog, groups=getattr(inputs, "groups", None))
    info = initialize_model(jax.random.PRNGKey(0), model, model_args=args)
    f = jax.jit(jax.value_and_grad(info.potential_fn))
    z = info.param_info.z
    jax.block_until_ready(f(z))
    t = time.time()
    for _ in range(n_eval):
        out = f(z)
    jax.block_until_ready(out)
    sec = (time.time() - t) / n_eval
    iters = (num_warmup + num_samples) * chains
    est = {steps: sec * steps * iters / 60 for steps in (255, 511, 1023)}
    return dict(ms_per_gradient=sec * 1e3, minutes_if_255_steps=est[255], minutes_if_511_steps=est[511],
                minutes_if_1023_steps=est[1023], n_parameters=int(sum(np.size(v) for v in jax.tree_util.tree_leaves(z))),
                genes_with_program_weight=n_prog, platform=jax.default_backend())


FLOAT32_WARN_SIZE = 20000     # samples x genes above which float32 rounding disturbs NUTS


def run_palm(inputs, use_m=True, theta=None, num_warmup=200, num_samples=400, chains=2, seed=0,
             level=0.9, progress_bar=False, chain_method="auto", dispersion="gene", reparam=False,
             init="svi", svi_steps=3000, check=True):
    """Fit the Bayesian model on prepared inputs (see palm_bulk.data.prepare_inputs).
    use_m=True fits a gene-wise platform scale (recommended for real bulk vs scRNA references).
    theta=None samples the NB dispersion; pass a float to fix it. dispersion='gene' (default) gives
    each gene its own dispersion. Covariates and lineage grouping set in prepare_inputs are used
    automatically. init='svi' (default) starts NUTS from a short VI fit, so warmup can be short.
    chain_method='auto' runs chains in parallel when `configure_compute` made enough devices
    available (or batched on a GPU), else one after another. reparam: see bayes.fit.fit.
    chains=2 by default so R-hat can detect chains that disagree; check=True warns when R-hat,
    effective sample size or divergences fail (see res.diagnostics() and res.convergence()).
    Defaults follow benchmarks/calibration."""
    if not jax.config.jax_enable_x64 and inputs.Y.size >= FLOAT32_WARN_SIZE:
        warnings.warn("float32 with %d samples x %d genes: rounding error in the likelihood can shrink the "
                      "NUTS step size and degrade results. Call pb.configure_compute(precision='float64') at the "
                      "start of the session (see benchmarks/precision.json)." % inputs.Y.shape)
    draws = fit(inputs.Y, inputs.L, inputs.T, inputs.B, inputs.Gmat, theta=theta, use_m=use_m,
                num_warmup=num_warmup, num_samples=num_samples, chains=chains, seed=seed,
                progress_bar=progress_bar, chain_method=chain_method, dispersion=dispersion,
                reparam=reparam, init=init, svi_steps=svi_steps, X=inputs.X,
                groups=getattr(inputs, "groups", None))
    res = PalmResult(inputs, draws, level=level)
    if check:
        problems = res.diagnostics()["problems"]
        if problems:
            warnings.warn("convergence checks failed; consider more warmup/samples or chains:\n  "
                          + "\n  ".join(problems))
    return res
