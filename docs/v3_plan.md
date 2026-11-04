# Method v3: candidate changes (draft, 2026-11-04)

Status: candidates only. Nothing here changes the frozen method (v2, tag `method-freeze-v2`). The final plan
is written after the held-out v2 report, matching each candidate to where held-out fell short.

**Protocol.**
- v3 is developed on development splits 0–2 and on 20–29, which become development data for v3 once their
  v2 results are reported.
- New behaviour is opt-in until frozen; v2 stays the package default.
- Before the v3 freeze and its held-out run (new splits 30–39, run once), the development results are shown
  to the PI for a go-ahead. The freeze is a pre-registration decision.
- A candidate that needs data we do not have is reported as blocked; missing data is not simulated in its
  place.

Sources: our own list (2026-11-03) and an external critique (`PALM_biological_critique.md`, 2026-11-04;
points checked against the code, the three that were wrong are listed at the end).

## A. Validation before any whole-blood TB claim

1. **Granulocyte-containing benchmark.** Every current benchmark is cell lines or PBMC; COMBAT's single-cell
   arm is PBMC. Nothing could have shown a neutrophil failure.
   - composition truth: COMBAT whole-blood CyTOF (check availability; not in our download);
   - activity truth: a held-out split of a whole-blood single-cell dataset that captures granulocytes
     (RBC-lysed whole blood on Seq-Well, Parse or BD Rhapsody); to be found.
2. **Separate cell-line and primary-immune results** in every headline table (no averaging across them).
   Kang's recall (0.40–0.48) is the expectation for blood, not the cell-line numbers. *Reporting only; also
   applied to the held-out v2 report.*
3. **"All cell types respond" as a named regime** with its own row (interferon in TB is broad). *Reporting
   only; held-out v2 report where the truth allows it.*
4. **Coverage by route** (unmixed / calibrated; guard fired / silent). Choosing the narrower interval can hide
   a branch with poor coverage. *Held-out v2: program-level route is recorded, the per-type guard is not;
   record both in v3.*

## B. Model and estimator changes

5. **Power when composition barely varies** (Kang recall 0.40–0.48): gene-level unmixing pooled across a
   program's genes.
6. **Blood counts as a second likelihood on π** (CBC differential counts; available in most TB cohorts).
   Anchors the cell mix, identifies RNA content, and turns "more cells" into cells per µL.
   - **RNA content:** currently mean UMIs per cell in the atlas (capture efficiency mixed in, worst for
     neutrophils), a point value with no uncertainty, and fixed across disease status: a state-driven change
     in RNA output can only appear as "more cells". Give it a prior and propagate it.
7. **Real-bulk platform gap** (COMBAT r 0.2–0.6 on real bulk): anchor the average composition (blood counts,
   or a subset with paired bulk + single-cell).
8. **Missing-reference detector:** a non-negative "unassigned" component with a sparsity penalty, reported per
   sample, and/or a held-out-gene check of the fit. If it grows with disease, suppress the neighbouring types'
   effects. Whole-blood reference with neutrophils and low-density granulocytes.
9. **Shared response as the estimand:** report each type's deviation from the shared response
   (β_cp = s_p + d_cp, d shrunk to 0) instead of falling back to a program-level change. The unmixing already
   shrinks towards a shared value (REML); this makes the deviation the reported quantity.
   `plant="shared"` (v2 development) failed and is not the route.
10. **Attenuation as a function of effect size:** calibrate on stimulated donors (Kang, Parse) or on spiked
    effects at TB-like size, instead of one slope from between-donor variation near zero. The design check
    already plants study-sized effects and held-out coverage held on large effects; this tests it directly.
11. **Cell-type-specific program footprints** (G_pcg = G_pg + u_pcg, u shrunk hard). Cheap first step: report,
    per program, how consistent its footprint is across cell types in the atlas, and down-weight
    inconsistent programs.
12. **Whole-blood contaminants:** mask globin (HBB, HBA1/2, HBD) and platelet genes (PPBP, PF4, …), or model
    an erythroid/platelet component; report the masked read share per sample.
13. **Ordered disease axis:** uninfected / latent / incipient / active / on treatment, through the continuous
    path, instead of a binary D. Most "healthy" controls are IGRA-positive.
14. **Single-group path** (unmixing along z: 21% false calls against ≤ 10%): calibrate along z; use the queued
    single-group run.

## C. Smaller checks

15. **Stage A two-step fit for rare types:** compare against full NUTS on a few samples across the abundance
    range; report the interval-width ratio for the rarest types.
16. **Differentiation vs activation states:** write the partition policy (naive / memory / effector) and
    report at two granularities (lineage and fine type; supported via lineage groups).

## Not adopted from the critique (checked against the code)

- "No covariate adjustment": covariates enter the model and the unmixing; paired designs and IPW/AIPW are
  supported. The explainer did not say so (fixed).
- "Program weights are ±1": PROGENy's continuous weights are used, rescaled so the median |weight| is 1
  (explainer fixed).
- "π → δ arrow": no such arrow in the model or the slides.
