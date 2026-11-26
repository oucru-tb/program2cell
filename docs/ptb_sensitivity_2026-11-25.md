# PTB MDR vs DS: should the calling standard be lowered? (2026-11-25)

Trigger: freeze v5 on PTB MDR vs DS (148 vs 148, Run1, batch covariate, 5 innate Reactome programs, 8 COMBAT types,
anchored offsets) made 0 of 40 calls. The user asked to lower the threshold or the standard of the test.

Status: **post-hoc sensitivity analysis.** The frozen method (level 0.90, interval calls, no multiplicity
correction) is unchanged. Nothing here is a new freeze or a pre-registered test, and it must be reported as such.
Scripts: `scripts/experiments/ptb_sensitivity.py` (bulk / width / rescale / summary),
`scripts/experiments/ptb_sensitivity_chain.sh` (overnight lanes p, x, y, z), `run_ptb.py LEVEL=`,
`causal_loop.py` estimators `v5_l80`, `v5_l67`.

## 0. Positive results (summary)

1. **PALM recovers real shifts in the major whole-blood types, without bias, on real PTB bulk.** A +0.3 IFNg
   shift planted into the MDR samples is estimated at +0.324 [+0.197, +0.451] in neutrophils and +0.358
   [+0.228, +0.489] in CD14 monocytes (base +0.100), and called in both (section 8b). With the whole TB-vs-healthy
   matrix planted, the neutrophil recovery slope is 1.01 and both neutrophil calls have the right sign (8a).
2. **No wrong-sign calls and no calls on near-zero planted pairs** across the matrix spike-ins (0 of 52 near-zero
   pairs over scales 0.25 / 0.5 / 1.0), and no off-target calls in 7 of 8 single-pair spike-ins.
3. **The frozen standard is robust.** Lowering the level hardly changes the answer: 0 calls at 0.80 on PTB, and the
   held-out power gain is small (+0.02), so the 0.90 result is not a borderline artefact of the threshold (6a, 6b).
4. **PALM stays far safer than current practice at every level tested**: null-call rate 0.022 (0.90), 0.035 (0.80),
   0.059 (0.67) on held-out 30-39 vs ~0.53 for practice on the same splits (6a).
5. **PALM and the bulk data agree.** No program differs between MDR and DS in bulk after BH (section 3), and
   PALM's IFN-I point estimates are consistently negative across all eight types (−0.03 to −0.14), matching the
   nominal bulk IFN-I decrease (−5%, p 0.08), so PALM adds no unsupported claims.
6. **Biological reading:** in neutrophils and CD14 monocytes, where the study can resolve +0.3 shifts, MDR and DS
   whole blood show no innate-program difference of that size. Within active TB, drug-resistance status does
   not leave a large innate signature in blood.

Limits to state alongside: minor types (B, T, NK, DC, CD16 Mono) are not resolvable in this study at any level
(section 8b), and a strong neutrophil shift leaked into 3 minor-type IFNg calls in one spike-in.

## 1. The current standard is already lenient

- Calls are 90% intervals that exclude 0, per pair, with **no multiplicity correction** over the 40 pairs.
  (A BH mode exists, `calls="bh"`; it is stricter, not looser.)
- So "lowering the standard" can only mean a lower interval level (80%, 67%) or a coarser question.

## 2. Why nothing is called: the width is reference leakage, not sample size

`ptb_sensitivity.py width`:
- Study noise alone (sample bootstrap + residual): sd 0.021, i.e. 90% half-width **0.035**.
- Reported 90% half-widths: median **0.110** (0.039 Neutro TNF to 0.52 CD16 Mono IFNI). About 69% of the median
  half-width is the calibrated leakage term (how much of another type's shift the reference lets into this type).
- Largest |effect| / half-width over the 40 pairs: **0.89** (CD4 T IFNI −0.090 [−0.191, +0.011]).
- With 148 vs 148 the study noise is small. More samples or a lower level does not remove the leakage term,
  which comes from how well the reference separates the types.

## 3. Is there a signal to split? Bulk program test, no cell types

`ptb_sensitivity.py bulk` (program score = weighted mean log2 CPM of the program genes; OLS on group + batch, HC3):

| program | diff (log2) | 90% CI | p | q (BH) |
|---|---|---|---|---|
| IFNI | −0.067 | [−0.131, −0.004] | 0.081 | 0.24 |
| IFNg | −0.027 | [−0.073, +0.019] | 0.34 | 0.56 |
| NLR | −0.005 | [−0.018, +0.009] | 0.58 | 0.73 |
| TLR | +0.001 | [−0.011, +0.014] | 0.85 | 0.85 |
| TNF | −0.027 | [−0.053, −0.000] | 0.096 | 0.24 |

Even with no deconvolution, no program differs between MDR and DS after BH. The best is IFN-I, about 5% lower in MDR
(0.2 sd of the score, nominal p 0.08). The cell-type step cannot create signal that the bulk does not carry. The
PALM point estimates agree: IFN-I is negative in all 8 types (−0.03 to −0.14), consistent with one weak bulk
decrease spread across types.

## 4. Was the design able to see a real difference?

The PTB vs healthy effects (same programs, reference and recipe; results/ptb/combat_v5_immune_reactome_innate5_anchored)
are a yardstick for a disease-sized difference. **20 of 40** of those effects are larger than the MDR vs DS 90%
half-widths (point-estimate comparison, about 50% power at that size; an approximation, not a spike-in).
**Corrected by the spike-in (section 8):** this yardstick was too optimistic. It ignores that planted effects
widen the size-matched interval and are attenuated outside neutrophils. The null is informative for neutrophils
(the majority type), not for the minor types.

## 5. Cost of a lower level on known-answer data (held-out 30-39, freeze v5)

Preview (`rescale`: the 0.90 intervals shrunk by the normal quantile ratio; 13 datasets, dataset means; practice
null-call rate on the same splits about 0.53):

| level | null calls (mean) | worst dataset | power | power cell lines | power PBMC | coverage |
|---|---|---|---|---|---|---|
| 0.90 (frozen) | 0.022 | 0.074 mcf_tram | 0.299 | 0.493 | 0.133 | 0.967 |
| 0.80 | 0.042 | 0.132 mcf_tram | 0.362 | 0.529 | 0.219 | 0.939 |
| 0.67 | 0.090 | 0.231 mcf_tram | 0.456 | 0.550 | 0.376 | 0.889 |
| 0.50 | 0.218 | 0.367 mcf_tram | 0.533 | 0.601 | 0.474 | 0.782 |

At 0.80 the worst dataset (mcf_tram) breaks the freeze safety rule S1 (null calls <= 0.10). Exact reruns
(`v5_l80`, `v5_l67`, all 30-39) are in section 6.

## 6. Exact reruns (overnight 2026-11-25/26)

### 6a. Known-answer data: freeze v5 rerun at level 0.80 / 0.67 on held-out 30-39 (130 runs each, no errors)

| level | null calls (mean) | worst dataset | power | power cell lines | power PBMC | coverage |
|---|---|---|---|---|---|---|
| 0.90 (frozen) | 0.022 | 0.074 mcf_tram | 0.299 | 0.493 | 0.133 | 0.967 |
| 0.80 | 0.035 | **0.112** mcf_tram | 0.319 | 0.481 | 0.180 | 0.941 |
| 0.67 | 0.059 | **0.171** mcf_tram | 0.354 | 0.461 | 0.261 | 0.898 |

- The exact power gain is smaller than the preview (+0.02 at 0.80, +0.055 at 0.67). The level also changes the
  routing (width_guard compares interval widths), and in cell lines power even falls slightly.
- At 0.80, mcf_tram breaks S1 (null calls <= 0.10); at 0.67, mcf_ida, mcf_tram and zhang break it, and cell-line
  coverage drops to 0.74-0.85.
- Every level stays far below current practice (~0.53 null calls on the same splits).

### 6b. PTB MDR vs DS at lower levels / coarser types

| setting | calls | called pairs |
|---|---|---|
| 8 types, 0.90 (frozen) | 0/40 | - |
| 8 types, 0.80 | 0/40 | closest CD4 T IFNI −0.090 [−0.189, +0.009] |
| 8 types, 0.67 | 2/40 | Neutro IFNI −0.133 [−0.183, −0.083]; CD14 Mono IFNI +0.191 [+0.015, +0.368] |
| 5 types (combat5), 0.90 | 0/25 | closest Myeloid IFNg +0.061 [−0.000, +0.123] |
| 5 types (combat5), 0.80 | 1/25 | Myeloid IFNg +0.061 [+0.012, +0.110] |

- The two 0.67 calls are a **route change, not a weaker version of the 0.90 result**: at 0.67 IFNI switches to the
  unmixing route (info["unmix"]["programs"] = ["IFNI"]), and the IFNI estimates become a see-saw (B −0.17, DC −0.20,
  Neutro −0.13 vs CD14/CD16 Mono +0.19/+0.20; at 0.90 all eight are −0.03 to −0.14). Opposite-sign type effects
  around a weak bulk decrease are the leakage pattern the calibration exists to stop.
- The combat5 Myeloid IFNg call is positive while the bulk IFNg score is slightly negative (−0.027, p 0.34): it comes
  from the composition adjustment, at the edge of the interval.

## 7. Recommendation

1. **Keep level 0.90 as the primary standard** for PTB and the paper. Lowering it buys little power on known-answer
   data (+0.02 at 0.80) and breaks the per-dataset safety rule at both lower levels.
2. **Report PTB MDR vs DS as a null that is informative for neutrophils only**, with: the bulk program test (no
   program differs after BH); the spike-in (section 8: a TB-vs-healthy-sized difference is recovered and called in
   neutrophils, not in the minor types); the neutrophil 90% intervals (half-widths 0.04-0.17) as bounds.
   For minor types the study is not informative: the limit is reference leakage, which a lower level does not fix.
3. If an exploratory tier is wanted, **0.80 with its measured cost** (held-out mean null calls 0.035, worst
   dataset 0.112, vs ~0.53 for practice), labelled exploratory, never mixed with the primary calls. On PTB it adds
   nothing at 8 types; at 5 types one weak Myeloid IFNg call. Do not use 0.67: it changes the routing.
4. Nominal bulk IFN-I decrease in MDR (−5%, p 0.08) is a hypothesis for a validation cohort, not a finding.
5. Nothing to change in the package: `level=` and `calls="bh"` already exist; `run_ptb.py LEVEL=` writes the
   exploratory runs to separate `_l<level>` folders.

## 8. Spike-in power check on the real PTB bulk (2026-11-26)

`scripts/experiments/ptb_spike.py`: plants effects into the MDR samples of the real counts through the model's
forward equation (type profile x exp(d G), weighted by the base run's fractions; Poisson noise on the added part;
fractions unchanged), then reruns freeze v5 (level 0.90, same recipe). Results in results/ptb/spike/.

### 8a. Whole TB-vs-healthy matrix, scaled

| scale | planted pairs with abs(d) >= 0.1 | called, right sign | calls on abs(d) < 0.05 | slope (moved ~ planted) |
|---|---|---|---|---|
| 0.25 | 2 | 0 | 0/29 | 0.23 |
| 0.50 | 11 | 0 | 0/16 | 0.23 |
| 1.00 | 24 | 1 (Neutro IFNg) | 0/7 | 0.35 |

At scale 1.0, Neutro TLR (planted +0.074) is also called with the right sign; no wrong-sign or null calls.
Per-type recovery slope at scale 1.0: Neutro 1.01, B 0.73, CD4 T 0.61, CD16 Mono 0.37, CD14 Mono 0.35, CD8 T 0.34,
DC 0.02, NK −0.04. A full matrix of shifts leaks between the minor types (e.g. NK IFNg planted −0.36, moved +0.08),
and the size-matched interval widens with the planted size (B NLR half-width 0.32 -> 0.94).

### 8b. One planted pair at a time (IFNg +0.3 in each type)

| planted type | mean fraction | estimate [90% CI] | base | called | other calls |
|---|---|---|---|---|---|
| Neutro | ~0.5 | +0.324 [+0.197, +0.451] | +0.011 | yes | 3 (CD8 T IFNg +0.47, CD14 Mono IFNg +0.17, DC IFNg −0.39) |
| CD14 Mono | ~0.17 | +0.358 [+0.228, +0.489] | +0.100 | yes | 0 |
| DC | small | +0.149 [−0.071, +0.369] | +0.107 | no | 0 |
| CD16 Mono | small | +0.056 [−0.092, +0.205] | +0.033 | no | 0 |
| CD4 T | small | +0.054 [−0.028, +0.136] | +0.010 | no | 0 |
| NK | small | +0.028 [−0.093, +0.148] | +0.006 | no | 0 |
| CD8 T | small | +0.027 [−0.122, +0.176] | +0.004 | no | 0 |
| B | small | +0.004 [−0.145, +0.152] | +0.002 | no | 0 |

- A +0.3 IFNg shift is recovered and called in **neutrophils and CD14 monocytes** (the two large types); in the
  minor types the estimate barely moves (B +0.004 for +0.3 planted), so they are **not resolvable** in this study
  at any level: the attenuation is in the estimate, not in the threshold.
- The neutrophil spike produced **3 off-target calls** in other types' IFNg, two of them opposite in sign
  (CD8 T +0.47, DC −0.39): a large real shift in the dominant type leaks into minor types. This is a safety point
  to state for real whole blood (dominant neutrophils), and a reason not to read minor-type calls next to a strong
  neutrophil call without a check.

### 8c. What this means for PTB MDR vs DS

- Neutrophils and CD14 monocytes: a +0.3 shift would have been seen, so MDR vs DS truly lacks a shift of that size
  in those types (the null is informative there).
- Lymphocytes, NK, DC, CD16 monocytes: the study cannot resolve them (estimates attenuated toward 0 regardless of
  level). Report them as not resolvable, not as "no difference".
- Lowering the level cannot fix this; only a better-separating reference / more informative design could.
