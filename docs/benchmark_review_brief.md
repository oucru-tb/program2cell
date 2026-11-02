# Benchmark design for cell-type attribution from bulk RNA-seq: request for review

*PALM project, OUCRU. Draft of 2026-11-02. We would value your critique of the evaluation design before we
freeze the method and run the final held-out test.*

## Summary

We want to know whether a bulk RNA-seq method can say **where a gene-program change comes from**: which cell
types, how much from more cells versus cells that changed, and which cell types themselves changed.
- **Answer key:** we hide the true per-cell-type answer inside real perturbation data by splitting each
  sample's cells. Controls are fixed by genetics in advance.
- **Comparators:** standard deconvolution tools and the usual t-test, under three calling rules (q only,
  q plus an effect-size cut-off, threshold-free ranking).
- **What we found so far:**
  - Under a fair cut-off, the comparators' false flags fall to the level of our method's.
  - The comparators' 90% intervals still hold the truth only 2–55% of the time (ours 92–100%), and they still
    label non-responding cell lines as responders.
  - Ranking accuracy is similar for both, under two independent answer keys.
  - Our method misses per-cell changes shared by all cell types (interferon), but recovers where that signal
    comes from better than the comparator.
- **Known weaknesses:**
  - composition identical across arms;
  - reference from the same experiment;
  - development and test overlap;
  - missing comparators.
- We ask seven specific questions (section 6).

## 1. The question we evaluate

Bulk RNA-seq shows that a gene program (e.g. interferon response) changed in a sample. We ask two linked
questions, together called *attribution*:
1. **Contribution:** which cell types does the bulk change come from, and how much is more cells versus cells
   that changed? (e.g. "IFN-γ up: 75% from monocytes, 25% from neutrophils")
2. **Per-cell change:** inside each cell type, did the program change, by how much, and can the data tell?

These differ from deconvolution (how many cells) and from bulk differential expression (what changed in the
sample). They also differ from each other: an abundant cell type can contribute most of a bulk change with
only a small change per cell.

Our method (PALM) estimates both, with 90% intervals and "can't tell" calls. The benchmark is meant to be
method-agnostic.

## 2. How the answer key is built

1. **Hidden answer in real data.** We use single-cell perturbation datasets (real drugs and cytokines).
   - Each test sample's cells are split by binomial thinning.
   - Half A, summed over cell types, is the bulk that methods see.
   - Half B, kept per cell type, gives the true per-cell change and the true contributions. It is never shown
     to the method.
   - Programs are 7 PROGENy footprints, about 500 weighted genes each.
2. **Controls fixed by biology before running.** Responders come from genetics and were checked on target
   genes before scoring.
   - Idasanutlin: p53 up only in TP53-wild-type lines.
   - BRAF inhibitor: MAPK down only in BRAF-V600 lines.
   - KRAS-G12C inhibitor: MAPK down only in G12C lines.
   - Negative controls:
     - unchanged boxes;
     - non-target programs;
     - non-responder lines;
     - vehicle vs vehicle.
3. **Separation of training and test.**
   - Reference profiles and any calibration come only from other donors, or from a separate pool of control
     cells.
   - Targets were set in advance:
     - unchanged boxes flagged ≤ 10%;
     - 90%-interval coverage ≥ 85%;
     - r ≥ 0.9;
     - responders found ≥ 90%.
   - The method is frozen, then checked once on held-out splits.

**Datasets (a ladder; each rung covers a weakness of the previous one):**

| Rung | Data | Truth from | Weakness |
|---|---|---|---|
| Assembled bulk, real responses | McFarland 2020 MIX-Seq (6 lines; idasanutlin, trametinib); Kang 2018 IFN-β (PBMC, 4 types); Zhang 2021 TNBC blood (chemo ± anti-PD-L1); Tahoe-100M plate 3 (9 lines; 5 drugs + vehicle) | Split cells + genetics | Bulk is assembled, not sequenced |
| Real bulk | GSE60424: whole blood + sorted neutrophils, monocytes, B, CD4, CD8 from the same 20 donors | Sorted cells, same donors | Sorting shifts stress genes; n = 3–4 per group |
| Application | TB patients vs healthy (in-house) | Known TB biology (positive control); random gene sets, shuffled labels (negative) | No full answer key |

## 3. Comparators, calling rules, metrics

- **Comparators:** CIBERSORT, InstaPrism, NNLS and a bulk score, each followed by the paired t-test users
  typically run. Also a lab workflow (InstaPrism → GSVA → correlation), and TCA and TOAST.
- **Calling rules for per-cell change, all reported:**
  1. BH q < 0.1;
  2. q < 0.1 and |change| ≥ 0.1;
  3. threshold-free ranking (AUROC).

  Rules 2 and 3 were added after we found that, on assembled (nearly noise-free) bulk, rule 1 turns small
  biases into "significant" calls. The comparator's false flags *rise* with sample size (24% → 96% from 3 to
  20 paired samples).
- **Per-cell metrics:**
  - unchanged boxes flagged;
  - real changes found with the right sign;
  - non-responder types flagged;
  - 90%-interval coverage;
  - r, mean absolute error, AUROC.

  Threshold curves are reported instead of single cut-offs.
- **Contribution metrics:**
  - error in each cell type's share of the bulk change;
  - r of contributions;
  - recovery of the "more cells" part;
  - coverage of the share intervals.

  Both methods are put through the same decomposition formula.
- **Second answer key:** a different activity model (ULM), scored with scale-free metrics only.
- **Design limit:** a Gauss–Markov bound on the smallest per-cell change any unbiased method could detect,
  given proportions, composition spread, sample size and noise.

## 4. Results so far (development splits; not final)

**Per-cell change** (InstaPrism + t-test vs PALM):

| | p53 drug | MEK drug | Interferon |
|---|---|---|---|
| Unchanged flagged: t-test (q only) / t-test (q + cut-off) / PALM | 91 / 7 / 5% | 81 / 5 / 4% | 83 / 5 / 0% |
| 90% interval holds the truth: t-test / PALM | 7 / 92% | 2 / 97% | 55 / 100% |
| Real changes found (\|true\| ≥ 0.1): t-test with cut-off / PALM | 49 / 57% | 69 / 80% | 94 / 2% |
| Ranking (AUROC): t-test / PALM | 0.80 / 0.79 | 0.93 / 0.95 | 0.99 / 0.98 |

**Contribution** (same formula for both methods):

| | p53 drug | MEK drug | Interferon |
|---|---|---|---|
| Share error, mean abs (0–1): InstaPrism / PALM | 0.17–0.20 / 0.16–0.21 | 0.08–0.12 / 0.11–0.16 | 0.10–0.15 / 0.07–0.10 |
| Contribution r: InstaPrism / PALM | 0.75–0.76 / 0.85–0.90 | 0.91–0.92 / 0.80–0.87 | 0.81–0.92 / 0.90–0.97 |
| "More cells" part, truth / InstaPrism / PALM | 32–35 / 63–67 / 52–63% | 44–48 / 43–48 / 26–48% | 4–7 / 21–26 / 4–7% |
| PALM share interval holds the truth | 96–100% | 92–100% | 100% |

**Reading:**
- The comparators' intervals are confidently wrong.
- Under a fair cut-off the false-flag gap mostly closes, but mis-attribution survives: p53-mutant lines are
  still flagged as p53 responders.
- Both methods rank boxes about equally, under the main answer key and under the independent second one, so
  there is no home-field advantage.
- The conclusions hold at every threshold tested (0.02–0.1 for "unchanged", 0.05–0.2 for "real").
- PALM finds more cell-type-specific changes. It misses per-cell changes shared by all cell types
  (interferon), but gets their contributions right.
- On contributions the two methods are close on the drugs, each better on one.
- In the cell-line data the "more cells" part is really *more reads*: the drugs change each line's RNA output
  while cell numbers are equal.
- On real whole blood (GSE60424), PALM made no false calls in 4 comparisons. The comparator flagged 24% of
  unchanged boxes in sepsis.
- On a 9-line drug screen (Tahoe, first drug only), PALM's intervals held the truth 89–100% of the time with no
  false flags. The drug's effect was too small for either method to find the responders.

## 5. Known limitations and planned fixes

1. **Composition is identical in both arms** of each assembled pair. In disease it shifts. A stress test with
   treatment-correlated shifts (responder lines ×1.6) is running. It also tests the "more cells" vs
   "cells changed" split.
2. **Reference and calibration come from the same experiment** as the bulk, so there is no batch or platform
   shift. Planned: reference from another study or plate.
3. **The method was developed on these datasets.** A new freeze, a written protocol, fresh splits and
   untouched datasets (rest of Tahoe; Parse 10M PBMC cytokines) are planned before final numbers.
4. **Comparators:** CIBERSORTx high-resolution, bMIND and ENIGMA are not yet included. TCA and TOAST are
   scored with our ULM-on-log-fold-change test, not their native tests. A scoring bug in that step was found
   and fixed today, and they are being rescored.
5. **Independence:** splits reuse the same cells, so the replication unit should be the experiment, not the
   split.
6. **Composition spread** in assembled mixtures is larger than in blood cohorts. A tighter-spread variant is
   running.
7. **Read shares vs cell numbers:** contributions are shares of the bulk signal. A change in RNA output per
   cell looks like "more cells" unless RNA content is modelled.

## 6. Questions for you

1. Is binomial thinning of each specimen an acceptable way to create a hidden per-cell-type truth? Would you
   require real bulk with matched sorted or single-cell truth as the primary evidence?
2. Are the three calling rules a fair treatment of the comparators? Is there a better standard (e.g. TOST
   equivalence tests, or the comparators' native tests only)?
3. Which comparators would you consider essential? (CIBERSORTx high-resolution? bMIND? ENIGMA? Others?)
4. Is "proportions identical across arms" a serious flaw, and what shift magnitude would you consider
   realistic?
5. Is a Gauss–Markov design limit an adequate notion of identifiability here, given that the leakage
   operator is itself estimated?
6. For contributions, is a first-order (Shapley) split into "more cells" and "cells changed" the right
   estimand? How should RNA content per cell be handled?
7. What would make you trust the held-out test (a pre-registered protocol, an independent party running it),
   and is anything missing that would make you reject the claims?

*Contact: Duy Pham (duy.pham@oucru.org). Full protocol and code: `docs/benchmarking.md` §0 and
`scripts/experiments/` in the project repository.*
