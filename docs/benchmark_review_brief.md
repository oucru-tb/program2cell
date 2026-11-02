# Benchmark design for cell-type attribution from bulk RNA-seq: request for review

*PALM project, OUCRU. Draft of 2026-11-02. We would value your critique of the evaluation design before we
freeze the method and run the final held-out test.*

## 1. The question we evaluate

Bulk RNA-seq shows that a gene program (e.g. interferon response) changed in a sample. We ask **which cell
type changed it**: for each cell type *c* and program *p*, did *p* change in *c*, by how much, and can the
data tell? We call this *attribution*.
- It differs from deconvolution, which asks how many cells of each type there are.
- It differs from bulk differential expression, which asks what changed in the sample.
- Our method (PALM) estimates per-cell-type program changes with 90% intervals and "can't tell" calls.
- The benchmark is meant to be method-agnostic.

## 2. How the answer key is built

1. **Hidden answer in real data.** We use single-cell perturbation datasets (real drugs and cytokines).
   - Each test sample's cells are split by binomial thinning.
   - Half A, summed over cell types, is the bulk that methods see.
   - Half B, kept per cell type, gives the true change per (cell type, program). It is never shown to the
     method.
   - Programs are 7 PROGENy footprints, about 500 weighted genes each.
2. **Controls fixed by biology before running.** Responders come from genetics and were checked on target
   genes before scoring.
   - Idasanutlin (MDM2 inhibitor): p53 up only in TP53-wild-type lines.
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
- **Calling rules, all reported:**
  1. BH q < 0.1;
  2. q < 0.1 and |change| ≥ 0.1;
  3. threshold-free ranking (AUROC).

  Rules 2 and 3 were added after we found that, on assembled (nearly noise-free) bulk, rule 1 turns small
  biases into "significant" calls. The comparator's false flags *rise* with sample size (24% → 96% from 3 to
  20 paired samples).
- **Metrics:**
  - unchanged boxes flagged;
  - real changes found with the right sign;
  - non-responder types flagged in the known program;
  - 90%-interval coverage;
  - r and mean absolute error;
  - AUROC.

  Threshold curves are reported instead of single cut-offs.
- **Second answer key:** a different activity model (ULM), scored with scale-free metrics only, to check that
  the method doesn't win by sharing the truth's estimator.
- **Design limit:** a Gauss–Markov bound on the smallest change any unbiased method could detect, given
  proportions, composition spread, sample size and noise. It is used to say which boxes are resolvable at all.

## 4. Results so far (development splits; not final)

| | p53 drug | MEK drug | Interferon |
|---|---|---|---|
| Unchanged flagged: t-test (q only) / t-test (q + cut-off) / PALM | 91 / 7 / 5% | 81 / 5 / 4% | 83 / 5 / 0% |
| 90% interval holds the truth: t-test / PALM | 7 / 92% | 2 / 97% | 55 / 100% |
| Real changes found (\|true\| ≥ 0.1): t-test with cut-off / PALM | 49 / 57% | 69 / 80% | 94 / 2% |
| Ranking (AUROC): t-test / PALM | 0.80 / 0.79 | 0.93 / 0.95 | 0.99 / 0.98 |

**Reading:**
- The comparators' intervals are confidently wrong.
- Under a fair cut-off the false-flag gap mostly closes.
- Mis-attribution survives: p53-mutant lines are still flagged as p53 responders.
- PALM finds more type-specific changes but misses changes shared by all cell types (interferon). There it
  reports a program-level change instead.
- Ranking is similar under both answer keys.
- On real whole blood (GSE60424), PALM made no false calls in 4 comparisons. The comparator flagged 24% of
  unchanged boxes in sepsis.

## 5. Known limitations and planned fixes

1. **Composition is identical in both arms** of each assembled pair. In disease it shifts. A stress test
   with treatment-correlated shifts is running.
2. **Reference and calibration come from the same experiment** as the bulk, so there is no batch or platform
   shift. Planned: reference from another study or plate.
3. **The method was developed on these datasets.** A new freeze, a written protocol, fresh splits and
   untouched datasets (rest of Tahoe; Parse 10M PBMC cytokines) are planned before final numbers.
4. **Comparators:** CIBERSORTx high-resolution, bMIND and ENIGMA are not yet included. TCA and TOAST are
   scored with our ULM-on-log-fold-change test, not their native tests.
5. **Independence:** splits reuse the same cells, so the replication unit should be the experiment, not the
   split.
6. **Composition spread** in assembled mixtures is larger than in blood cohorts. A tighter-spread variant is
   running.

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
6. What would make you trust the held-out test: a pre-registered protocol, an independent party running it,
   or something else?
7. Anything missing that would make you reject the claims?

*Contact: Duy Pham (duy.pham@oucru.org). Full protocol and code: `docs/benchmarking.md` §0 and
`scripts/experiments/` in the project repository.*
