#!/bin/bash
# Method freeze v5 (docs/method_spec.md, "Method freeze v5"): held-out splits 30-39, never run before, ONE run with
# the frozen setting (git tag method-freeze-v5). PALM effects from freeze v5 ("v5") and freeze v2 ("auto_v2") on the
# same splits, so the pre-registered comparison (heldout_v5_compare.py) is out of sample for both.
# Comparators: CIBERSORT and InstaPrism only (for practice_vs_palm); TOAST, TCA and cellGeometry were compared on
# held-out 20-29 and 40-49 and are not rerun. COMBAT: combat_v5.py (lane d).
#   LANE=a|b|c|d bash heldout_v5_chain.sh      (lanes run in parallel; restart-safe: finished files are skipped)
#   LANE=e: transport check, 9 perturbations never used in v5 development (v5, auto_v2; PALM-lite fractions)
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONIOENCODING=utf-8
export PALM_EST=v5
cd /c/Users/duy.pham/Documents/dev/program2cell
SPLITS=30-39
LOG=benchmarks/dev/heldout_v5_$LANE.log
COMP="cibersort,instaprism"
case $LANE in
  a) DS="mcf_ida"; CFG='{"joint nb": {"likelihood": "nb"}}' ;;
  b) DS="mcf_tram"; CFG='{"joint nb": {"likelihood": "nb"}}' ;;
  c) DS="kang tahoe_rmc tahoe_enc tahoe_tram tahoe_gem"; CFG='{}' ;;
  d) DS="zhang parse_ifng parse_ifnb parse_gmcsf parse_il4 parse_tnf"; CFG='{}' ;;
esac
if [ "$LANE" = e ]; then
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  echo "START $(date) lane e transport (freeze tag: $(git describe --tags --always 2>/dev/null))" >> $LOG
  for ds in parse_ifna parse_il10 parse_il1b parse_il6 tahoe_afa tahoe_alp tahoe_bor tahoe_evr tahoe_ptx; do
    python scripts/experiments/causal_loop.py run $ds $SPLITS v5,auto_v2 >> $LOG 2>&1
  done
  echo "DONE $(date)" >> $LOG
  exit 0
fi
echo "START $(date) lane $LANE (freeze tag: $(git describe --tags --always 2>/dev/null), PALM_EST=$PALM_EST)" >> $LOG
for ds in $DS; do
  cfg=$CFG
  case $ds in kang|zhang) cfg='{"joint nb": {"likelihood": "nb"}}' ;; esac
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout --configs "$cfg" --competitors $COMP >> $LOG 2>&1
  rm -rf benchmarks/dev/work/$ds/s*/instaprism*
  python scripts/experiments/causal_loop.py run $ds $SPLITS v5,auto_v2,diff >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
if [ "$LANE" = d ]; then
  python scripts/experiments/combat_v5.py $SPLITS >> $LOG 2>&1
fi
echo "DONE $(date)" >> $LOG
