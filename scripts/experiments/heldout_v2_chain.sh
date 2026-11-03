#!/bin/bash
# Method freeze v2 (docs/method_spec.md): fresh held-out splits 20-29, never run before, ONE run with the frozen
# setting (git tag method-freeze-v2). Per dataset: PALM fractions (joint NB where the fit is affordable, else
# PALM-lite inside causal_loop) + baselines + competitors (dev_bench --heldout), PALM effects (causal_loop $PALM_EST,
# diff for reference), current practice vs the PALM module (practice_vs_palm). Run one lane per call:
#   LANE=a|b|c|d bash heldout_v2_chain.sh      (lanes run in parallel; restart-safe: finished files are skipped)
# Scratch (InstaPrism / cellGeometry work folders) is deleted after each dataset; TOAST/TCA logfc.csv is kept.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export PALM_EST=${PALM_EST:-auto_v2}
cd /c/Users/duy.pham/Documents/dev/program2cell
SPLITS=${SPLITS:-20-29}
LOG=benchmarks/dev/heldout_v2_$LANE.log
FULL="cibersort,instaprism,toast,tca,cellgeometry"
case $LANE in
  a) DS="mcf_ida"; CFG='{"joint nb": {"likelihood": "nb"}}'; COMP=$FULL ;;
  b) DS="mcf_tram"; CFG='{"joint nb": {"likelihood": "nb"}}'; COMP=$FULL ;;
  c) DS="kang tahoe_rmc tahoe_enc tahoe_tram tahoe_gem"; CFG='{}'; COMP="cibersort,instaprism" ;;
  d) DS="zhang parse_ifng parse_ifnb parse_gmcsf parse_il4 parse_tnf"; CFG='{}'; COMP="cibersort,instaprism" ;;
esac
echo "START $(date) lane $LANE (freeze tag: $(git describe --tags --always 2>/dev/null), PALM_EST=$PALM_EST)" >> $LOG
for ds in $DS; do
  cfg=$CFG; comp=$COMP
  case $ds in kang|zhang) cfg='{"joint nb": {"likelihood": "nb"}}'; comp=$FULL ;; esac
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout --configs "$cfg" --competitors $comp >> $LOG 2>&1
  rm -rf benchmarks/dev/work/$ds/s*/instaprism* benchmarks/dev/work/$ds/s*/cellgeometry*
  python scripts/experiments/causal_loop.py run $ds $SPLITS $PALM_EST,diff >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
if [ "$LANE" = d ]; then
  python scripts/experiments/combat_real.py run $SPLITS sepsis >> $LOG 2>&1
  python scripts/experiments/combat_real.py run $SPLITS severe >> $LOG 2>&1
fi
echo "DONE $(date)" >> $LOG
