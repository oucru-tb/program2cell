#!/bin/bash
# Method freeze v4 (docs/method_spec.md, "Method freeze v4"): fresh held-out splits 40-49, never run before, ONE run
# with the frozen setting (git tag method-freeze-v4). As heldout_v2_chain.sh, with PALM effects from both the frozen
# v4 default (causal_loop v4g; v4g_pair on the paired designs Kang and Zhang) and freeze v2 (auto_v2) on the same
# splits, so the pre-registered comparison (heldout_v4_compare.py) is out of sample for both. COMBAT: combat_v4.py.
#   LANE=a|b|c|d bash heldout_v4_chain.sh      (lanes run in parallel; restart-safe: finished files are skipped)
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONIOENCODING=utf-8
export PALM_EST=v4g
cd /c/Users/duy.pham/Documents/dev/program2cell
SPLITS=40-49
LOG=benchmarks/dev/heldout_v4_$LANE.log
FULL="cibersort,instaprism,toast,tca,cellgeometry"
case $LANE in
  a) DS="mcf_ida"; CFG='{"joint nb": {"likelihood": "nb"}}'; COMP=$FULL ;;
  b) DS="mcf_tram"; CFG='{"joint nb": {"likelihood": "nb"}}'; COMP=$FULL ;;
  c) DS="kang tahoe_rmc tahoe_enc tahoe_tram tahoe_gem"; CFG='{}'; COMP="cibersort,instaprism" ;;
  d) DS="zhang parse_ifng parse_ifnb parse_gmcsf parse_il4 parse_tnf"; CFG='{}'; COMP="cibersort,instaprism" ;;
esac
echo "START $(date) lane $LANE (freeze tag: $(git describe --tags --always 2>/dev/null), PALM_EST=$PALM_EST)" >> $LOG
for ds in $DS; do
  cfg=$CFG; comp=$COMP; est="v4g,auto_v2,diff"
  case $ds in kang|zhang) cfg='{"joint nb": {"likelihood": "nb"}}'; comp=$FULL; est="v4g_pair,auto_v2,auto_v2_pair,diff" ;; esac
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout --configs "$cfg" --competitors $comp >> $LOG 2>&1
  rm -rf benchmarks/dev/work/$ds/s*/instaprism* benchmarks/dev/work/$ds/s*/cellgeometry*
  python scripts/experiments/causal_loop.py run $ds $SPLITS $est >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
if [ "$LANE" = d ]; then
  python scripts/experiments/combat_v4.py $SPLITS >> $LOG 2>&1
fi
echo "DONE $(date)" >> $LOG
