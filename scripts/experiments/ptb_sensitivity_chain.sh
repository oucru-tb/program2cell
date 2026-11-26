#!/bin/bash
# PTB sensitivity analysis (2026-11-25), after the null PTB MDR vs DS result (0 of 40 calls at freeze v5, level 0.90).
# Post-hoc: the frozen method is unchanged; nothing here is a new freeze or a pre-registered test.
#   LANE=p : PTB MDR vs DS at interval level 0.80 and 0.67 (8 types), then 5 coarse types (REF=combat5) at 0.90 / 0.80
#   LANE=x|y|z : freeze v5 at level 0.80 / 0.67 (causal_loop v5_l80, v5_l67) on held-out splits 30-39, 13 datasets,
#                to measure the null-call rate and power of the lower tiers on known-answer data
# Results: docs/ptb_sensitivity_2026-11-25.md (ptb_sensitivity.py summarises)
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONIOENCODING=utf-8
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/ptb_sensitivity_$LANE.log
SPLITS=30-39
echo "START $(date) lane $LANE ($(git describe --tags --always 2>/dev/null))" >> $LOG
case $LANE in
  p)
    export EST=v5 PROG=immune_reactome_innate5 OFFSET=anchored CMP=cmp2 COV=batch
    for spec in "combat 0.80" "combat 0.67" "combat5 0.90" "combat5 0.80"; do
      set -- $spec
      REF=$1 LEVEL=$2 python scripts/experiments/run_ptb.py >> $LOG 2>&1
    done ;;
  x) DS="mcf_ida mcf_tram tahoe_enc tahoe_gem" ;;
  y) DS="tahoe_rmc tahoe_tram kang zhang" ;;
  z) DS="parse_gmcsf parse_ifnb parse_ifng parse_il4 parse_tnf" ;;
esac
if [ "$LANE" != p ]; then
  for name in v5_l80 v5_l67; do
    for ds in $DS; do
      python scripts/experiments/causal_loop.py run $ds $SPLITS $name >> $LOG 2>&1
    done
  done
fi
echo "DONE $(date)" >> $LOG
