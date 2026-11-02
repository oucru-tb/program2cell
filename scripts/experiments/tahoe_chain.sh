#!/bin/bash
# Tahoe-100M plate 3 benchmark (build_tahoe.py), development splits 0-2: mixtures, then per drug arm
# baselines + CIBERSORT + InstaPrism (dev_bench; PALM fractions from the fast engine, PALM-lite: the full NB fit
# takes ~75 min per split with 9 lines; tahoe_dab s0 has one for comparison), PALM effects (causal_loop auto_m,
# auto_abs), current practice vs PALM module (practice_vs_palm). Restart-safe: every step skips finished files.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/tahoe.log
SPLITS=${SPLITS:-0-2}
DS=${DS:-"tahoe_dab tahoe_adag tahoe_tram tahoe_dmso2 tahoe_gem tahoe_oxa"}
echo "START $(date)" >> $LOG
[ -f data/tahoe/tahoe_mix.h5ad ] || python -I scripts/experiments/build_tahoe.py mix >> $LOG 2>&1
for ds in $DS; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS \
    --configs '{}' --competitors cibersort,instaprism >> $LOG 2>&1
  python scripts/experiments/causal_loop.py run $ds $SPLITS auto_m,auto_abs >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
