#!/bin/bash
# Parse 10M PBMC cytokine benchmark (build_parse.py), development splits 0-2: per cytokine vs PBS
# baselines + CIBERSORT + InstaPrism (dev_bench; PALM fractions from PALM-lite), PALM effects (causal_loop auto_m,
# auto_abs), current practice vs PALM module (practice_vs_palm). Restart-safe: every step skips finished files.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/parse.log
SPLITS=${SPLITS:-0-2}
DS=${DS:-"parse_ifng parse_ifnb parse_il4 parse_gmcsf parse_tnf"}
echo "START $(date)" >> $LOG
for ds in $DS; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS \
    --configs '{}' --competitors cibersort,instaprism >> $LOG 2>&1
  python scripts/experiments/causal_loop.py run $ds $SPLITS auto_m,auto_abs >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
