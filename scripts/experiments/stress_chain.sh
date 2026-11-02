#!/bin/bash
# Stress tests (docs/benchmarking.md "Stress tests"), splits 0-2: McFarland with the composition shifted by treatment
# (mcf_*_shift) and with less composition spread (mcf_*_tight). PALM fractions (joint nb, ~5 min per split with 6
# lines) + baselines + CIBERSORT + InstaPrism (dev_bench), PALM effects (causal_loop auto_m, auto_abs), current
# practice (practice_vs_palm). Restart-safe.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/stress.log
SPLITS=${SPLITS:-0-2}
DS=${DS:-"mcf_ida_shift mcf_ida_tight mcf_tram_shift mcf_tram_tight"}
echo "START $(date)" >> $LOG
for ds in $DS; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS \
    --configs '{"joint nb": {"likelihood": "nb"}}' --competitors cibersort,instaprism >> $LOG 2>&1
  python scripts/experiments/causal_loop.py run $ds $SPLITS auto_m,auto_abs >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
