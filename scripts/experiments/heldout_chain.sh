#!/bin/bash
# Method freeze v1 (docs/method_spec.md): the held-out splits 10-12, run ONCE with the frozen setting.
# PALM fractions (joint nb) + baselines + CIBERSORT (dev_bench --heldout), PALM's default effects (auto_m; diff
# for reference), current practice vs the PALM module (practice_vs_palm). Rows are s10-s12 files in the usual
# directories; summaries: causal_loop.py summary heldout, practice_vs_palm.py summary heldout.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/heldout.log
SPLITS=10-12
DS="mcf_ida mcf_tram zhang kang"
# wait for the development scoring run to finish (shares the CPU)
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*practice_vs_palm.py run*' }) { exit 0 } else { exit 1 }"; do
  sleep 120
done
echo "START $(date) (freeze tag: $(git describe --tags --always 2>/dev/null))" >> $LOG
for ds in $DS; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout \
    --configs '{"joint nb": {"likelihood": "nb"}}' --competitors cibersort >> $LOG 2>&1
done
python scripts/experiments/causal_loop.py run ${DS// /,} $SPLITS auto_m,diff >> $LOG 2>&1
python scripts/experiments/practice_vs_palm.py run ${DS// /,} $SPLITS naive,reallocation,cibersort >> $LOG 2>&1
{ python scripts/experiments/causal_loop.py summary heldout; echo; python scripts/experiments/practice_vs_palm.py summary heldout; } \
  > benchmarks/dev/heldout_summary.txt 2>&1
echo "DONE $(date)" >> $LOG
