#!/bin/bash
# C1 (research_plan 3.10): competitor fits on the known-answer datasets, splits 0-2 (dev_bench rows + arrays).
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export PALM_BLADE_PYTHON='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-blade\python.exe'
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/competitors_c1.log
echo "START $(date)" >> $LOG
for comp in toast,tca cellgeometry bayesprism blade; do
  for ds in mcf_ida mcf_tram zhang kang; do
    python scripts/dev_bench.py run --dataset $ds --splits 0-2 --configs '{}' --competitors $comp >> $LOG 2>&1
  done
done
echo "DONE $(date)" >> $LOG
