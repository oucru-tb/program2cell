#!/bin/bash
# Single-group (TB-only) development run (benchmarking.md 0.8), splits 0-9, started after the held-out v2 lanes
# finish (shares the CPU). Four lanes; restart-safe.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=benchmarks/dev/single_group.log
until [ "$(cat benchmarks/dev/heldout_v2_?.log | grep -c '^DONE')" -ge 4 ]; do sleep 300; done
echo "START $(date)" >> $LOG
python scripts/experiments/single_group.py run mcf_grad_ida 0-9 >> $LOG 2>&1 &
python scripts/experiments/single_group.py run mcf_grad_tram 0-9 >> $LOG 2>&1 &
python scripts/experiments/single_group.py run combat_covid 0-4 >> $LOG 2>&1 &
python scripts/experiments/single_group.py run combat_covid 5-9 >> $LOG 2>&1 &
wait
python scripts/experiments/single_group.py summary > benchmarks/dev/single_group_summary.txt 2>&1
echo "DONE $(date)" >> $LOG
