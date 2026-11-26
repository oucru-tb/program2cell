#!/bin/bash
# Paper 1 completion (2026-11-23, docs/paper1_brief_communication_plan.md): TOAST and TCA on the held-out v5 splits
# 30-39, so the current-practice table uses one split set. heldout_v5_chain.sh ran CIBERSORT and InstaPrism only;
# TOAST/TCA ran on 20-29 and 40-49 on the same 4 datasets (McFarland idasanutlin, trametinib, Kang, Zhang). No PALM
# setting changes: dev_bench adds only the missing competitor files (restart-safe), practice_vs_palm adds only the
# missing toast/tca rows (PALM_EST=v5 as in the v5 chain). Not to run alongside a PTB job.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export PALM_RSCRIPT='C:\Users\duy.pham\AppData\Local\miniforge3\envs\palm-r\Lib\R\bin\x64\Rscript.exe'
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONIOENCODING=utf-8
export PALM_EST=v5
cd /c/Users/duy.pham/Documents/dev/program2cell
SPLITS=30-39
LOG=benchmarks/dev/heldout_v5_csde.log
echo "START $(date) TOAST/TCA on $SPLITS (commit $(git rev-parse --short HEAD))" >> $LOG
for ds in mcf_ida mcf_tram kang zhang; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout --configs '{}' --competitors toast,tca >> $LOG 2>&1
  python scripts/experiments/practice_vs_palm.py run $ds $SPLITS >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
