#!/bin/bash
# Development comparison of estimators (causal_loop) on several datasets, one dataset after another.
# Usage: DS="kang mcf_ida" SPLITS=0-2 EST=auto_sh LOG=benchmarks/dev/dev_a.log bash dev_chain.sh. Restart-safe.
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd /c/Users/duy.pham/Documents/dev/program2cell
LOG=${LOG:-benchmarks/dev/dev_chain.log}
echo "START $(date) $DS $SPLITS $EST" >> $LOG
for ds in $DS; do
  python scripts/experiments/causal_loop.py run $ds $SPLITS $EST >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
