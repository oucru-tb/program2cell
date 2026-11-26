#!/bin/bash
# Single-pair spike-in power curve on PTB MDR vs DS (2026-11-26): IFNg +0.3 planted in one type at a time.
# LANE=1|2|3 splits the 8 types; log benchmarks/dev/ptb_spike_single_$LANE.log
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8
cd /c/Users/duy.pham/Documents/dev/program2cell
case $LANE in 1) TY=("Neutro" "CD14 Mono" "B");; 2) TY=("CD4 T" "NK" "DC");; 3) TY=("CD8 T" "CD16 Mono");; esac
for t in "${TY[@]}"; do SINGLE="$t:IFNg:${D:-0.3}" python scripts/experiments/ptb_spike.py >> benchmarks/dev/ptb_spike_single_$LANE.log 2>&1; done
echo DONE >> benchmarks/dev/ptb_spike_single_$LANE.log
