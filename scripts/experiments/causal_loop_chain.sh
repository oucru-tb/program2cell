#!/bin/bash
# After the running scoreboard (and its auto rerun) and the trametinib PALM fits: move the PALM-lite trametinib
# rows aside, then score every estimator on the four datasets, splits 0-2, and print the summary.
cd /c/Users/duy.pham/Documents/dev/program2cell
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3
OUT=benchmarks/dev/causal_loop
until grep -q "^targets:" "$1" 2>/dev/null; do sleep 120; done
until [ -f benchmarks/dev/mcf_tram/s2__joint_nb.npz ]; do sleep 120; done
mkdir -p $OUT/lite
for f in $OUT/mcf_tram__*.json; do
  grep -q '"fractions": "lite"' "$f" && mv "$f" $OUT/lite/
done
$E/python.exe -W ignore scripts/experiments/causal_loop.py run mcf_ida,mcf_tram,kang,zhang 0-2 >> benchmarks/dev/causal_loop2.log 2>&1
$E/python.exe -W ignore scripts/experiments/causal_loop.py summary >> benchmarks/dev/causal_loop2.log 2>&1
echo DONE >> benchmarks/dev/causal_loop2.log
