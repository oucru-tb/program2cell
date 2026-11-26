#!/bin/bash
# Paper 1 Extended Data (2026-11-23, docs/heldout_v5_results.md 4b): the PALM joint NB fraction fit on the held-out
# v5 splits 30-39 for Tahoe and Parse, where the v5 protocol used PALM-lite fractions, so the fraction-accuracy table
# has every method on every dataset. Descriptive only: no PALM setting changes, no effect is recomputed.
# Output goes to benchmarks/dev/fraction_joint/<ds>/, NOT benchmarks/dev/<ds>/: causal_loop._setup switches to
# joint-fit fractions when benchmarks/dev/<ds>/s<k>__joint_nb.npz exists, which would change any rerun of the frozen
# v5 effects. Restart-safe (dev_bench skips finished files). Not to run alongside a PTB job.
#   LANE=p|t1|t2 bash fraction_joint_chain.sh
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
cd /c/Users/duy.pham/Documents/dev/program2cell
SPLITS=30-39
OUTDIR=benchmarks/dev/fraction_joint
LOG=$OUTDIR/chain_$LANE.log
mkdir -p $OUTDIR
case $LANE in
  p) DS="parse_ifng parse_ifnb parse_gmcsf parse_il4 parse_tnf" ;;
  p2) DS="parse_tnf parse_il4 parse_gmcsf" ;;   # second Parse lane (reverse order; added 16:30 when Parse ran ~1.7 h/split)
  t1) DS="tahoe_tram tahoe_gem" ;;
  t2) DS="tahoe_enc tahoe_rmc" ;;
esac
echo "START $(date) lane $LANE (commit $(git rev-parse --short HEAD))" >> $LOG
for ds in $DS; do
  python scripts/dev_bench.py run --dataset $ds --splits $SPLITS --heldout --out $OUTDIR \
    --configs '{"joint nb": {"likelihood": "nb"}}' >> $LOG 2>&1
done
echo "DONE $(date)" >> $LOG
