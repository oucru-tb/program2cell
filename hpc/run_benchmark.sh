#!/usr/bin/env bash
# Benchmark job: (1) pseudo-bulk from single cells, (2) PALM vs the other methods with exact truth.
# Settings live in hpc/config.sh; run hpc/setup_envs.sh once first.
#
# Submit (both header sets are here; each scheduler ignores the other's lines):
#   LSF   : bsub < hpc/run_benchmark.sh
#   SLURM : sbatch hpc/run_benchmark.sh
#   no scheduler / interactive node: bash hpc/run_benchmark.sh
# Run only some steps:  STEPS="compare" bsub < hpc/run_benchmark.sh   (STEPS: build, compare)
#
# Resources: the 50k-cell build needs ~8 GB; the full 1.25M build ~40 GB (raw counts read in chunks,
# only the sums are kept). The comparison is dominated by PALM: check the runtime estimate it prints.
#BSUB -J palm_bench
#BSUB -o logs/palm_bench.%J.out
#BSUB -e logs/palm_bench.%J.err
#BSUB -n 8
#BSUB -R "select[mem>32000] rusage[mem=32000] span[hosts=1]"
#BSUB -M 32000
#BSUB -q long
#SBATCH --job-name=palm_bench
#SBATCH --output=logs/palm_bench.%j.out
#SBATCH --error=logs/palm_bench.%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail
HERE="${LS_SUBCWD:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$HERE"
source hpc/config.sh
source "$CONDA_BASE/etc/profile.d/conda.sh"
mkdir -p "$OUTDIR" logs
STEPS="${STEPS:-build compare}"

# one BLAS/XLA thread pool sized to the job (PALM's parallel chains share these cores)
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export PYTHONUNBUFFERED=1

R_PREFIX=$(conda env list | awk -v e="$R_ENV" '$1==e {print $NF}')
BLADE_PREFIX=$(conda env list | awk -v e="$BLADE_ENV" '$1==e {print $NF}')
[[ -n "$R_PREFIX" ]] && export PALM_RSCRIPT="$R_PREFIX/bin/Rscript"
[[ -n "$BLADE_PREFIX" ]] && export PALM_BLADE_PYTHON="$BLADE_PREFIX/bin/python"
echo "== $(date) host $(hostname), threads $THREADS, steps: $STEPS"
echo "   R: ${PALM_RSCRIPT:-missing}   BLADE: ${PALM_BLADE_PYTHON:-missing}"

set +u; conda activate "$PALM_ENV"; set -u
python -c "import palm_bulk, sys; print('   palm_bulk', palm_bulk.__version__, 'python', sys.version.split()[0])"

# ---- 1. pseudo-bulk tables + cell-level reference ----------------------------------------------
if [[ " $STEPS " == *" build "* ]]; then
  echo "== $(date) build"
  python scripts/build_pseudobulk_from_cells.py "$DATA" "$OUTDIR/$PREFIX" $SELECT $BUILD_EXTRA \
      --cells-per-type "$CELLS_PER_TYPE" --seed "$SEED"
fi

# ---- 2. comparison ------------------------------------------------------------------------------------
if [[ " $STEPS " == *" compare "* ]]; then
  echo "== $(date) compare"
  DEVICE=cpu; [[ "$USE_GPU" == 1 ]] && DEVICE=gpu
  ARGS=(--pseudobulk "$OUTDIR/${PREFIX}_broad.h5ad" --cells "$OUTDIR/${PREFIX}_cells.h5ad"
        --keep-types "$KEEP_TYPES" --label-map "$LABEL_MAP" --min-cells "$MIN_CELLS" --reps "$REPS"
        --contrast "$CONTRAST" --programs-csv "$PROGRAMS_CSV" --max-genes "$MAX_GENES"
        --methods "$METHODS" --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS"
        --precision "$PRECISION" --device "$DEVICE" --method-kw "$METHOD_KW"
        --out "$OUTDIR/compare_${PREFIX}.json" --workdir "$OUTDIR/work")
  [[ -n "$N_TEST" ]] && ARGS+=(--n-test "$N_TEST")
  [[ -n "$PROGRAM_LIST" ]] && ARGS+=(--program-list "$PROGRAM_LIST")
  python scripts/run_compare.py "${ARGS[@]}"
fi
echo "== $(date) done; results in $OUTDIR"
