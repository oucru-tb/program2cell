# Settings for hpc/setup_envs.sh and hpc/run_benchmark.sh. Edit this file only; both scripts source it.
# Paths may be absolute or relative to the repository root.

# ---- conda ------------------------------------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"   # e.g. /nfs/team205/<you>/miniforge3
PALM_ENV=palm            # main env: palm_bulk, JAX, decoupler (PALM, naive, reallocation, scoring)
R_ENV=palm-r             # R 4.4 + BayesPrism, TOAST, TCA
BLADE_ENV=palm-blade     # Python 3.8 + BLADE_Deconvolution
USE_GPU=0                # 1: install CUDA JAX in the main env and run PALM on a GPU

# ---- data -------------------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA=/path/to/onek1k.h5ad                  # single-cell h5ad with raw counts (CELLxGENE OneK1K)
OUTDIR=$REPO/results/onek1k_50k            # everything produced by a run goes here
PREFIX=onek1k_50k                          # file prefix of the pseudo-bulk tables
PROGRAMS_CSV=$REPO/data/progeny_human_top500.csv   # written by setup_envs.sh

# ---- step 1: pseudo-bulk from cells (scripts/build_pseudobulk_from_cells.py) ---------------------
SELECT="--n-cells 50000"                   # or "--n-donors 200" or "--all"
BUILD_EXTRA=""                             # e.g. "--split-t" (CD4 T / CD8 T), "--min-score 0.5"
CELLS_PER_TYPE=200                         # cell-level reference kept per donor and type
SEED=0

# ---- step 2: comparison (scripts/run_compare.py) -------------------------------------------------
KEEP_TYPES="B cell,Monocyte,NK cell,T cell"      # with --split-t: "B cell,Monocyte,NK cell,CD4 T,CD8 T"
LABEL_MAP='{}'                                   # e.g. '{"DC": "Monocyte"}' to merge labels
MIN_CELLS=30                                     # cells per cell type for a test donor
N_TEST=""                                        # test donors per split (empty: half of eligible donors)
REPS=1                                           # donor splits (each a new random split)
CONTRAST=none                                    # OneK1K is healthy only; e.g. "disease,COVID-19,HC"
PROGRAM_LIST="JAK-STAT,NFkB,TNFa,TGFb,Hypoxia,p53,MAPK"   # empty: all 14 PROGENy programs
MAX_GENES=1000
METHODS="palm,naive_two_step,reallocation,bayesprism,toast,tca,blade"   # toast/tca need a contrast
WARMUP=500
SAMPLES=500
CHAINS=2
PRECISION=float64
METHOD_KW='{"blade": {"grid": "small", "n_rep": 1, "n_jobs": 8}, "bayesprism": {"n_cores": 8}}'

# ---- job resources (used by run_benchmark.sh for threads; scheduler headers are in that file) ----
THREADS="${LSB_DJOB_NUMPROC:-${SLURM_CPUS_PER_TASK:-8}}"
