#!/usr/bin/env bash
# One-time setup of the three conda environments and the PROGENy program weights.
# Run on a node with internet access (usually a login node), from anywhere:
#     bash hpc/setup_envs.sh            # all three envs
#     bash hpc/setup_envs.sh palm r     # only some: palm | r | blade
# Safe to re-run: existing envs are updated, not recreated. The R step compiles some packages from
# source and can take 30-60 minutes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
source "$CONDA_BASE/etc/profile.d/conda.sh"
MAMBA=$(command -v mamba || echo conda)
STEPS="${*:-palm r blade}"
LOG="$REPO/hpc/setup_envs.log"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) setup: $STEPS (conda base $CONDA_BASE, repo $REPO)"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# ---- main env: palm_bulk + JAX + decoupler ---------------------------------------------------------
if [[ " $STEPS " == *" palm "* ]]; then
  env_exists "$PALM_ENV" || $MAMBA create -y -n "$PALM_ENV" -c conda-forge python=3.12 pip h5py
  set +u; conda activate "$PALM_ENV"; set -u
  pip install --upgrade pip
  pip install -e "$REPO[sc,progeny,dev]"
  if [[ "$USE_GPU" == 1 ]]; then pip install "jax[cuda12]"; fi
  mkdir -p "$(dirname "$PROGRAMS_CSV")"
  python - "$PROGRAMS_CSV" <<'EOF'
import sys, decoupler as dc
net = dc.op.progeny(organism="human", top=500)
net.rename(columns={"source": "program", "target": "gene"})[["program", "gene", "weight"]].to_csv(sys.argv[1], index=False)
print("PROGENy:", net.source.nunique(), "programs ->", sys.argv[1])
EOF
  python -m pytest -q -m "not slow" "$REPO/tests" -p no:cacheprovider
  set +u; conda deactivate; set -u
fi

# ---- R env: InstaPrism, TOAST, TCA ---------------------------------------------------------------
if [[ " $STEPS " == *" r "* ]]; then
  env_exists "$R_ENV" || $MAMBA create -y -n "$R_ENV" -c conda-forge \
      r-base=4.4 r-matrix r-remotes r-devtools r-biocmanager compilers make pkg-config \
      libcurl openssl libxml2 zlib glpk gmp
  set +u; conda activate "$R_ENV"; set -u
  Rscript - <<'EOF'
options(repos = BiocManager::repositories(), Ncpus = max(1L, parallel::detectCores() - 1L))
need <- function(p) !requireNamespace(p, quietly = TRUE)
bioc <- c("GenomeInfoDbData", "TOAST")
if (any(sapply(bioc, need))) BiocManager::install(bioc[sapply(bioc, need)], update = FALSE, ask = FALSE)
cran <- c("TCA", "NMF", "caret", "pheatmap", "abind", "pbapply", "RcppArmadillo")
if (any(sapply(cran, need))) install.packages(cran[sapply(cran, need)])
if (need("InstaPrism"))   # compiles C++ (RcppArmadillo); the conda compilers suffice on Linux, no Rtools
  remotes::install_github("humengying0907/InstaPrism", upgrade = "never", dependencies = TRUE)
ok <- sapply(c("TOAST", "TCA", "InstaPrism"), function(p) requireNamespace(p, quietly = TRUE))
print(ok)
if (!all(ok)) quit(status = 1)
EOF
  set +u; conda deactivate; set -u
fi

# ---- BLADE env -------------------------------------------------------------------------------------
if [[ " $STEPS " == *" blade "* ]]; then
  env_exists "$BLADE_ENV" || $MAMBA create -y -n "$BLADE_ENV" -c conda-forge \
      python=3.8 pip numpy scipy scikit-learn numba joblib
  set +u; conda activate "$BLADE_ENV"; set -u
  pip install BLADE_Deconvolution==0.0.7          # run_blade.py patches this version's import bug
  python -c "import BLADE_Deconvolution.BLADE_framework; print('BLADE ok')"
  set +u; conda deactivate; set -u
fi

echo "== $(date) setup done"
