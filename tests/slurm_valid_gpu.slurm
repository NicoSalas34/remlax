#!/bin/bash
# =============================================================================
# VALIDATION COMPLETE DE remlkit SUR GPU (tranche MIG A100)
#
# Tout tourne dans UNE SEULE image, ige_reml.sif, qui contient a la fois R
# (asreml, lme4, sommer, pbkrtest) et JAX CUDA. C'est ce qui rend ce job
# possible : remlkit fait appeler le solveur PAR R, et avec deux images il
# aurait fallu imbriquer apptainer dans apptainer — impossible ici, le binaire
# de l'hote ne trouvant pas ses bibliotheques dans le conteneur.
#
# CE QUI EST VERIFIE, dans cet ordre :
#   1. noyau Python            algebre, REML analytique, cas degeneres
#   2. validation croisee      meme -2logL que lme4 et sommer, SUR GPU
#   3. asreml (catalogue)      ar1, ar1xar1, splines 2D, str, vpredict, Wald
#   4. asreml (ajouts)         lvr, mtrn anisotrope, own, dsum, predict, K-R
#   5. modele IGE complet      le cas reel du projet
#   6. parite CPU/GPU          35 dispositifs, meme paquet, deux machines
#
# Le point 6 est le seul qui teste la MACHINE : les autres testent le code et
# se trouvent simplement executes sur la carte.
#
#   sbatch scripts/tests/slurm_valid_gpu.slurm
# =============================================================================
#SBATCH --job-name="remlkit_valid"
#SBATCH --output="logs/remlkit_valid_%j.out"
#SBATCH --error="logs/remlkit_valid_%j.out"
#SBATCH --partition=gpu-cirad-dedicated
#SBATCH --account=dedicated-gpu@cirad-long
#SBATCH --qos=dedicated
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs output/tests

SIF=${SIF:-/scratch/users/salasn/ige_reml.sif}

# renv DOIT ETRE DESARME dans le conteneur. Le projet a un .Rprofile qui charge
# renv, lequel repointe .libPaths() vers la bibliotheque de l'HOTE — construite
# pour une autre version de R. Le conteneur perd alors sa propre bibliotheque et
# s'arrete sur "there is no package called 'jsonlite'", alors que tout est la.
export APPTAINERENV_RENV_CONFIG_AUTOLOADER_ENABLED=FALSE
RUN() { apptainer exec --nv "$SIF" "$@"; }

echo "=== $(hostname) | job ${SLURM_JOB_ID:-local} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
RUN python3 -c "import jax; print('[jax]', jax.__version__, jax.devices())"

etat=0
etape() {                       # etape <numero> <titre> <commande...>
  local n="$1" titre="$2"; shift 2
  echo; echo "########## $n. $titre ##########"
  if "$@"; then echo "--> $n OK"; else echo "--> $n ECHEC (code $?)"; etat=1; fi
}

# 1-5 : les suites, telles quelles, mais avec le backend GPU.
etape 1 "noyau (Python)"            RUN python3 scripts/tests/test_remlkit_core.py
etape 2 "croisee lme4/sommer (GPU)" env APPTAINERENV_RK_BACKEND=gpu \
        apptainer exec --nv "$SIF" Rscript scripts/tests/test_remlkit.R
etape 3 "asreml : catalogue"        RUN Rscript scripts/tests/test_remlkit_asreml.R
etape 4 "asreml : ajouts 09/2026"   env APPTAINERENV_RK_BACKEND=gpu \
        apptainer exec --nv "$SIF" Rscript scripts/tests/test_remlkit_asreml2.R
etape 5 "modele IGE complet"        RUN Rscript scripts/tests/test_remlkit_ige.R

# 6 : parite. L'export est R, l'ajustement Python : les deux backends lisent
# STRICTEMENT le meme paquet, ce qui est tout l'interet du format sur fichier.
#
# COMPTER EN COMPILATIONS, PAS EN CALCULS. 34 modeles x 2 backends font 68
# compilations XLA, et sur une TRANCHE MIG (7 SM sur 108) le compilateur est
# lent au point que XLA emet lui-meme des "Very slow compile?". C'est le poste
# dominant du balayage, pas l'algebre : les modeles font 160 a 480 observations.
# Prevoir plusieurs dizaines de minutes ici, et ne pas confondre avec un blocage.
echo; echo "########## 6. parite CPU/GPU ##########"
RUN Rscript scripts/tests/export_bundles.R --out=output/tests/bundles || etat=1
RUN python3 scripts/tests/parite_gpu.py output/tests/bundles || etat=1

echo; echo "=== fin $(date) | etat global : $([ $etat -eq 0 ] && echo OK || echo ECHEC) ==="
exit $etat
