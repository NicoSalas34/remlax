#!/bin/bash
# ==============================================================================
# 03_cube.sbatch — le cube d'un caractere en tableau de taches SLURM
# ------------------------------------------------------------------------------
# Dimensionnement : une cellule = 1 coeur, moins de 2 Go, 20 min en mediane sur
# CPU (n = 1 700 a 1 900, 12 parametres, maxiter 3000). Un fichier de taches de
# 22 500 cellules decoupe en CUBE_K = 750 tranches entrelacees donne 30 cellules
# par tache, soit 10 h en mediane ; la limite de temps ci-dessous laisse une
# marge. Une tache qui s'arrete avant la fin se relance a l'identique : les
# cellules deja ecrites sont sautees (reprise sur fit_summary.csv).
#
#   Rscript 03_cube.R --make_tasks --species Ble --trait Hauteur.4 --tasks tasks.csv
#   sbatch --array=1-750 --export=ALL,TASKS=tasks.csv,CUBE_K=750 03_cube.sbatch
#   Rscript 03_cube.R --aggregate --species Ble --trait Hauteur.4 --tasks tasks.csv
#
# A adapter au site : partition, module ou environnement R, RX_PY. Les
# variables REMLAX_IGE_DATA et REMLAX_CH3_OUT doivent etre exportees (ou posees
# ici) ; data.rds doit avoir ete ecrit par 00_data.R sous REMLAX_CH3_OUT.
# ==============================================================================
#SBATCH --job-name=cube_ch3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/cube_%A_%a.out
#SBATCH --error=logs/cube_%A_%a.err

set -euo pipefail
mkdir -p logs
: "${TASKS:?definir TASKS=<fichier de taches>}"
: "${CUBE_K:?definir CUBE_K=<nombre de tranches>}"
export CUBE_I="${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# export RX_PY=/chemin/vers/python   # interpreteur portant jax, si different du defaut
cd "$(dirname "$0")"
Rscript 03_cube.R --run --tasks "$TASKS" --hessian "${CUBE_HESSIAN:-TRUE}"
