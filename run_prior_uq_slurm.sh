#!/usr/bin/env bash
set -euo pipefail

# SLURM array runner for Study B UQ (prior regularization on/off).
#
# Submits an array job where each task runs a disjoint shard of replicates and
# writes `results/prior_regularization_uq_..._shard{i}of{K}.npz`.
#
# Usage:
#   sbatch run_prior_uq_slurm.sh
#
# After it finishes, merge shards on a login node:
#   bash run_prior_uq_slurm.sh merge
#

MODE="${1:-run}"

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

ENV_NAME="recursivemixtures"     # conda env name on the cluster
OUT_DIR="results/prior_uq_$(date +%Y%m%d_%H%M%S)"

# Total replicates and sharding
NITER=1000
NUM_SHARDS=20                    # number of array tasks

# Experiment settings
N_DATA=1000
STORE_EVERY=10
REF_SIZE=2048
SEED=123
FULL=""                          # set to "--full" for slower/full config
NO_PLOT="--no-plot"              # omit PDFs on HPC

# SLURM resources
JOB_NAME="hk_prior_uq"
TIME="08:00:00"
CPUS_PER_TASK=1
MEM="8G"

# -----------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${MODE}" == "merge" ]]; then
  # Merge all shards in OUT_DIR (defaults to newest matching directory).
  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "ERROR: OUT_DIR does not exist: ${OUT_DIR}" >&2
    echo "Edit OUT_DIR at top of script, or set it before calling merge:" >&2
    echo "  OUT_DIR=results/prior_uq_... bash run_prior_uq_slurm.sh merge" >&2
    exit 1
  fi
  SHARDS_CSV="$(ls "${OUT_DIR}"/prior_regularization_uq_*_shard*of${NUM_SHARDS}.npz 2>/dev/null | paste -sd, - || true)"
  if [[ -z "${SHARDS_CSV}" ]]; then
    echo "ERROR: no shard .npz files found under ${OUT_DIR}" >&2
    exit 1
  fi
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/hk_prior_regularization_uq.py" \
    --merge-shards "${SHARDS_CSV}" \
    --merge-out "${OUT_DIR}/prior_regularization_uq_merged.npz"
  echo "Merged to ${OUT_DIR}/prior_regularization_uq_merged.npz"
  exit 0
fi

# If running inside SLURM, execute the shard. Otherwise, submit the array job.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OUT_DIR}"
  echo "Running shard ${SLURM_ARRAY_TASK_ID:-0}/${NUM_SHARDS} -> ${OUT_DIR}"
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/hk_prior_regularization_uq.py" \
    --niter "${NITER}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --n-data "${N_DATA}" \
    --store-every "${STORE_EVERY}" \
    --ref-size "${REF_SIZE}" \
    --seed "${SEED}" \
    ${FULL} \
    --out-dir "${OUT_DIR}" \
    ${NO_PLOT}
  exit 0
fi

mkdir -p "${OUT_DIR}"
echo "Submitting array job: 0-$((${NUM_SHARDS}-1))"
echo "Output directory: ${OUT_DIR}"

sbatch \
  --job-name="${JOB_NAME}" \
  --time="${TIME}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --array="0-$((${NUM_SHARDS}-1))" \
  --output="${OUT_DIR}/slurm_%A_%a.out" \
  --error="${OUT_DIR}/slurm_%A_%a.err" \
  --export=ALL,OUT_DIR="${OUT_DIR}",ENV_NAME="${ENV_NAME}",NITER="${NITER}",NUM_SHARDS="${NUM_SHARDS}",N_DATA="${N_DATA}",STORE_EVERY="${STORE_EVERY}",REF_SIZE="${REF_SIZE}",SEED="${SEED}",FULL="${FULL}",NO_PLOT="${NO_PLOT}" \
  "${BASH_SOURCE[0]}"

