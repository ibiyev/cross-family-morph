#!/usr/bin/bash --login
#SBATCH --job-name=byt5-xeval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

# Cross-language evaluation: run all 16 cells of the 4x4 transfer matrix.
# Assumes 4 monolingual models already trained at:
#   experiments/aze_seed42/best/
#   experiments/tur_seed42/best/
#   experiments/hun_seed42/best/
#   experiments/fin_seed42/best/
# And test sets at: data/test_<lang>.tsv

set -euo pipefail
mkdir -p logs results/matrix

module purge
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source "${HOME}/environments/morph_venv/bin/activate"

LANGS=(aze tur hun fin)
SEED=${1:-42}

echo "Running 4×4 transfer matrix (seed=${SEED})..."
for TRAIN in "${LANGS[@]}"; do
    MODEL_DIR="experiments/${TRAIN}_seed${SEED}/best"
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "Skipping ${TRAIN}: model not found at ${MODEL_DIR}"
        continue
    fi
    for TEST in "${LANGS[@]}"; do
        TEST_TSV="data/test_${TEST}.tsv"
        if [ ! -f "${TEST_TSV}" ]; then
            echo "Skipping ${TRAIN}->${TEST}: test set not found"
            continue
        fi
        OUT="results/matrix/${TRAIN}_to_${TEST}_seed${SEED}.json"
        if [ -f "${OUT}" ]; then
            echo "Already done: ${TRAIN}->${TEST}"
            continue
        fi
        echo "=== ${TRAIN} -> ${TEST} ==="
        python scripts/cross_eval.py \
            --model_dir "${MODEL_DIR}" \
            --test_tsv "${TEST_TSV}" \
            --label "${TRAIN}_to_${TEST}_seed${SEED}" \
            --output_json "${OUT}"
    done
done

# Aggregate results
python scripts/aggregate_matrix.py \
    --results_dir results/matrix \
    --seed "${SEED}" \
    --output results/transfer_matrix_seed${SEED}.json

echo "Transfer matrix complete."
