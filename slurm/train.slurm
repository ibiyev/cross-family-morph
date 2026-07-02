#!/usr/bin/bash --login
#SBATCH --job-name=byt5-morph
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

# GPU job: train ByT5 morpheme segmenter for one language.
# Uses pre-installed PyTorch module (CUDA 12.6) + venv overlay for transformers.
#
# Usage:
#   sbatch slurm/train.slurm aze 42 experiments/aze_seed42
#   sbatch slurm/train.slurm tur 42 experiments/tur_seed42
#   sbatch slurm/train.slurm hun 42 experiments/hun_seed42
#   sbatch slurm/train.slurm fin 42 experiments/fin_seed42
#
# Notes for V100 (16 GB, sm_70):
#   - ByT5 + fp16 = NaN, V100 does NOT support bf16 → use fp32 (--no_amp)
#   - batch_size=16, gradient_accumulation=4 → effective batch 64
#   - 12-hour wall is enough for ~20 epochs on ~250K pairs

set -euo pipefail

LANG=${1:-aze}
SEED=${2:-42}
OUT_DIR=${3:-experiments/${LANG}_seed${SEED}}

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

mkdir -p logs experiments

echo "================================================================"
echo " V100 — ByT5 morpheme segmentation training"
echo "================================================================"
echo " Language:   ${LANG}"
echo " Seed:       ${SEED}"
echo " Output dir: ${OUT_DIR}"
echo " Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo " Node:       $(hostname)"
echo " Started:    $(date)"
echo "================================================================"

module purge || print_error_and_exit "No 'module' command"
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0 \
    || print_error_and_exit "Failed to load PyTorch module"

# Avoid incompatible packages from ~/.local, especially torchvision
export PYTHONNOUSERSITE=1
export HF_HOME=/hf_cache
export TRANSFORMERS_CACHE=/hf_cache
mkdir -p /hf_cache

# Activate project venv (must be created beforehand — see SETUP.md)
VENV="${HOME}/environments/morph_venv"
[ -d "${VENV}" ] || print_error_and_exit "venv not found at ${VENV}; run setup_env.sh first"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# Sanity check
echo "Python: $(which python)"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# Build dataset if not present
DATA_TSV="data/morph_${LANG}.tsv"
if [ ! -f "${DATA_TSV}" ]; then
    echo "Building dataset for ${LANG}..."
    python scripts/build_dataset.py \
        --lang "${LANG}" \
        --cache_dir data/cache \
        --ud_root data/ud-treebanks \
        --sigmorphon_root data/sigmorphon2022 \
        --output "${DATA_TSV}"
fi

# Train (V100-optimized config)
python scripts/train.py \
    --data_path "${DATA_TSV}" \
    --output_dir "${OUT_DIR}" \
    --model_name google/byt5-small \
    --epochs 5 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --early_stopping_patience 5 \
    --seed "${SEED}" \
    --no_amp \
    --resume

echo "================================================================"
echo " Job finished at $(date)"
echo "================================================================"
