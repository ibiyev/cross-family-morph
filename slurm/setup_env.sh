#!/usr/bin/bash --login
#
# Usage:
#   srun -p gpu --gpus=1 -c 7 -t 1:00:00 --pty bash setup_env.sh
#
# Or inside an interactive job (si-gpu):
#   bash setup_env.sh

set -euo pipefail

VENV="${HOME}/environments/morph_venv"

echo "Setting up venv at ${VENV}..."

module purge
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Create venv with system-site-packages
python -m venv --system-site-packages "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# Install only what is missing — DO NOT install torch (use the module's)
pip install --upgrade pip
pip install \
    'transformers>=4.40,<5' \
    'tokenizers>=0.15' \
    'accelerate>=0.30' \
    'datasets>=2.18' \
    'sentencepiece' \
    'protobuf<5' \
    'tqdm>=4.65' \
    'numpy<2' \
    'scikit-learn>=1.3' \
    'pandas>=2.0' \
    'pyyaml' \
    'tensorboard' \
    'lang2vec'   # for typological distance metrics

# Verify
echo ""
echo "===== verification ====="
python -c "
import torch, transformers, lang2vec
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'Transformers: {transformers.__version__}')
print(f'lang2vec:     OK')
"
echo "Setup complete. Activate with:"
echo "  source ${VENV}/bin/activate"
