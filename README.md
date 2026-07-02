# Cross-Family Morpheme Segmentation for Agglutinative Languages

A ByT5-based study of morpheme segmentation transfer between Turkic
(Azerbaijani, Turkish) and Uralic (Hungarian, Finnish) language families,
designed to run on the NVIDIA Tesla V100 GPUs.

## Project structure

```
cross-family-morph/
├── scripts/                   # Python pipeline
│   ├── build_dataset.py       # UniMorph + UD + SIGMORPHON → segmentation pairs
│   ├── train.py               # ByT5 training (V100-aware, fp32 default)
│   ├── cross_eval.py          # Evaluate one model on one test set
│   ├── aggregate_matrix.py    # Build 4×4 transfer matrix from per-cell JSONs
│   └── distance_correlation.py# Linguistic distance vs transfer correlation
├── baselines/
│   ├── rule_based.py          # Greedy suffix-stripping
│   └── morfessor_baseline.py  # Unsupervised Morfessor 2.0
├── utils/
│   └── morphology.py          # Suffix inventories, vowel harmony, distance metrics
├── slurm/
│   ├── setup_env.sh           # One-time venv setup on a compute node
│   ├── train.slurm            # Training job submission
│   └── cross_eval.slurm       # 4×4 matrix evaluation
└── data/                      # Created at first run
```

## Setup (one-time)

```bash
# 1. Clone this project
git clone <your-repo-url> ~/cross-family-morph
cd ~/cross-family-morph

# 3. Get an interactive GPU session and create the venv
si-gpu -c 7 -t 1:00:00
bash slurm/setup_env.sh
exit
```

This creates a venv at `~/environments/morph_venv`
PyTorch module and adds `transformers`, `accelerate`, `lang2vec`, etc.

## Build datasets

```bash
# Download UD treebanks (one-time; do this on a login node — no GPU needed)
mkdir -p data/ud-treebanks
cd data/ud-treebanks
git clone --depth 1 https://github.com/UniversalDependencies/UD_Azerbaijani-TueCL.git
git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-IMST.git
git clone --depth 1 https://github.com/UniversalDependencies/UD_Turkish-BOUN.git
git clone --depth 1 https://github.com/UniversalDependencies/UD_Hungarian-Szeged.git
git clone --depth 1 https://github.com/UniversalDependencies/UD_Finnish-TDT.git
cd ../..

# Build per-language segmentation TSVs
for lang in aze tur hun fin; do
  python scripts/build_dataset.py \
    --lang $lang \
    --cache_dir data/cache \
    --ud_root data/ud-treebanks \
    --output data/morph_${lang}.tsv \
    --preview
done
```

UniMorph data is downloaded automatically on first run.

## Train models

```bash
# Submit one training job per (language, seed) combination
for lang in aze tur hun fin; do
  for seed in 42 123 456; do
    sbatch slurm/train.slurm $lang $seed experiments/${lang}_seed${seed}
  done
done
```

Each job uses 1× V100 (16 GB) for up to 12 hours. Default config:
- `batch_size=16`, `gradient_accumulation_steps=4` (effective batch 64)
- `fp32` (V100 cannot do bf16 and ByT5 + fp16 = NaN)
- Early stopping with patience=5

## Run the cross-family transfer matrix

After all 4 monolingual models finish:

```bash
sbatch slurm/cross_eval.slurm 42
```

This runs all 16 cells (4 train langs × 4 test langs), aggregates them into
a single matrix, and writes `results/transfer_matrix_seed42.json`.

## Compute typological distance correlations

```bash
python scripts/distance_correlation.py \
  --matrix_json results/transfer_matrix_seed42.json \
  --output_json results/correlations_seed42.json \
  --output_csv results/correlations_seed42.csv
```

This is the analysis section that explains *why* the transfer matrix looks
the way it does.

## Run baselines

```bash
for lang in aze tur hun fin; do
  python baselines/rule_based.py \
    --test_tsv data/test_${lang}.tsv \
    --lang $lang \
    --output_json results/baselines/rule_${lang}.json
  python baselines/morfessor_baseline.py \
    --train_tsv data/train_${lang}.tsv \
    --test_tsv data/test_${lang}.tsv \
    --label "morfessor_${lang}" \
    --output_json results/baselines/morfessor_${lang}.json
done
```

## V100 specifics — why fp32?

ByT5 processes raw bytes, producing sequences ~4× longer than subword models.
Internal activations frequently exceed the fp16 representable range
(±65,504), causing NaN on the very first batch.

| Precision | V100 (sm_70) | A100 (sm_80) |
|-----------|--------------|---------------|
| fp16      | NaN          | NaN           |
| bf16      | not supported| optimal       |
| **fp32**  | **stable**   | stable (slow) |

Our scripts auto-detect V100 and force fp32 even if `--fp16` or `--bf16` is set.

## References

- Xue et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *TACL* 10:291–306.
- Batsuren et al. (2022). The SIGMORPHON 2022 Shared Task on Morpheme Segmentation. *SIGMORPHON Workshop*.
- Batsuren et al. (2022). UniMorph 4.0: Universal Morphology. *LREC 2022*.
- Nivre et al. (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. *LREC 2020*.
- Littell et al. (2017). URIEL and lang2vec. *EACL 2017*.
- Virpioja et al. (2013). Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline. Aalto University.
