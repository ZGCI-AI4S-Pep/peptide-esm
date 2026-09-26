# Peptide ESM Pretraining Tutorial

For **ICLR 2026 PepBenchmark: _A Standardized Benchmark for Peptide Machine Learning_**

This repository provides a **peptide-focused masked language modeling (MLM) pretraining pipeline** used in **PepBenchmark**. Starting from [`facebook/esm2_t30_150M_UR50D`](https://huggingface.co/facebook/esm2_t30_150M_UR50D), it prepares short peptide corpora with sequence length **≤ 50** and fine-tunes the model using **Hugging Face Trainer**, **Accelerate**, and **DeepSpeed**.

The project is designed as a **small, reproducible tutorial** that walks through the full workflow:

- prepare or inspect the pretraining corpus
- split the corpus into training and validation sets
- run a single-device smoke test
- launch the full 8-GPU training configuration
- regenerate dataset statistics and figures from the provided notebook

---

## Repository Structure

```text
.
├── assets/                   # Generated dataset statistics and figures
├── configs/config-1/         # Quickstart and full training configs
├── pretrain_data/            # Raw peptide / short-protein corpora
├── utils/                    # Helper functions
├── dataset_analyse.ipynb     # Dataset summary notebook
├── environment.yml           # Reproducible conda environment
├── preprocessing.py          # Train/validation split utility
└── run_mlm.py                # MLM training entry point
```
official checkpoint: https://huggingface.co/jiahuizhang/esm-150m-peptide-fine-tune
## 1. Environment Setup

Create the conda environment from the pinned specification:

```bash
conda env create -f environment.yml
conda activate pepllm
```

Optional: install PEFT only if you want to enable LoRA in `run_mlm.py`.

```bash
pip install peft
```

All commands below assume that you run them from the repository root.

## 2. Pretraining Data


Please download the pretraining data from the Hugging Face dataset repository: https://huggingface.co/datasets/jiahuizhang/pepbenchmark_pretrain_data
After downloading, rename the folder to `pretrain_data/` and place it under the project root.

All sequences are expected to use the 20 canonical amino acids and have length between `1` and `50`. A dataset card with statistics and figures is available in [pretrain_data/README.md](/home/dataset-assist-0/jiahui/pepbenchmark/final/generation/lm-based/esm-peptide/peptide-esm/pretrain_data/README.md).

The default dataset is `uniref50`, which is also the dataset referenced our paper.

## 3. Prepare Train/Validation Splits

Use `preprocessing.py` to filter sequences, normalize them to uppercase, and create a deterministic train/validation split.

Example: prepare the recommended `uniref50` split.

```bash
python preprocessing.py \
  --input_file ./pretrain_data/uniref50.txt \
  --output_dir ./processed_data/uniref50 \
  --validation_split 0.1 \
  --seed 42 \
  --min_length 1 \
  --max_length 50
```

Expected outputs:

- `./processed_data/uniref50/train.txt`
- `./processed_data/uniref50/validation.txt`

Quick sanity check:

```bash
wc -l ./processed_data/uniref50/train.txt ./processed_data/uniref50/validation.txt
```


## 4. Training

### Quick Smoke Test

Use the lightweight config below to verify the pipeline before launching the full run:

```bash
TENSORBOARD_LOGGING_DIR=./logs/tutorial_quickstart \
python run_mlm.py ./configs/config-1/config.quickstart.json
```

This quickstart configuration:

- uses the processed `uniref50` split
- disables DeepSpeed
- limits the number of samples
- writes outputs to `checkpoint/tutorial_quickstart`

### Full 8-GPU Run

The main experiment uses DeepSpeed ZeRO-2 with `Accelerate`:

```bash
TENSORBOARD_LOGGING_DIR=./logs/uniref50_esm2_150m \
OMP_NUM_THREADS=16 accelerate launch \
  --config_file ./configs/config-1/acc_config.yaml \
  run_mlm.py ./configs/config-1/config.json
```

The default full config:

- model: `facebook/esm2_t30_150M_UR50D`
- dataset: `processed_data/uniref50`
- precision: `bf16`
- per-device train batch size: `512`
- per-device eval batch size: `512`
- warmup steps: `2000`
- epochs: `500`
- scheduler: `linear`
- optimizer: `adamw_torch`

Outputs are written to:

- `checkpoint/uniref50_esm2_150m`
- `logs/uniref50_esm2_150m` when `TENSORBOARD_LOGGING_DIR` is set as above
