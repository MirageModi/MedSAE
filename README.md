# MedSAE: Sparse Autoencoder Analysis Pipeline for Medical Language Models

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

**MedSAE** is an end-to-end pipeline for training top-k Sparse Autoencoders (SAEs) on hidden activations from medical LLMs (e.g., MedGemma-27B-Text-IT, OpenBioLLM-Llama3-70B) and analyzing how polysemous medical terms (e.g., *discharge*, *failure*, *block*) are represented in latent space.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
  - [1. Activation Generation](#1-activation-generation)
  - [2. SAE Training](#2-sae-training)
  - [3. Polysemy Analysis](#3-polysemy-analysis)
- [Analysis Tools](#analysis-tools)
- [Oncology Robustness Experiments](#oncology-robustness-experiments)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Citation](#citation)

---

## Overview

Medical terminology is inherently polysemous—the same word can carry vastly different meanings depending on clinical context:

| Term | Sense A | Sense B |
|------|---------|---------|
| **discharge** | Administrative (hospital discharge) | Clinical (bodily secretion) |
| **failure** | Cardiac (heart failure) | Renal (kidney failure) |
| **block** | Cardiac (heart block) | Neurological (nerve block) |
| **mass** | Anatomical (tissue mass) | Radiological (mass on imaging) |

MedSAE trains sparse autoencoders on LLM hidden states to discover interpretable latent features that can distinguish between these senses, enabling:

- **Mechanistic interpretability** of how medical LLMs encode polysemy
- **Sense disambiguation** using learned latent features
- **Retrieval improvement** through sense-aware re-ranking

---

## Key Features

- 🔬 **Multi-GPU distributed training** with FSDP support for large-scale SAE training
- 🏥 **Medical NER integration** using GLiNER-BioMed for clinical entity extraction
- 📊 **Polysemy metrics** including Jaccard similarity, entropy, and sense-linked latent identification
- 🔄 **Character-level alignment** between NER spans and LLM token positions
- 📈 **Probe-based evaluation** with AUROC/AP filtering for latent quality assessment
- 🧪 **Mixed-effects modeling** for statistical analysis (R integration)
- 🧬 **Oncology robustness testing** with medical concept ablation studies across multiple LLMs

---

## Project Structure

```
MedSAE/
├── SAE_pipeline/                    # Core pipeline scripts
│   ├── sae_pipeline.sh              # Master orchestration script
│   ├── gen_LLMActivations.py        # Distributed activation extraction
│   ├── sae_trainer.py               # SAE training with FSDP
│   ├── calibrate_sae_feature_scales.py  # Post-hoc feature calibration
│   ├── ner_infer_gliner_fast.py     # GLiNER NER inference
│   ├── align_spans_to_tokens.py     # NER-to-token alignment
│   ├── build_polysemy_lexicon.py    # Polysemy seed lexicon builder
│   ├── make_labels.py               # Token-level label generation
│   ├── sae_viewer_topk_sharded_chars.py  # Top-K latent export & probing
│   ├── compute_polysemy_metrics.py  # Jaccard/entropy/sense-linked metrics
│   ├── run_ablation_delta_ce.py     # Ablation cross-entropy analysis
│   └── run_mixed_effects.R          # Mixed-effects statistical models
│
├── Oncology/                         # LLM robustness testing experiments
│   ├── LLM_generate.py             # Medical concept removal experiments
│   └── gpt5_thinking_model.py      # GPT-5 wrapper using OpenAI Responses API
│
└── analysis/                        # Post-hoc analysis scripts
    ├── jaccard_10.py                # Top-10 Jaccard similarity analysis
    ├── jaccard_control.py           # Monosemantic baseline comparisons
    ├── reranking_retrieval.py       # Sense-aware retrieval experiments
    ├── run_jacc10.sh                # Jaccard analysis runner
    └── control_jaccard.sh           # Control experiment runner
```

**Note:** A SLURM batch script example (`run_removals_optimized.sh`) is documented in the usage section but is not included in the repository.

---

## Requirements

### Python Dependencies

```
torch>=2.0
transformers>=4.35
accelerate>=0.24
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
tqdm
gliner                  # For biomedical NER
bitsandbytes            # Optional: 8-bit Adam optimizer
```

### R Dependencies (for mixed-effects modeling)

```r
install.packages(c("lme4", "lmerTest", "broom.mixed", "dplyr"))
```

### Additional Dependencies (for Oncology module)

```bash
# OpenAI Python client (for GPT-5 support)
pip install openai
```

**Note:** GPT-5 access requires an OpenAI API key. Set `OPENAI_API_KEY` environment variable or pass `--openai_api_key` when running experiments.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 24GB (single) | 8×40GB (A100) |
| System RAM | 64GB | 256GB+ |
| Storage | 500GB | 2TB+ SSD |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/MedSAE.git
cd MedSAE

# Create conda environment
conda create -n medsae python=3.10
conda activate medsae

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate pandas numpy scipy scikit-learn tqdm gliner

# Optional: 8-bit optimizer
pip install bitsandbytes
```

---

## Pipeline Stages

The complete pipeline is orchestrated via `sae_pipeline.sh`. Below are the individual stages:

### 1. Activation Generation

Extract hidden-state activations from a target layer of a medical LLM.

```bash
torchrun --nproc-per-node=4 gen_LLMActivations.py \
    --csv_path /path/to/mimic_notes.csv \
    --text_column input_with_target \
    --model_id google/medgemma-27b-text-it \
    --layer_index 52 \
    --ctx_len 64 \
    --stride 64 \
    --batch_texts 8 \
    --normalize \
    --output_dir /path/to/activations/ \
    --val_percent 0.05 \
    --max_rows 50000 \
    --shuffle --shuffle_seed 42 \
    --save_char_offsets \
    --save_doc_id \
    --doc_id_column note_id
```

**Key Options:**
| Flag | Description |
|------|-------------|
| `--layer_index` | Target decoder layer (e.g., 52 for MedGemma-27B) |
| `--normalize` | Apply per-token L2 normalization |
| `--save_char_offsets` | Save character offsets for NER alignment |
| `--gpus_per_rank` | GPUs per process for tensor parallelism |

**Output Structure:**
```
activations/
├── shard_00/
│   ├── train/batch_000000.pt, ...
│   ├── val/batch_000000.pt, ...
│   ├── metadata.json
│   └── indices_train.npy
├── shard_01/
│   └── ...
```

---

### 2. SAE Training

Train a TopK Sparse Autoencoder on the extracted activations.

```bash
torchrun --nproc-per-node=8 sae_trainer.py \
    --activations_dir /path/to/activations/ \
    --n_latents 65536 \
    --d_model 5376 \
    --k 128 \
    --k_aux 512 \
    --dead_threshold_steps 900 \
    --relu_after_topk \
    --optimizer adafactor \
    --lr 2e-4 \
    --train_steps 200000 \
    --micro_bsz 128 \
    --sae_fsdp \
    --sae_fsdp_devices 0,1,2,3,4,5,6,7 \
    --save_dir /path/to/sae/
```

**SAE Architecture:**
- **Encoder**: `z = TopK(ReLU(W_enc @ x + b_enc))`
- **Decoder**: `x_hat = W_dec @ z + b_dec`
- **Auxiliary Loss**: Residual reconstruction on dead latents

**Key Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_latents` | 65536 | Number of latent features |
| `k` | 128 | TopK sparsity level |
| `k_aux` | 512 | Auxiliary loss TopK |
| `dead_threshold_steps` | 900 | Steps before reinitializing dead latents |

---

### 3. Polysemy Analysis

Run the complete polysemy analysis pipeline:

```bash
# Step 3a: Run NER on clinical notes
python ner_infer_gliner_fast.py \
    --csv_path /path/to/mimic_notes.csv \
    --text_column input_with_target \
    --out_jsonl polysemy_analysis/ner_spans.jsonl \
    --batch_size 1024 \
    --device 0 \
    --min_score 0.8

# Step 3b: Align NER spans to LM tokens
python align_spans_to_tokens.py \
    --ner_jsonl polysemy_analysis/ner_spans.jsonl \
    --csv_path /path/to/mimic_notes.csv \
    --text_column input_with_target \
    --tokenizer_name google/medgemma-27b-text-it \
    --output polysemy_analysis/ner_spans_with_tokens.jsonl

# Step 3c: Build polysemy lexicon
python build_polysemy_lexicon.py \
    --output polysemy_analysis/polysemy_lexicon.csv \
    --seed_only

# Step 3d: Export Top-K latents per token
python sae_viewer_topk_sharded_chars.py \
    --ckpt_path /path/to/sae/best_calibrated.pt \
    --model_id google/medgemma-27b-text-it \
    --acts_root /path/to/activations/ \
    --acts_split val \
    --export_topk_latents_csv polysemy_analysis/topk_latents.csv \
    --run_probes \
    --run_delta_ce

# Step 3e: Compute polysemy metrics
python compute_polysemy_metrics.py \
    --topk_csv polysemy_analysis/topk_latents.csv \
    --lexicon_csv polysemy_analysis/polysemy_lexicon.csv \
    --ner_jsonl polysemy_analysis/ner_spans.jsonl \
    --output_dir polysemy_analysis/metrics \
    --fdr_threshold 0.05
```

---

## Analysis Tools

### Jaccard@10 Similarity Analysis

Compare latent feature overlap between different senses of polysemous terms:

```bash
python analysis/jaccard_10.py \
    --topk_csv polysemy_analysis/topk_latents.csv \
    --ner_jsonl polysemy_analysis/ner_spans.jsonl \
    --lexicon_csv polysemy_analysis/polysemy_lexicon.csv \
    --output_csv results/jaccard_comparison.csv
```

**Interpretation:**
- **High Jaccard (>0.6)**: Senses share core latent features (monosemantic baseline)
- **Low Jaccard (<0.3)**: Senses activate distinct latent features (polysemy detected)

### Sense-Aware Retrieval Re-ranking

Improve retrieval accuracy by re-ranking candidates using semantic latent features:

```bash
python analysis/reranking_retrieval.py \
    --topk_csv polysemy_analysis/topk_latents.csv \
    --ner_jsonl polysemy_analysis/ner_spans.jsonl \
    --lexicon_csv polysemy_analysis/polysemy_lexicon.csv \
    --output_dir results/retrieval/
```

---

## Oncology Robustness Experiments

The `Oncology/` module contains an experimental framework for testing LLM robustness in clinical decision-making. This module performs ablation studies by systematically removing medical concepts from clinical notes and evaluating how different models respond to incomplete information.

### Purpose and Methodology

This module tests model robustness through controlled ablation experiments:

- **Medical Concept Removal**: Systematically removes medical terms from clinical notes, replacing them with `[MISSING INFO]` placeholders
- **Multi-Model Comparison**: Evaluates responses across OpenBioLLM, MedGemma, and GPT-5 to assess robustness to missing information
- **Clinical Decision-Making**: Focuses on oncology-specific cases requiring staging, treatment planning, and management recommendations
- **Self-Consistency**: Uses majority voting across multiple generations (n=5) for HuggingFace models to improve reliability

### Usage

#### Direct Python Execution (Primary Method)

For HuggingFace models (OpenBioLLM, MedGemma):

```bash
python Oncology/LLM_generate.py \
    --model=medgemma \
    --prompt=NSCLC-initiateTreatment \
    --batch_size=1
```

For GPT-5 models:

```bash
python Oncology/LLM_generate.py \
    --model=gpt-5 \
    --prompt=UC \
    --reasoning_effort=medium \
    --verbosity=high \
    --max_output_tokens=16384 \
    --openai_api_key=$OPENAI_API_KEY
```

**Model-specific parameters:**

| Parameter | Models | Options | Description |
|-----------|--------|---------|-------------|
| `TASK_TAG` | All | directedTBD/fullAPTBD/fullAP_TBD/CoT | Task format (set in code, default: "directedTBD") |
| `--reasoning_effort` | GPT-5 only | minimal/low/medium/high | Controls reasoning depth for GPT-5 thinking model |
| `--verbosity` | GPT-5 only | low/medium/high | Controls output verbosity level |
| `--max_output_tokens` | GPT-5 only | integer (default: 16384) | Maximum output tokens for GPT-5 |
| `--batch_size` | OpenBioLLM, MedGemma | integer (default: 1) | Batch size for HuggingFace models (ignored for GPT-5) |

**Supported prompt types:**

- `UC`: Urethral Carcinoma case
- `NSCLC-initiateTreatment`: Non-small cell lung cancer treatment initiation
- `NSCLC-RoutineFollowup`: Non-small cell lung cancer routine follow-up

### Output Structure

Results are saved in `data/{PROMPT}/{MODEL}-{TASK_TAG}/`:

```
data/
└── {PROMPT}/
    └── {MODEL}-{TASK_TAG}/
        ├── routine_{PROMPT}_{MODEL}_{description}_2048tokens.txt  # Individual responses per removal
        ├── routine_{PROMPT}_{MODEL}_all_umls_removals_summary.txt  # Summary of all responses
        └── routine_{PROMPT}_{MODEL}_umls_removals_results.json     # JSON results with metadata
```

Each individual response file contains:
- Terms removed for that experiment
- Description of the removal combination
- Self-consistency parameters (for HuggingFace models) or GPT-5 config (for GPT-5)
- Full model response

The JSON results file includes:
- Complete list of Medical concepts tested
- All responses with metadata
- Model configuration (self-consistency params or GPT-5 reasoning settings)

---

## Configuration

All pipeline parameters can be configured in `sae_pipeline.sh`:

```bash
# === Dataset Configuration ===
CSV_PATH="/path/to/MIMIC/dataset.csv"
TEXT_COLUMN="input_with_target"
MAX_ROWS=50000

# === Model Configuration ===
MODEL_ID="google/medgemma-27b-text-it"
LAYER_INDEX=52
DTYPE="bfloat16"

# === SAE Architecture ===
N_LATENTS=65536
D_MODEL=5376
K=128
K_AUX=512

# === Training ===
LR=2e-4
TRAIN_STEPS=200000
OPTIMIZER="adafactor"

# === Pipeline Control ===
SKIP_ACTIVATION_GENERATION=false
SKIP_SAE_TRAINING=false
RUN_POLYSEMY_ANALYSIS=true
```

---

## Outputs

### SAE Checkpoints

```
sae/
├── best.pt              # Best validation NMSE checkpoint
├── latest.pt            # Most recent checkpoint
├── cfg.json             # Training configuration
└── best_meta.json       # Checkpoint metadata
```

### Polysemy Analysis Results

```
polysemy_analysis/
├── ner_spans.jsonl              # Raw NER predictions
├── ner_spans_with_tokens.jsonl  # Token-aligned NER
├── polysemy_lexicon.csv         # Seed terms with senses
├── topk_latents.csv             # Per-token top-K latent activations
├── viewer/
│   ├── viewer_pack.json         # Feature viewer data
│   ├── top_tokens_per_latent.csv
│   └── probes/
│       └── single_latent_probes.csv  # Probe AUROC/AP scores
└── metrics/
    ├── jaccard_cross_sense.csv  # Cross-sense Jaccard similarities
    ├── entropy_per_sense.csv    # Latent entropy per sense
    ├── sense_linked_latents.csv # Statistically significant sense-linked features
    └── ablation_delta_ce.csv    # Ablation ΔCE results
```

---

## Sense-to-NER Mapping

The pipeline uses GLiNER-BioMed entity types to disambiguate word senses:

| Sense | Expected NER Labels |
|-------|---------------------|
| `administrative` | administrative healthcare event, medical procedure |
| `clinical` | bodily fluid or secretion, clinical finding |
| `cardiac` | cardiovascular disorder |
| `respiratory` | respiratory or pulmonary disorder |
| `neurological` | neurological or mental disorder |
| `radiological` | radiological finding or shadow |
| `anatomical` | anatomical structure |
| `renal` | renal or kidney disorder |

---

## Example Results

### Jaccard Similarity: Monosemantic vs. Polysemous

```
--- Monosemantic Baseline (Top-10) ---
hypertension: 58.2% (n=1523)
patient:      61.4% (n=8421)
treatment:    55.7% (n=2891)

--- Polysemous Cross-Sense (Top-10) ---
discharge (administrative vs clinical): 23.1% (n=412,89)
failure (cardiac vs renal):             28.4% (n=1203,456)
block (cardiac vs neurological):        19.6% (n=234,67)
```

### Retrieval P@1 Improvement

| Method | P@1 | ΔP@1 |
|--------|-----|------|
| Full latent similarity | 42.3% | — |
| Semantic re-ranking (k=20) | 51.8% | +9.5%** |

---

## Citation

If you use MedSAE in your research, please cite:

```bibtex
@article{medsae2026,
  title={Understanding Clinical Reasoning Variability in Medical Large Language Models: A Mechanistic Interpretability Study},
  author={Modi, Mirage and Krull, Jordan E. and Johnson, Donte and Wang, Xiaoying and Gauntner, Timothy D. and Li, Mingjia and Cheng, Hao and Ma, Anjun and Zhang, Ping and Stover, Daniel G. and Li, Zihai and Ma, Qin},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.01.26.26344845},
  url={https://www.medrxiv.org/content/10.64898/2026.01.26.26344845v2}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [GLiNER-BioMed](https://huggingface.co/Ihor/gliner-biomed-large-v1.0) for biomedical NER
- [MedGemma](https://huggingface.co/google/medgemma-27b-text-it) and [OpenBioLLM](https://huggingface.co/aaditya/OpenBioLLM-Llama3-70B) for medical LLMs
- MIMIC-IV for clinical note data
