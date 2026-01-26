#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=warn
# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Dataset Configuration
CSV_PATH="/path/to/MIMIC/dataset.csv"
TEXT_COLUMN="input_with_target"
STRIP_BHC_SECTION=false
MAX_ROWS=50000
SHUFFLE=true
SHUFFLE_SEED=42
DOC_ID_COLUMN="note_id"   # Optional: set to CSV column name containing unique note IDs
SAVE_CHAR_OFFSETS=true
SAVE_DOC_ID=true

# Model Configuration
MODEL_ID="google/medgemma-27b-text-it"
# "aaditya/OpenBioLLM-Llama3-70B"
ATTN_IMPL="sdpa"
DTYPE="bfloat16"
LAYER_INDEX=52
# 67 for Llama3-70B

# Processing Configuration
CTX_LEN=64
STRIDE=64

BATCH_TEXTS=8
NORMALIZE=true
PREFIX_RANK_IN_FILENAMES=true
# Output Configuration
ACTIVATIONS_DIR="/path/to/activations/" 
VAL_PERCENT=0.05

# NER / Polysemy Configuration
NER_BATCH_SIZE=1024
NER_DEVICE="2" # Maps to physical GPU 2 (index 2 in CUDA_VISIBLE_DEVICES=0,1,2,6,7)
NER_MIN_SCORE=0.8
LM_TOKENIZER_NAME="$MODEL_ID"

# Viewer / Polysemy Export Configuration
RESERVE_FOR_SAE=0
LLM_MEM_OTHER_GIB=50
LLM_MEM_RESERVED_GIB=30
LLM_MICRO_BSZ=32
FEATURE_SHARD_SIZE=16384
NEIGHBORS_K=0
TOP_TOKENS_K=50
TOP_EXAMPLES_K=10
RUN_DELTA_CE=true
rerun_viewer=false
ALPHA_SWAP=1.0
RUN_PROBES=true
TOKEN_PROBE="corpus"
PROBE_BATCH_SIZE=1024
PROBE_VOCAB_LIMIT=0
PROBE_MAX_STEPS=2
PROBE_USE_PRE=true
PROBE_LABEL_FIELD="entity_group"
PROBE_LABEL_VALUES=""
PROBE_LABELS_SUBDIR="labels"
PROBE_LABELS_FILENAME="token_labels.npz"
PROBE_AUROC_THRESHOLD=0.20  
PROBE_AP_THRESHOLD=0.0
PROBE_PRESENCE_MIN=0.01  
PROBE_PRESENCE_DIFF_MIN=0.0  
RUN_N2G=true
ACTS_NORMALIZED="yes"
VIEWER_MAX_VAL_BATCHES=6000

# Multi-GPU Configuration
GPU_IDS="0,1,2,3,4,5,6,7"  
if [[ "$MODEL_ID" == "aaditya/OpenBioLLM-Llama3-70B" ]]; then
    NPROC_PER_NODE=2
    GPUS_PER_RANK_ACTS=4  
else
    NPROC_PER_NODE=4
    GPUS_PER_RANK_ACTS=2  
fi
USE_FALLBACK=false
RESUME_ACTIVATION_GENERATION=false
RESUME_SAE_TRAINING=false

# SAE Architecture Configuration
N_LATENTS=65536
# 98304 for Llama3-70B
PROBE_MAX_LATENTS=$N_LATENTS

D_MODEL=5376
# 8192 for Llama3-70B
K=128
K_AUX=512
DEAD_THRESHOLD_STEPS=900
RELU_AFTER_TOPK=true

# SAE Training Configuration
LR=2e-4
BETA1=0.9
BETA2=0.999
EPS=6.25e-10
WD=0.0
GRAD_CLIP=1.0
EMA_DECAY=0.999
EMA_ENABLE=false  # Disabled when using FSDP
AUX_COEFF=0.03125
TRAIN_STEPS=200000
MICRO_BSZ=128
LOG_EVERY=200
VAL_EVERY=2000
VAL_MICRO_BSZ=360
OPTIMIZER="adafactor"
ALLOW_LAST_SMALL_BATCH=true
# Data Loading Configuration (per process)
NUM_WORKERS_TRAIN=8  
PREFETCH_TRAIN=512      
NUM_WORKERS_VAL=16
PREFETCH_VAL=16
# SAE FSDP Configuration
SAE_FSDP=true
SAE_FSDP_DEVICES="0,1,2,3,4,5,6,7"
# Compute NPROC_PER_NODE_SAE from SAE_FSDP_DEVICES
IFS=',' read -ra SAE_GPU_ARRAY <<< "$SAE_FSDP_DEVICES"
NPROC_PER_NODE_SAE=${#SAE_GPU_ARRAY[@]}

# Output Configuration
SAE_SAVE_DIR="/path/to/sae"
mkdir -p $SAE_SAVE_DIR
# Pipeline Control
SKIP_ACTIVATION_GENERATION=true
SKIP_SAE_TRAINING=true

# =============================================================================
# END CONFIGURATION - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

# Create logs directory
mkdir -p logs

echo "=== SAE Pipeline Configuration ==="
echo "Model ID: $MODEL_ID"
echo "CSV Path: $CSV_PATH"
echo "Text Column: $TEXT_COLUMN"
echo "Strip BHC Section: $STRIP_BHC_SECTION"
echo "Max Rows: $MAX_ROWS"
echo "Layer Index: $LAYER_INDEX"
echo "N Latents: $N_LATENTS"
echo "K: $K"
echo "Optimizer: $OPTIMIZER"
echo "Data Type: $DTYPE"
echo "GPU IDs: $GPU_IDS"
echo "Processes per Node: $NPROC_PER_NODE"
echo "GPUs per Rank (Acts): $GPUS_PER_RANK_ACTS"
echo "SAE Processes per Node: $NPROC_PER_NODE_SAE"
echo "Use Fallback: $USE_FALLBACK"
echo "Activations Directory: $ACTIVATIONS_DIR"
echo "SAE Save Directory: $SAE_SAVE_DIR"
echo "Validation Percent: $VAL_PERCENT"
echo "Batch Texts: $BATCH_TEXTS"
echo "Normalize: $NORMALIZE"
echo "Prefix Rank in Filenames: $PREFIX_RANK_IN_FILENAMES"
echo "Training Steps: $TRAIN_STEPS"
echo "Micro Batch Size: $MICRO_BSZ"
echo "Num Workers (train): $NUM_WORKERS_TRAIN"
echo "Prefetch (train): $PREFETCH_TRAIN"
echo "Skip Activation Generation: $SKIP_ACTIVATION_GENERATION"
echo "Skip SAE Training: $SKIP_SAE_TRAINING"
echo "Resume: $RESUME"
echo "================================="

# Step 1: Generate LLM Activations
# --- Step 1: Generating LLM Activations (multi-proc, 1 proc/GPU) ---
if [[ "$SKIP_ACTIVATION_GENERATION" != "true" ]]; then
  echo ""
  echo "=== Step 1: Generating LLM Activations (distributed) ==="
  echo "Starting activation generation at $(date)"

  # For single-node VM setup
  MASTER_ADDR="localhost"
  MASTER_PORT=29501

  export TOKENIZERS_PARALLELISM=false
  ACT_CMD="torchrun \
    --nnodes=1 \
    --nproc-per-node=${NPROC_PER_NODE} \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    gen_LLMActivations.py \
      --csv_path \"$CSV_PATH\" \
      --text_column \"$TEXT_COLUMN\" \
      $( [[ \"$STRIP_BHC_SECTION\" == \"true\" ]] && echo --strip_bhc_section ) \
      $( [[ \"$MODEL_ID\" == \"aaditya/OpenBioLLM-Llama3-70B\" ]] && echo --device_map \"auto\" ) \
      --gpus_per_rank $GPUS_PER_RANK_ACTS \
      --model_id \"$MODEL_ID\" \
      --attn_impl \"$ATTN_IMPL\" \
      --dtype \"$DTYPE\" \
      --layer_index $LAYER_INDEX \
      --ctx_len $CTX_LEN \
      --stride $STRIDE \
      --batch_texts $BATCH_TEXTS \
      $( [[ \"$NORMALIZE\" == \"true\" ]] && echo --normalize ) \
      --output_dir \"$ACTIVATIONS_DIR\" \
      --val_percent $VAL_PERCENT \
      --max_rows $MAX_ROWS \
      $( [[ \"$SHUFFLE\" == \"true\" ]] && echo --shuffle --shuffle_seed $SHUFFLE_SEED ) \
      $( [[ \"$RESUME_ACTIVATION_GENERATION\" == \"true\" ]] && echo --resume ) \
      $( [[ \"$PREFIX_RANK_IN_FILENAMES\" == \"true\" ]] && echo --prefix_rank_in_filenames ) \
      $( [[ \"$SAVE_CHAR_OFFSETS\" == \"true\" ]] && echo --save_char_offsets ) \
      $( [[ \"$SAVE_DOC_ID\" == \"true\" ]] && echo --save_doc_id ) \
      $( [[ -n \"$DOC_ID_COLUMN\" ]] && echo --doc_id_column \"$DOC_ID_COLUMN\" )"

  echo "=== Debug: Normalization Configuration ==="
  echo "NORMALIZE flag: $NORMALIZE"
  if [[ "$NORMALIZE" == "true" ]]; then
    echo "Will pass normalize flag: --normalize"
  else
    echo "Will pass normalize flag: (not set)"
  fi
  echo "=========================================="
  echo "Running: $ACT_CMD"
  eval $ACT_CMD
  CODE=$?
  if [[ $CODE -ne 0 ]]; then
    echo "ERROR: Activation generation failed with exit code $CODE"
    exit $CODE
  fi
  echo "Activation generation completed at $(date)"
else
  echo "=== Step 1: Skipping Activation Generation ==="
fi

# Check if activations exist
if [[ ! -d "$ACTIVATIONS_DIR" ]]; then
    echo "ERROR: Activations directory $ACTIVATIONS_DIR does not exist"
    exit 1
fi

# Check for at least one shard containing train/val subdirs
FOUND_SPLITS=false
if [[ -d "$ACTIVATIONS_DIR" ]]; then
    for SHARD_DIR in "$ACTIVATIONS_DIR"/shard_*; do
        if [[ -d "$SHARD_DIR/train" && -d "$SHARD_DIR/val" ]]; then
            FOUND_SPLITS=true
            break
        fi
    done
fi

if [[ "$FOUND_SPLITS" != "true" ]]; then
    echo "ERROR: No train/val splits found in any shard under $ACTIVATIONS_DIR"
    echo "Please run activation generation first or check the directory structure"
    exit 1
fi

# Step 2: Train SAE
if [[ "$SKIP_SAE_TRAINING" != "true" ]]; then
    echo ""
    echo "=== Step 2: Training SAE ==="
    echo "Starting SAE training at $(date)"
    
    # For single-node VM setup
    MASTER_ADDR="localhost"
    MASTER_PORT=29500
    
    # Build SAE training command
    SAE_CMD="torchrun"
    SAE_CMD="$SAE_CMD --nnodes=1"
    SAE_CMD="$SAE_CMD --nproc-per-node=${NPROC_PER_NODE_SAE}"
    SAE_CMD="$SAE_CMD --node_rank=0"
    SAE_CMD="$SAE_CMD --rdzv_backend=c10d"
    SAE_CMD="$SAE_CMD --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
    SAE_CMD="$SAE_CMD sae_trainer.py"
    
    if [[ "$SAE_FSDP" == "true" ]]; then
        SAE_CMD="$SAE_CMD --sae_fsdp"
    fi
    
    SAE_CMD="$SAE_CMD --activations_dir \"$ACTIVATIONS_DIR\""
    SAE_CMD="$SAE_CMD --n_latents $N_LATENTS"
    SAE_CMD="$SAE_CMD --d_model $D_MODEL"
    SAE_CMD="$SAE_CMD --k $K"
    SAE_CMD="$SAE_CMD --k_aux $K_AUX"
    SAE_CMD="$SAE_CMD --dead_threshold_steps $DEAD_THRESHOLD_STEPS"
    
    if [[ "$ALLOW_LAST_SMALL_BATCH" == "true" ]]; then
        SAE_CMD="$SAE_CMD --allow_last_small_batch"
    fi

    if [[ "$RELU_AFTER_TOPK" == "true" ]]; then
        SAE_CMD="$SAE_CMD --relu_after_topk"
    fi
    
    SAE_CMD="$SAE_CMD --optimizer \"$OPTIMIZER\""
    SAE_CMD="$SAE_CMD --dtype \"$DTYPE\""
    SAE_CMD="$SAE_CMD --lr $LR"
    SAE_CMD="$SAE_CMD --beta1 $BETA1"
    SAE_CMD="$SAE_CMD --beta2 $BETA2"
    SAE_CMD="$SAE_CMD --eps $EPS"
    SAE_CMD="$SAE_CMD --wd $WD"
    SAE_CMD="$SAE_CMD --grad_clip $GRAD_CLIP"
    SAE_CMD="$SAE_CMD --ema_decay $EMA_DECAY"
    
    if [[ "$EMA_ENABLE" == "true" ]]; then
        SAE_CMD="$SAE_CMD --ema_enable"
    fi
    
    SAE_CMD="$SAE_CMD --aux_coeff $AUX_COEFF"
    SAE_CMD="$SAE_CMD --train_steps $TRAIN_STEPS"
    SAE_CMD="$SAE_CMD --micro_bsz $MICRO_BSZ"
    SAE_CMD="$SAE_CMD --log_every $LOG_EVERY"
    SAE_CMD="$SAE_CMD --val_every $VAL_EVERY"
    SAE_CMD="$SAE_CMD --val_micro_bsz $VAL_MICRO_BSZ"
    SAE_CMD="$SAE_CMD --num_workers_train $NUM_WORKERS_TRAIN"
    SAE_CMD="$SAE_CMD --prefetch_train $PREFETCH_TRAIN"
    SAE_CMD="$SAE_CMD --num_workers_val $NUM_WORKERS_VAL"
    SAE_CMD="$SAE_CMD --prefetch_val $PREFETCH_VAL"
    SAE_CMD="$SAE_CMD --sae_fsdp_devices \"$SAE_FSDP_DEVICES\""
    SAE_CMD="$SAE_CMD --save_dir \"$SAE_SAVE_DIR\""
    if [[ "$RESUME_SAE_TRAINING" == "true" ]]; then
        SAE_CMD="$SAE_CMD --resume"
    fi
    
    echo "Running: $SAE_CMD"
    eval $SAE_CMD
    
    SAE_EXIT_CODE=$?
    if [[ $SAE_EXIT_CODE -ne 0 ]]; then
        echo "ERROR: SAE training failed with exit code $SAE_EXIT_CODE"
        exit $SAE_EXIT_CODE
    fi
    
    echo "SAE training completed at $(date)"
    echo "SAE model saved to: $SAE_SAVE_DIR"
else
    echo ""
    echo "=== Step 2: Skipping SAE Training ==="
fi




# =============================================================================
# Step 3: Polysemy Analysis Pipeline (optional)
# =============================================================================

# Set this to true to run polysemy analysis

RUN_POLYSEMY_ANALYSIS=true

echo "=== Calibrating feature scales ==="
CKPT="/path/to/best/SAE.pt"
SAE_CKPT=${CKPT%.pt}_calibrated.pt

CAL_SPLIT="train"           # train or val
CAL_STEPS=20000             # number of activation batches to scan (set 0/blank for all)
CAL_QUANTILE=0.999          # high quantile for robust scaling
CAL_MAX_FILES=20000         # Limit the number of activation files to load
python calibrate_sae_feature_scales.py \
    "$CKPT" \
    "$ACTIVATIONS_DIR" \
    --split "$CAL_SPLIT" \
    --device cuda \
    --quantile "$CAL_QUANTILE" \
    --preload \
    ${CAL_STEPS_ARG} \
    --output "$SAE_CKPT" \
    --max-files "$CAL_MAX_FILES"

echo "Calibrated checkpoint: $SAE_CKPT"

if [[ "$RUN_POLYSEMY_ANALYSIS" == "true" ]]; then
    echo ""
    echo "=== Step 3: Polysemy Analysis Pipeline ==="
    echo "Starting polysemy analysis at $(date)"
    
    # Configuration for polysemy analysis
    POLYSEMY_ROOT="polysemy_analysis"
    POLYSEMY_OUTPUT_DIR="${POLYSEMY_ROOT}/${MODEL_ID}/layer${LAYER_INDEX}"
    mkdir -p "$POLYSEMY_OUTPUT_DIR"
    VIEWER_PROBES_DIR="$POLYSEMY_OUTPUT_DIR/viewer/probes"
    LABELS_DIR="$POLYSEMY_OUTPUT_DIR/$PROBE_LABELS_SUBDIR"
    PROBE_LABELS_PATH="$LABELS_DIR/$PROBE_LABELS_FILENAME"
    PROBES_CSV="$VIEWER_PROBES_DIR/single_latent_probes.csv"
    
    # Step 3a: Run OpenMed NER on raw notes
    if [[ -f "polysemy_analysis/ner_spans.jsonl" ]]; then
        echo "Step 3a: Skipping NER (found ner_spans.jsonl)"
    else
        echo "Step 3a: Running OpenMed NER..."
        CMD_NER="python ner_infer_gliner_fast.py \
            --csv_path \"$CSV_PATH\" \
            --text_column \"$TEXT_COLUMN\" \
            --out_jsonl polysemy_analysis/ner_spans.jsonl \
            --batch_size $NER_BATCH_SIZE \
            --device \"$NER_DEVICE\" \
            --max_tokens 256 \
            --stride 64 \
            --min_score $NER_MIN_SCORE"
        if [[ -n "$DOC_ID_COLUMN" ]]; then
            CMD_NER+=" --doc_id_column \"$DOC_ID_COLUMN\""
        fi
        echo "Running: $CMD_NER"
        eval $CMD_NER
    fi

    # Step 3b: Align NER spans to LM token positions
    if [[ -f "$POLYSEMY_OUTPUT_DIR/ner_spans_with_tokens.jsonl" ]]; then
        echo "Step 3b: Skipping alignment (found ner_spans_with_tokens.jsonl)"
    else
        echo "Step 3b: Aligning NER spans to LM tokens..."
        CMD_ALIGN="python align_spans_to_tokens.py \
            --ner_jsonl polysemy_analysis/ner_spans.jsonl \
            --csv_path \"$CSV_PATH\" \
            --text_column \"$TEXT_COLUMN\" \
            --tokenizer_name \"$LM_TOKENIZER_NAME\" \
            --output \"$POLYSEMY_OUTPUT_DIR/ner_spans_with_tokens.jsonl\" \
            --add_special_tokens"
        if [[ -n "$DOC_ID_COLUMN" ]]; then
            CMD_ALIGN+=" --doc_id_column \"$DOC_ID_COLUMN\""
        fi
        echo "Running: $CMD_ALIGN"
        eval $CMD_ALIGN
    fi

    # Step 3c: Build polysemy lexicon
    if [[ -f "$POLYSEMY_OUTPUT_DIR/polysemy_lexicon.csv" ]]; then
        echo "Step 3c: Skipping lexicon build (found polysemy_lexicon.csv)"
    else
        echo "Step 3c: Building polysemy lexicon..."
        python build_polysemy_lexicon.py \
            --output "$POLYSEMY_OUTPUT_DIR/polysemy_lexicon.csv" \
            --seed_only
    fi
# Step 3d: Export Top-K latents per token (and run probes) - PARALLELIZED
# ========================================================================
    VIEWER_LATENTS_LIMIT=$N_LATENTS  

    if [[ -f "$PROBES_CSV" && -f "$POLYSEMY_OUTPUT_DIR/viewer/viewer_pack.json" && -f "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" && "$rerun_viewer" == "false" ]]; then
        echo "Step 3d: Skipping Probes, Viewer, and TopK export (all artifacts present)"
    else
        echo "Step 3d: Running Probes, Viewer Export, and TopK CSV (5x Parallel Shards)..."
        
        # 1. Pre-compute Probe Targets (Runs once on CPU)
        if [[ "$RUN_PROBES" == "true" ]]; then
            mkdir -p "$LABELS_DIR"
            if [[ ! -f "$PROBE_LABELS_PATH" ]]; then
                echo "Building token-level probe labels from new NER..."
                python make_labels.py \
                    --acts_root "$ACTIVATIONS_DIR" \
                    --acts_split val \
                    --ner_tokens_jsonl "$POLYSEMY_OUTPUT_DIR/ner_spans_with_tokens.jsonl" \
                    --entity_field "$PROBE_LABEL_FIELD" \
                    --entity_values "$PROBE_LABEL_VALUES" \
                    --out_npz "$PROBE_LABELS_PATH" \
                    --source_csv "$CSV_PATH" \
                    --max_rows "$MAX_ROWS" \
                    --shuffle_seed "$SHUFFLE_SEED" \

                    
            fi
        fi
        mkdir -p "$VIEWER_PROBES_DIR"

        # ---------------------------------------------------------------------
        # BEST PRACTICE: Define Base Command as an Array
        # ---------------------------------------------------------------------
        CMD_ARGS=(
            python sae_viewer_topk_sharded_chars.py
            --ckpt_path "$SAE_CKPT"
            --export_dir "$POLYSEMY_OUTPUT_DIR/viewer"
            --model_id "$MODEL_ID"
            --dtype "$DTYPE"
            --layer_index "$LAYER_INDEX"
            --ctx_len "$CTX_LEN"
            --stride "$STRIDE"
            --llm_micro_bsz "$LLM_MICRO_BSZ"
            --feature_shard_size "$FEATURE_SHARD_SIZE"
            --neighbors_k "$NEIGHBORS_K"
            --top_tokens_k "$TOP_TOKENS_K"
            --top_examples_k "$TOP_EXAMPLES_K"
            --dataset_format "bhc_csv"
            --csv_path "$CSV_PATH"
            --text_column "$TEXT_COLUMN"
            --use_saved_acts
            --acts_root "$ACTIVATIONS_DIR"
            --acts_split val
            --acts_max_files 0
            --max_viewer_latents "$VIEWER_LATENTS_LIMIT"
            --max_val_batches "$VIEWER_MAX_VAL_BATCHES"
            --token_probe "$TOKEN_PROBE"
            --acts_normalized "$ACTS_NORMALIZED"
            --alpha_swap "$ALPHA_SWAP"
            --export_topk_latents_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv"
            --filter_terms_csv "$POLYSEMY_OUTPUT_DIR/full_lexicon.csv"
            --with_offsets
            --total_shards 8
        )
        
        # Add merged probes CSV if it exists (to skip already-processed latents)
        if [[ -f "$PROBES_CSV" ]]; then
            CMD_ARGS+=( --merged_probes_csv "$PROBES_CSV" )
        fi
        
        # Conditionally add relu_after_topk if enabled
        if [[ "$RELU_AFTER_TOPK" == "true" ]]; then
            CMD_ARGS+=( --relu_after_topk )
        fi

        # Conditionally append arguments using +=
        if [[ "$RUN_DELTA_CE" == "true" ]]; then #TODO: FLIP THIS LATER
            CMD_ARGS+=( --run_delta_ce --delta_ce_max_batches 200 )
        fi
        
        if [[ "$RUN_PROBES" == "true" ]]; then
            CMD_ARGS+=( 
                --run_probes 
                --labels_path "$PROBE_LABELS_PATH"
                --probe_use_pre
                --probe_max_steps "$PROBE_MAX_STEPS"
                --probe_max_latents "$PROBE_MAX_LATENTS"
                --probe_batch_size "$PROBE_BATCH_SIZE"
                --probe_vocab_limit "$PROBE_VOCAB_LIMIT"
            )
        fi
        # ---------------------------------------------------------------------

        # 2. Launch Parallel Shards
        IFS=',' read -r -a PHYSICAL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        echo "Launching 5 independent GPU jobs on physical devices: ${PHYSICAL_GPUS[@]}..."
        pids=()

        for i in {0..7}; do
            TARGET_GPU=${PHYSICAL_GPUS[$i]}
            
            (
                export CUDA_VISIBLE_DEVICES="$TARGET_GPU"
                echo "Starting Shard $i on physical GPU $TARGET_GPU..."
                
                # Execute the array using "${CMD_ARGS[@]}"
                "${CMD_ARGS[@]}" --shard_id "$i" > "logs/shard_${i}.log" 2>&1
            ) &
            pids+=($!)
        done

        # 3. Wait for completion
        echo "Waiting for PIDs: ${pids[*]}..."
        wait "${pids[@]}"
        echo "All shards finished."

        # 4. Merge Outputs
        echo "Merging sharded outputs..."

        # A) Merge Probes CSV
        if [[ -f "$PROBES_CSV" ]]; then
            echo "Skipping probes merge (final file already exists: $PROBES_CSV)"
        elif [[ -f "$VIEWER_PROBES_DIR/single_latent_probes_shard_0.csv" ]]; then
            head -n 1 "$VIEWER_PROBES_DIR/single_latent_probes_shard_0.csv" > "$PROBES_CSV"
            for f in "$VIEWER_PROBES_DIR"/single_latent_probes_shard_*.csv; do
                tail -n +2 "$f" >> "$PROBES_CSV"
                rm "$f"
            done
            echo "Merged probes to $PROBES_CSV"
        fi

        # B) Merge Top Tokens CSV (Latent Sharded)
        TOP_TOKENS_OUT="$POLYSEMY_OUTPUT_DIR/viewer/top_tokens_per_latent.csv"
        if [[ -f "$TOP_TOKENS_OUT" ]]; then
            echo "Skipping top tokens merge (final file already exists: $TOP_TOKENS_OUT)"
        else
            # Use array for globbing to be safe
            TOK_FILES=( "$POLYSEMY_OUTPUT_DIR"/viewer/top_tokens_shard_*.csv )
            if [[ -f "${TOK_FILES[0]}" ]]; then
                head -n 1 "${TOK_FILES[0]}" > "$TOP_TOKENS_OUT"
                for f in "${TOK_FILES[@]}"; do
                    tail -n +2 "$f" >> "$TOP_TOKENS_OUT"
                    rm "$f"
                done
                echo "Merged top tokens (viewer) to $TOP_TOKENS_OUT"
            fi
        fi

        # C) Merge TopK Latents CSV (Data Sharded)
        MAIN_TOPK_OUT="$POLYSEMY_OUTPUT_DIR/topk_latents.csv"
        if [[ -f "$MAIN_TOPK_OUT" ]]; then
            echo "Skipping TopK latents merge (final file already exists: $MAIN_TOPK_OUT)"
        else
            DATA_SHARD_FILES=( "$POLYSEMY_OUTPUT_DIR"/topk_latents_shard_*.csv )
            if [[ -f "${DATA_SHARD_FILES[0]}" ]]; then
                head -n 1 "${DATA_SHARD_FILES[0]}" > "$MAIN_TOPK_OUT"
                for f in "${DATA_SHARD_FILES[@]}"; do
                    tail -n +2 "$f" >> "$MAIN_TOPK_OUT"
                    rm "$f"
                done
                echo "Merged Top-K Latents (Polysemy Data) to $MAIN_TOPK_OUT"
            fi
        fi

        # D) Merge Viewer JSONs (Refactored to HEREDOC for safety)
        VIEWER_JSON_OUT="$POLYSEMY_OUTPUT_DIR/viewer/viewer_pack.json"
        if [[ -f "$VIEWER_JSON_OUT" ]]; then
            echo "Skipping viewer JSON merge (final file already exists: $VIEWER_JSON_OUT)"
        else
            python - <<EOF
import json, glob, os
output_file = '$POLYSEMY_OUTPUT_DIR/viewer/viewer_pack.json'
merged = {'features': {}}
files = sorted(glob.glob('$POLYSEMY_OUTPUT_DIR/viewer/viewer_pack_shard_*.json'))
if files:
    print(f'Merging {len(files)} JSON shards...')
    for f in files:
        try:
            with open(f) as h:
                data = json.load(h)
                if 'features' in data:
                    merged['features'].update(data['features'])
            os.remove(f)
        except Exception as e:
            print(f'Error merging {f}: {e}')
    with open(output_file, 'w') as f:
        json.dump(merged, f, separators=(',', ':'))
EOF
        fi
    fi

    if [[ -f "$POLYSEMY_OUTPUT_DIR/metrics/sense_linked_latents.csv" ]]; then
        echo "Step 3e: Skipping polysemy metrics (found metrics/sense_linked_latents.csv)"
    else
        echo "Step 3e: Computing polysemy metrics..."
        METRIC_PROBE_ARGS=()
        if [[ "$RUN_PROBES" == "true" && -f "$PROBES_CSV" ]]; then
            METRIC_PROBE_ARGS+=(--probe_csv "$PROBES_CSV")
            METRIC_PROBE_ARGS+=(--probe_auroc_threshold "$PROBE_AUROC_THRESHOLD")
            METRIC_PROBE_ARGS+=(--probe_ap_threshold "$PROBE_AP_THRESHOLD")
        fi
        python compute_polysemy_metrics.py \
            --topk_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" \
            --lexicon_csv "$POLYSEMY_OUTPUT_DIR/polysemy_lexicon.csv" \
            --ner_jsonl "polysemy_analysis/ner_spans.jsonl" \
            --output_dir "$POLYSEMY_OUTPUT_DIR/metrics" \
            --fdr_threshold 0.05 \
            --log_level "DEBUG" \
            --probe_presence_min "$PROBE_PRESENCE_MIN" \
            --probe_presence_diff_min "$PROBE_PRESENCE_DIFF_MIN" \
            "${METRIC_PROBE_ARGS[@]}"
    fi
    
    # Step 3f: Run ablation ΔCE analysis
    if [[ -f "$POLYSEMY_OUTPUT_DIR/metrics/ablation_delta_ce.csv" ]]; then
        echo "Step 3f: Skipping ablation ΔCE (found metrics/ablation_delta_ce.csv)"
    else
        echo "Step 3f: Running ablation ΔCE analysis..."
        python run_ablation_delta_ce.py \
            --model_id "$MODEL_ID" \
            --ckpt_path "$SAE_CKPT" \
            --layer_index $LAYER_INDEX \
            --sense_linked_csv "$POLYSEMY_OUTPUT_DIR/metrics/sense_linked_latents.csv" \
            --topk_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" \
            --acts_root "$ACTIVATIONS_DIR" \
            --acts_split val \
            --acts_max_files 0 \
            --output "$POLYSEMY_OUTPUT_DIR/metrics/ablation_delta_ce.csv" \
            --dtype "$DTYPE" \
            --max_ablations 2000 \
            --max_ablations_per_sense 20
    fi
    
    # Step 3g: Run mixed-effects models
    if [[ -d "$POLYSEMY_OUTPUT_DIR/models" && -n "$(ls -A "$POLYSEMY_OUTPUT_DIR/models" 2>/dev/null)" ]]; then
        echo "Step 3g: Skipping mixed-effects models (models directory already populated)"
    else
        echo "Step 3g: Fitting mixed-effects models..."
        Rscript run_mixed_effects.R \
            "$POLYSEMY_OUTPUT_DIR/metrics" \
            "$POLYSEMY_OUTPUT_DIR/models"
    fi
fi

echo ""
echo "=== Pipeline Complete ==="
echo "Pipeline finished at $(date)"
echo "Activations: $ACTIVATIONS_DIR"
echo "SAE Model: $SAE_SAVE_DIR"
if [[ "$RUN_POLYSEMY_ANALYSIS" == "true" ]]; then
    echo "Polysemy Analysis: $POLYSEMY_OUTPUT_DIR"
fi
echo "========================="
