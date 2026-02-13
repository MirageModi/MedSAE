#!/bin/bash

# Set output directory (can be overridden via environment variable)
POLYSEMY_OUTPUT_DIR="${POLYSEMY_OUTPUT_DIR:-output/polysemy_analysis}"
NER_JSONL="${NER_JSONL:-polysemy_analysis/ner_spans.jsonl}"

if [ ! -f "$POLYSEMY_OUTPUT_DIR/metrics/monosemantic_baseline_dist.csv" ]; then
    python jaccard_control.py \
        --topk_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" \
        --ner_jsonl "$NER_JSONL" \
        --output_dir "$POLYSEMY_OUTPUT_DIR/metrics"
fi