#!/bin/bash

POLYSEMY_OUTPUT_DIR="MedGemma/Layer52/polysemy_analysis"

if [ ! -f "$POLYSEMY_OUTPUT_DIR/metrics/monosemantic_baseline_dist.csv" ]; then
    python jaccard_control.py \
        --topk_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" \
        --ner_jsonl "polysemy_analysis/ner_spans.jsonl" \
        --output_dir "$POLYSEMY_OUTPUT_DIR/metrics"
fi