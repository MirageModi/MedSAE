#!/bin/bash

# Set output directory (can be overridden via environment variable)
POLYSEMY_OUTPUT_DIR="${POLYSEMY_OUTPUT_DIR:-output/polysemy_analysis}"
NER_JSONL="${NER_JSONL:-$POLYSEMY_OUTPUT_DIR/ner_spans.jsonl}"

python jaccard_10.py \
    --topk_csv "$POLYSEMY_OUTPUT_DIR/topk_latents.csv" \
    --ner_jsonl "$NER_JSONL" \
    --lexicon_csv "$POLYSEMY_OUTPUT_DIR/polysemy_lexicon.csv" \
    --output_csv "$POLYSEMY_OUTPUT_DIR/metrics/jaccard_10.csv"

echo "Jaccard 10 final analysis complete."