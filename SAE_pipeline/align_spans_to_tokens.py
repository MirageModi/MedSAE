#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_spans_to_tokens.py — Map NER spans (char offsets) to LM token positions.

Parallelized version for large datasets (e.g., 7GB+ JSONL files).
"""

import argparse
import json
import multiprocessing
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

worker_tokenizer = None

def get_local_model_path(model_id: str) -> str:
    """Map HuggingFace model IDs to local paths if models are available locally."""
    model_path_map = {
        "google/medgemma-27b-text-it": "/bmbl_data/mirage/MedGemma",
        "aaditya/OpenBioLLM-Llama3-70B": "/bmbl_data/mirage/OpenBioLLM",
    }
    return model_path_map.get(model_id, model_id)

def worker_init(tokenizer_name: str):
    """Initialize tokenizer in each worker process."""
    global worker_tokenizer
    local_path = get_local_model_path(tokenizer_name)
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        worker_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
        if worker_tokenizer.pad_token is None and worker_tokenizer.eos_token is not None:
            worker_tokenizer.pad_token = worker_tokenizer.eos_token
    except Exception as e:
        print(f"Error initializing tokenizer in worker: {e}")

def build_offsets(text: str, add_special_tokens: bool = False) -> List[List[int]]:
    """Return per-token character offsets for a document."""
    if not text or worker_tokenizer is None:
        return []
    encoded = worker_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        return_tensors=None,
    )
    return encoded.get("offset_mapping", [])

def process_document(args: Tuple[str, str, List[Dict], bool]) -> List[Dict]:
    """Worker function to process a single document."""
    doc_id, text, spans, add_special_tokens = args
    
    offsets = build_offsets(text, add_special_tokens=add_special_tokens)
    if not offsets:
        return []

    aligned: List[Dict] = []
    for span in spans:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
        if end <= start:
            continue
            
        token_positions: List[int] = []
        
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            token_positions.append(idx)
            
        if not token_positions:
            continue

        aligned.append({
            "doc_id": doc_id,
            "start": start,
            "end": end,
            "text": span.get("text", text[start:end] if len(text) > end else ""),
            "entity_group": span.get("entity_group", span.get("label")),
            "score": span.get("score"),
            "token_positions": token_positions,
            "token_start": token_positions[0],
            "token_end": token_positions[-1],
            "token_count": len(token_positions),
        })
    return aligned

def load_ner_spans_generator(path: Path) -> Iterable[Dict]:
    """Yield spans one by one to save memory on initial load."""
    if not path.exists():
        raise FileNotFoundError(f"NER spans file not found: {path}")

    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        with path.open() as f:
            data = json.load(f)
            items = data.get("spans", []) if isinstance(data, dict) else data
            for item in items:
                yield item
    elif path.suffix == ".csv":
        for chunk in pd.read_csv(path, chunksize=10000):
            for _, row in chunk.iterrows():
                yield {
                    "doc_id": row.get("doc_id"),
                    "start": int(row.get("start", row.get("char_start", 0))),
                    "end": int(row.get("end", row.get("char_end", 0))),
                    "text": row.get("text", ""),
                    "entity_group": row.get("entity_group", row.get("label", "")),
                    "score": row.get("score", row.get("confidence", None)),
                }
    else:
        raise ValueError(f"Unsupported span file type: {path.suffix}")

def main():
    parser = argparse.ArgumentParser(description="Align NER spans to LM token indices (Parallel)")
    parser.add_argument("--ner_jsonl", required=True, help="NER spans file (jsonl/json/csv)")
    parser.add_argument("--csv_path", required=True, help="Source text CSV (same used for activations)")
    parser.add_argument("--text_column", required=True, help="Column with raw note text")
    parser.add_argument("--doc_id_column", default="", help="Optional doc id column; defaults to row index")
    parser.add_argument("--tokenizer_name", required=True, help="AutoTokenizer name for LM (fast)")
    parser.add_argument("--output", required=True, help="Output file path (jsonl/json/csv)")
    parser.add_argument("--output_format", default="jsonl", choices=["jsonl", "json", "csv"], help="Output format")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes (default: CPU count)")
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Include tokenizer BOS/EOS tokens when computing offsets.",
    )

    args = parser.parse_args()

    print(f"Loading text from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    if args.doc_id_column and args.doc_id_column not in df.columns:
        raise KeyError(f"doc_id_column '{args.doc_id_column}' not found in CSV")

    if args.doc_id_column:
        df["__doc_id"] = df[args.doc_id_column].astype(str)
    else:
        df["__doc_id"] = df.index.astype(str)

    text_map: Dict[str, str] = dict(zip(df["__doc_id"], df[args.text_column].astype(str)))
    del df
    print(f"Loaded {len(text_map)} source documents.")

    print(f"Loading and grouping NER spans from {args.ner_jsonl}...")
    ner_path = Path(args.ner_jsonl)
    spans_by_doc: Dict[str, List[Dict]] = defaultdict(list)
    
    total_lines = None
    if ner_path.suffix == ".jsonl":
        print("Estimating file size...")
        import os
        total_lines = int(os.path.getsize(ner_path) / 300)
    
    count = 0
    for span in tqdm(load_ner_spans_generator(ner_path), total=total_lines, desc="Reading Spans", unit="span"):
        doc_id = span.get("doc_id")
        if doc_id is not None:
            spans_by_doc[str(doc_id)].append(span)
        count += 1
    
    print(f"Loaded {count} spans across {len(spans_by_doc)} documents.")

    work_items = []
    for doc_id, doc_spans in spans_by_doc.items():
        if doc_id in text_map:
            work_items.append((doc_id, text_map[doc_id], doc_spans, args.add_special_tokens))
    
    print(f"Prepared {len(work_items)} documents for alignment.")
    
    del spans_by_doc
    del text_map
    import gc; gc.collect()

    num_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    num_workers = max(1, num_workers - 1)
    
    print(f"Starting alignment with {num_workers} worker processes...")
    
    aligned_all = []
    chunksize = min(100, max(1, len(work_items) // (num_workers * 4)))

    with multiprocessing.Pool(
        processes=num_workers, 
        initializer=worker_init, 
        initargs=(args.tokenizer_name,)
    ) as pool:
        for doc_aligned in tqdm(
            pool.imap_unordered(process_document, work_items, chunksize=chunksize),
            total=len(work_items),
            desc="Aligning",
            unit="doc"
        ):
            aligned_all.extend(doc_aligned)

    print(f"Aligned {len(aligned_all)} total spans.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing results to {output_path}...")
    if args.output_format == "jsonl":
        with output_path.open("w") as f:
            for row in tqdm(aligned_all, desc="Saving"):
                f.write(json.dumps(row) + "\n")
    elif args.output_format == "json":
        with output_path.open("w") as f:
            json.dump(aligned_all, f, indent=2)
    else:
        pd.DataFrame(aligned_all).to_csv(output_path, index=False)

    print("Done.")

if __name__ == "__main__":
    # Required for PyTorch/HuggingFace in multiprocessing context
    multiprocessing.set_start_method('spawn', force=True)
    main()