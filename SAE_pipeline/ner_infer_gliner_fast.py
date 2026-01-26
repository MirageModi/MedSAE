#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run GLiNER-BioMed NER over clinical notes and cache span outputs with
document-level character offsets.

The resulting JSONL file contains one entity span per line with:
  doc_id, start, end, text, entity_group, score

Chunking is performed using word spans to preserve original character offsets
while supporting long documents.
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import os, shutil
import math

try:
    from gliner import GLiNER  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Missing optional dependency 'gliner'. "
        "Install it with `pip install gliner` to run OpenMed GLiNER inference."
    ) from exc

try:
    from tqdm import tqdm  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


# --- CONFIGURATION ---
# SOTA Biomedical GLiNER Model
GLINER_MODEL = "Ihor/gliner-biomed-large-v1.0"

DEFAULT_LABELS: List[str] = [
  # --- Distinguishing "Discharge" ---
  "administrative healthcare event",      # For 'discharge' (admin), 'admission'
  "bodily fluid or secretion",            # For 'discharge' (clinical/fluid)

  # --- Distinguishing Specialties (Arrest, Failure, Block) ---
  "cardiovascular disorder",              # For 'cardiac', 'cardiovascular' senses
  "respiratory or pulmonary disorder",    # For 'respiratory' senses
  "neurological or mental disorder",      # For 'neurological' senses
  "renal or kidney disorder",             # For 'renal' senses
  "gastrointestinal disorder",            # For 'biliary', 'gastric' senses
  "musculoskeletal or orthopedic disorder", # For 'orthopedic' senses
  
  # --- Distinguishing "Mass" / "Lesion" ---
  "anatomical structure",                 # For 'anatomical' mass (e.g. "the ventricular mass")
  "radiological finding or shadow",       # For 'radiological' mass (e.g. "lung mass on X-ray")
  "dermatological or skin lesion",        # For 'dermatological' lesion

  # --- Standard Medical Categories (Keep these for coverage) ---
  "medication or therapeutic agent",
  "medical procedure or intervention",
  "laboratory measurement"
]

WORD_PATTERN = re.compile(r"\S+")


def parse_device(device_arg: str) -> str:
    normalized = device_arg.strip().lower()
    if normalized in {"cpu", "-1"}:
        return "cpu"
    if normalized.startswith("cuda"):
        return device_arg.strip()
    try:
        idx = int(device_arg)
    except ValueError:
        return device_arg.strip()
    return "cpu" if idx < 0 else f"cuda:{idx}"


def word_spans(text: str) -> List[Tuple[int, int]]:
    return [(match.start(), match.end()) for match in WORD_PATTERN.finditer(text or "")]


def chunks_by_word_spans(
    text: str,
    max_words: int,
    overlap_words: int,
) -> List[Tuple[str, int]]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    spans = word_spans(text)
    total_words = len(spans)
    if total_words == 0:
        return []

    effective_overlap = max(0, min(overlap_words, max_words - 1))
    step = max_words - effective_overlap

    chunks: List[Tuple[str, int]] = []
    index = 0
    while index < total_words:
        end_index = min(total_words, index + max_words)
        start_char = spans[index][0]
        end_char = spans[end_index - 1][1]
        if end_char <= start_char:
            break
        chunks.append((text[start_char:end_char], start_char))
        if end_index == total_words:
            break
        index += step
    return chunks


def deduplicate_spans(spans: Iterable[Dict]) -> List[Dict]:
    seen: Dict[Tuple[str, int, int, str], Dict] = {}
    for span in spans:
        key = (
            span["doc_id"],
            span["start"],
            span["end"],
            span.get("entity_group", ""),
        )
        if key not in seen or (span.get("score", 0) or 0) > (seen[key].get("score", 0) or 0):
            seen[key] = span
    return list(seen.values())


def run_ner_on_document(
    doc_id: str,
    text: str,
    model,
    labels: List[str],
    batch_size: int,
    max_words: int,
    overlap_words: int,
    min_score: float,
) -> List[Dict]:
    chunks = chunks_by_word_spans(text, max_words=max_words, overlap_words=overlap_words)
    if not chunks:
        return []

    # Filter out empty or whitespace-only chunks to avoid langdetect errors
    valid_chunks: List[Tuple[str, int]] = []
    for chunk, offset in chunks:
        # Keep chunks that have at least one non-whitespace character
        if chunk and chunk.strip():
            valid_chunks.append((chunk, offset))
    
    if not valid_chunks:
        return []
    
    chunk_texts = [chunk for chunk, _ in valid_chunks]
    chunk_offsets = [offset for _, offset in valid_chunks]

    all_spans: List[Dict] = []
    seen: Dict[Tuple[str, int, int, str], Dict] = {}

    if batch_size <= 0:
        batch_size = len(chunk_texts)

    # Use no_grad context to prevent gradient computation and reduce memory usage
    with torch.no_grad():
        for start_idx in range(0, len(chunk_texts), batch_size):
            batch_text = chunk_texts[start_idx:start_idx + batch_size]
            batch_offsets = chunk_offsets[start_idx:start_idx + batch_size]
            
            # Filter out any empty strings from the batch (defensive check)
            filtered_batch = []
            filtered_offsets = []
            for txt, off in zip(batch_text, batch_offsets):
                if txt and txt.strip():
                    filtered_batch.append(txt)
                    filtered_offsets.append(off)
            
            if not filtered_batch:
                continue
            
            try:
                batch_results = model.batch_predict_entities(
                    filtered_batch,
                    labels,
                    flat_ner=True,
                    threshold=min_score,
                )
            except Exception as e:
                # Handle any errors from GLiNER (e.g., langdetect issues)
                # Log and continue with next batch
                print(f"Warning: Error processing batch at offset {filtered_offsets[0] if filtered_offsets else 'unknown'}: {e}")
                continue
            
            if batch_results is None:
                batch_results = [[] for _ in filtered_batch]
            elif not isinstance(batch_results, list):
                batch_results = [batch_results]

            for offset, entities in zip(filtered_offsets, batch_results):
                if not entities:
                    continue
                for entity in entities:
                    label = entity.get("label") or entity.get("entity_group")
                    if not label:
                        continue
                    start = int(entity.get("start", 0)) + offset
                    end = int(entity.get("end", 0)) + offset
                    score = float(entity.get("score", 0.0) or 0.0)
                    if end <= start or score < min_score:
                        continue

                    span = {
                        "doc_id": doc_id,
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "entity_group": label,
                        "score": score,
                    }
                    key = (doc_id, start, end, label)
                    existing = seen.get(key)
                    if not existing or score > (existing.get("score", 0) or 0):
                        seen[key] = span

            # Clear GPU cache periodically to prevent memory accumulation
            if torch.cuda.is_available() and start_idx % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

    all_spans.extend(seen.values())
    return sorted(all_spans, key=lambda span: (span["start"], span["end"], span["entity_group"]))


def main():
    import sys
    print(f"Starting NER inference script (PID: {os.getpid()})", flush=True)
    print(f"Arguments: {sys.argv}", flush=True)
    
    parser = argparse.ArgumentParser(description="Run OpenMed GLiNER NER over clinical notes")
    parser.add_argument("--csv_path", required=True, help="Input CSV with notes")
    parser.add_argument("--text_column", required=True, help="Column containing raw note text")
    parser.add_argument("--doc_id_column", default="", help="Optional column for document IDs")
    parser.add_argument("--out_jsonl", required=True, help="Output JSONL file path")
    parser.add_argument("--batch_size", type=int, default=16, help="Pipeline batch size")
    parser.add_argument("--device", default="0", help="Device specification (e.g., 0, cuda:0, cpu)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max words per chunk")
    parser.add_argument("--stride", type=int, default=64, help="Overlap words between chunks")
    parser.add_argument("--min_score", type=float, default=0.9, help="Minimum score to keep a span")

    
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="This process's shard id [0..num_shards-1]")
    parser.add_argument("--dist", action="store_true",
                        help="Use torchrun env (WORLD_SIZE/RANK/LOCAL_RANK); merge parts on rank 0")

    args = parser.parse_args()
    print(f"Parsed arguments: num_shards={args.num_shards}, shard_id={args.shard_id}, device={args.device}", flush=True)

    df = pd.read_csv(args.csv_path)
    if args.text_column not in df.columns:
        raise KeyError(f"text_column '{args.text_column}' not found in CSV")

    if args.doc_id_column:
        if args.doc_id_column not in df.columns:
            raise KeyError(f"doc_id_column '{args.doc_id_column}' not in CSV")
        df["__doc_id"] = df[args.doc_id_column].astype(str)
    else:
        df["__doc_id"] = df.index.astype(str)

    rank = 0
    world = 1
    device = parse_device(args.device)

    if args.dist:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        import torch.distributed as dist
        backend = "nccl" if (torch.cuda.is_available() and device != "cpu") else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        num_shards = world
        shard_id = rank
    else:
        num_shards = max(1, int(args.num_shards))
        shard_id = max(0, int(args.shard_id))
        if shard_id >= num_shards:
            raise ValueError(f"shard_id {shard_id} >= num_shards {num_shards}")

    # Balanced, disjoint sharding: every nth row.
    df = df.iloc[shard_id::num_shards].copy()

    print(f"[rank={rank}] loading model {GLINER_MODEL} on {device}...")
    ner_model = GLiNER.from_pretrained(GLINER_MODEL)

    ner_model = ner_model.to(device)
    ner_model.eval()

    shard_out = Path(f"{args.out_jsonl}.part{shard_id:02d}")
    shard_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with shard_out.open("w") as fout:
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                            desc=f"Shard {shard_id}/{num_shards}")):
            doc_id = row["__doc_id"]
            text = str(row[args.text_column])
            spans = run_ner_on_document(
                doc_id=doc_id,
                text=text,
                model=ner_model,
                labels=DEFAULT_LABELS,
                batch_size=args.batch_size,
                max_words=max(args.max_tokens, 1),
                overlap_words=max(args.stride, 0),
                min_score=args.min_score,
            )
            for span in spans:
                fout.write(json.dumps(span) + "\n")
                written += 1

            if torch.cuda.is_available() and (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    print(f"[rank={rank}] wrote {written} spans to {shard_out}")

    if args.dist:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            final_out = Path(args.out_jsonl)
            with final_out.open("w") as w:
                for s in range(world):
                    p = Path(f"{args.out_jsonl}.part{s:02d}")
                    if p.exists():
                        with p.open("r") as r:
                            shutil.copyfileobj(r, w)
            print(f"[rank=0] merged {world} parts into {final_out}")
        dist.barrier()

if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)