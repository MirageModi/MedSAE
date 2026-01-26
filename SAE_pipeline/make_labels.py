#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import multiprocessing
import os
from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm


def load_token_spans(
    path: str,
    entity_field: str,
    entity_values: Optional[Iterable[str]],
) -> Dict[str, Set[int]]:
    """
    Load aligned spans (JSONL/JSON) and return doc_id -> set(token_index).
    
    The NER file uses ORIGINAL (unshuffled) doc_ids from the source CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"NER token file not found: {path}")

    entity_values_set = None
    if entity_values:
        entity_values_set = {v.strip() for v in entity_values if v.strip()}

    doc_tokens: Dict[str, Set[int]] = defaultdict(set)

    def _handle_span(span: Dict) -> None:
        doc_id = span.get("doc_id")
        if doc_id is None:
            return
        if entity_values_set:
            field_value = span.get(entity_field)
            if field_value not in entity_values_set:
                return
        positions = span.get("token_positions", [])
        for pos in positions:
            try:
                idx = int(pos)
            except (TypeError, ValueError):
                continue
            doc_tokens[str(doc_id)].add(idx)

    print(f"[labels] Loading NER spans from {path}...")
    if path.endswith(".jsonl"):
        total_lines = None
        try:
            total_lines = sum(1 for _ in open(path))
        except: pass
        
        with open(path, "r") as f:
            for line in tqdm(f, total=total_lines, desc="Reading Spans", unit="line"):
                line = line.strip()
                if not line:
                    continue
                span = json.loads(line)
                _handle_span(span)
    else:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for span in data:
                _handle_span(span)
        elif isinstance(data, dict):
            for span in data.get("spans", []):
                _handle_span(span)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")

    return doc_tokens


def list_activation_files(root: str, split: str, max_files: Optional[int]) -> List[str]:
    patt1 = os.path.join(root, "shard_*", split, "batch_*.pt")
    patt2 = os.path.join(root, "shard_*", split, "rank*_batch_*.pt")
    files = sorted(glob.glob(patt1) + glob.glob(patt2))
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No activation batches found under {root}/shard_*/{split}")
    return files


def process_batch_file(
    file_path: str, 
    doc_token_map: Dict[str, Set[int]],
) -> Tuple[str, np.ndarray, dict]:
    """
    Worker function to process a single batch file.
    Returns (basename, label_array, stats_dict).
    
    KEY FIX: Uses 'doc_id' field from batch file directly, which contains
    the CORRECT original row index (same approach as fix_topk_char_offsets_v2.py).
    
    Args:
        file_path: Path to activation batch
        doc_token_map: NER spans indexed by (original) doc_id
    """
    stats = {"processed": 0, "matched": 0, "no_doc_id": 0, "doc_not_in_ner": 0, "positives": 0}
    
    try:
        batch = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # Required keys for label assignment
        required_keys = {"input_ids", "attention_mask", "token_start"}
        
        if not all(k in batch for k in required_keys):
            missing = required_keys.difference(batch.keys())
            print(f"[WARNING] {file_path} missing keys: {missing}")
            return (os.path.basename(file_path), None, stats)

        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        token_start = batch["token_start"]
        
        # KEY FIX: Get doc_id directly from batch file
        # This is the CORRECT original row index, saved during activation generation
        doc_ids = batch.get("doc_id", None)
        
        if doc_ids is None:
            # Fallback to global_row_index if doc_id not available
            # (This should not happen with properly generated activations)
            doc_ids = batch.get("global_row_index", None)
            if doc_ids is not None:
                print(f"[WARNING] {file_path}: Using global_row_index as fallback (doc_id not found)")

        B, T = input_ids.shape
        labels = np.zeros((B, T), dtype=np.float32)

        for b in range(B):
            stats["processed"] += 1
            
            # Get doc_id from batch (CORRECT approach)
            if doc_ids is not None:
                if isinstance(doc_ids, torch.Tensor):
                    doc_id = int(doc_ids[b].item())
                elif isinstance(doc_ids, list):
                    doc_id = int(doc_ids[b])
                else:
                    doc_id = int(doc_ids)
            else:
                stats["no_doc_id"] += 1
                continue
            
            doc_id_str = str(doc_id)
            
            # Look up NER spans for this document
            positives = doc_token_map.get(doc_id_str)
            
            if not positives:
                stats["doc_not_in_ner"] += 1
                continue
            
            stats["matched"] += 1

            # Get token_start for this window (global token offset)
            if isinstance(token_start, torch.Tensor):
                start_global = int(token_start[b].item())
            else:
                start_global = int(token_start[b]) if isinstance(token_start, list) else int(token_start)
            
            valid_mask = (attn_mask[b].numpy() != 0)
            
            for t in range(T):
                if valid_mask[t]:
                    global_token_idx = start_global + t
                    if global_token_idx in positives:
                        labels[b, t] = 1.0
                        stats["positives"] += 1
                        
        return (os.path.basename(file_path), labels, stats)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return (os.path.basename(file_path), None, stats)


def main():
    ap = argparse.ArgumentParser(description="Create token-level labels (Fixed v2 - uses doc_id from batch)")
    ap.add_argument("--acts_root", required=True, help="Root directory containing shard_XX/<split>/batch_*.pt")
    ap.add_argument("--acts_split", default="val", help="Activation split to label (default: val)")
    ap.add_argument("--max_files", type=int, default=0, help="If >0, limit number of activation batches processed.")
    ap.add_argument("--ner_tokens_jsonl", required=True, help="Aligned NER spans with token_positions")
    ap.add_argument("--entity_field", default="entity_group", help="Span field to filter on (default: entity_group)")
    ap.add_argument(
        "--entity_values",
        default="",
        help="Comma-separated entity values to keep. If empty, all spans are treated as positives.",
    )
    ap.add_argument("--out_npz", required=True, help="Output NPZ path for token labels")
    ap.add_argument("--num_workers", type=int, default=16, help="Number of worker processes")
    
    # Legacy arguments (kept for backward compatibility but not used)
    ap.add_argument("--source_csv", default="", help="[DEPRECATED] Not needed - doc_id comes from batch files")
    ap.add_argument("--max_rows", type=int, default=50000, help="[DEPRECATED] Not needed")
    ap.add_argument("--shuffle_seed", type=int, default=42, help="[DEPRECATED] Not needed")
    ap.add_argument("--no_shuffle_fix", action="store_true", help="[DEPRECATED] Not needed")

    args = ap.parse_args()
    
    # Warn about deprecated arguments
    if args.source_csv:
        print("[INFO] --source_csv is deprecated in v2. Doc IDs are read directly from batch files.")
    
    # 1. Load Spans (Main Process)
    entity_values = [v for v in args.entity_values.split(",") if v.strip()]
    doc_token_map = load_token_spans(args.ner_tokens_jsonl, args.entity_field, entity_values or None)
    print(f"[labels] Loaded token sets for {len(doc_token_map)} documents.")
    
    # Show sample doc_ids from NER
    sample_ner_docs = list(doc_token_map.keys())[:5]
    print(f"[labels] Sample NER doc_ids: {sample_ner_docs}")

    # 2. List Files
    files = list_activation_files(args.acts_root, args.acts_split, args.max_files if args.max_files > 0 else None)
    print(f"[labels] Processing {len(files)} activation batches...")
    
    # Quick check: peek at first batch to see doc_id format
    first_batch = torch.load(files[0], map_location="cpu", weights_only=False)
    if "doc_id" in first_batch:
        sample_doc_id = first_batch["doc_id"]
        if isinstance(sample_doc_id, torch.Tensor):
            sample_doc_id = sample_doc_id[0].item()
        elif isinstance(sample_doc_id, list):
            sample_doc_id = sample_doc_id[0]
        print(f"[labels] Sample batch doc_id: {sample_doc_id} (type: {type(sample_doc_id).__name__})")
    else:
        print("[WARNING] First batch does not have 'doc_id' field!")

    # 3. Parallel Processing
    labels_map: Dict[str, np.ndarray] = {}
    total_stats = {"processed": 0, "matched": 0, "no_doc_id": 0, "doc_not_in_ner": 0, "positives": 0}
    
    num_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    num_workers = max(1, num_workers - 1)

    print(f"[labels] Starting pool with {num_workers} workers...")
    
    worker_func = partial(process_batch_file, doc_token_map=doc_token_map)

    with multiprocessing.Pool(num_workers) as pool:
        for basename, label_array, stats in tqdm(
            pool.imap_unordered(worker_func, files, chunksize=10),
            total=len(files),
            desc="Generating Labels",
            unit="batch"
        ):
            if label_array is not None:
                labels_map[basename] = label_array
            for k, v in stats.items():
                total_stats[k] += v

    # 4. Save
    print(f"[labels] Saving compressed NPZ...")
    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    np.savez_compressed(args.out_npz, **labels_map)
    print(f"[labels] Wrote token labels to {args.out_npz} ({len(labels_map)} batches)")
    
    # Summary statistics
    total_tokens = 0
    total_positives = 0
    for arr in labels_map.values():
        total_tokens += arr.size
        total_positives += arr.sum()
    
    print(f"\n[labels] {'='*60}")
    print(f"[labels] SUMMARY")
    print(f"[labels] {'='*60}")
    print(f"  Batches processed: {len(labels_map)}")
    print(f"  Windows processed: {total_stats['processed']:,}")
    print(f"  Windows matched to NER: {total_stats['matched']:,} ({100*total_stats['matched']/max(1,total_stats['processed']):.1f}%)")
    print(f"  Windows with no doc_id: {total_stats['no_doc_id']:,}")
    print(f"  Windows doc not in NER: {total_stats['doc_not_in_ner']:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Positive labels: {int(total_positives):,}")
    print(f"  Positive rate: {100*total_positives/max(1,total_tokens):.4f}%")
    
    if total_stats['matched'] == 0:
        print("\n[WARNING] Zero windows matched! Possible issues:")
        print("  1. doc_id format mismatch between batch files and NER")
        print("  2. NER was run on different documents than activations")
        print("  3. doc_id field missing from batch files")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()