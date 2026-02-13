#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import math
import heapq
import argparse
import re
import glob
import random
import time
import csv
from concurrent.futures import ThreadPoolExecutor
import gc
from typing import Any, Optional, List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from accelerate import infer_auto_device_map
except ImportError:
    print("Please install required packages: pip install torch pandas transformers accelerate")
    exit(1)
class CharOffsetComputer:
    """
    Computes character offsets for tokens by re-tokenizing the text.
    Used when saved char_offsets are missing or all zeros.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        if tokenizer.pad_token_id is not None:
            self.special_ids.add(tokenizer.pad_token_id)
    
    def compute_offsets(self, text: str, input_ids: list) -> list:
        """
        Compute (char_start, char_end) for each token in input_ids.
        
        Args:
            text: The original text string
            input_ids: List of token IDs from the model
        
        Returns:
            List of (char_start, char_end) tuples, one per token
        """
        if not text:
            return [(0, 0)] * len(input_ids)
        
        # Tokenize with offset mapping
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            display_ids = encoded["input_ids"]
            offset_mapping = encoded["offset_mapping"]
        except Exception:
            return [(0, 0)] * len(input_ids)
        
        offsets = []
        disp_ptr = 0
        
        for tid in input_ids:
            if tid in self.special_ids:
                # Special token - no character span
                offsets.append((0, 0))
            elif disp_ptr < len(display_ids) and tid == display_ids[disp_ptr]:
                # Match - use the offset
                cs, ce = offset_mapping[disp_ptr]
                offsets.append((int(cs), int(ce)))
                disp_ptr += 1
            else:
                # Try to find the token in remaining display_ids (lookahead)
                found = False
                for j in range(disp_ptr, min(disp_ptr + 5, len(display_ids))):
                    if tid == display_ids[j]:
                        cs, ce = offset_mapping[j]
                        offsets.append((int(cs), int(ce)))
                        disp_ptr = j + 1
                        found = True
                        break
                
                if not found:
                    # Couldn't find match - use (0, 0)
                    offsets.append((0, 0))
        
        return offsets

class SpanCache:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            self.special_ids.add(pad_id)
        self._batch_offsets: List[List[Tuple[int, int]]] = []
        self._batch_model2disp: List[List[Optional[int]]] = []
        self._batch_disp_ids = []
        self._batch_disp_tokens = []
        self._prepared = False

    def clear(self):
        self._batch_offsets = []
        self._batch_model2disp = []
        self._batch_disp_ids = []
        self._batch_disp_tokens = []
        self._prepared = False

    def prepare_batch(self, texts: List[str], input_ids: List[List[int]]):
        self.clear()
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        self._batch_offsets = encoded["offset_mapping"]
        self._batch_disp_ids = encoded["input_ids"]
        self._batch_disp_tokens = [
            self.tokenizer.convert_ids_to_tokens(ids) for ids in self._batch_disp_ids
        ]
        self._batch_model2disp = [self._model_to_display_map(ids) for ids in input_ids]
        self._prepared = True

    def display_tokens(self, b: int) -> List[str]:
        return self._batch_disp_tokens[b]

    def aggregate_acts_to_display(self, b: int, z_b, agg: str = "max") -> List[float]:
        disp_map = self._batch_model2disp[b]
        n_disp = len(self._batch_offsets[b])
        out = [0.0] * n_disp
        vals = z_b.detach().float().cpu().tolist() if isinstance(z_b, torch.Tensor) else [float(v) for v in z_b]
        if agg == "sum":
            for t, v in enumerate(vals):
                j = disp_map[t]
                if j is not None:
                    out[j] += float(v)
        else:
            for t, v in enumerate(vals):
                j = disp_map[t]
                if j is not None and float(v) > out[j]:
                    out[j] = float(v)
        return out

    def _model_to_display_map(self, ids_row: List[int]) -> List[Optional[int]]:
        n = len(ids_row)
        non_special = [0 if tid in self.special_ids else 1 for tid in ids_row]
        ps = [0] * (n + 1)
        for i in range(n):
            ps[i + 1] = ps[i] + non_special[i]
        right = [None] * n
        last = None
        for i in range(n - 1, -1, -1):
            if non_special[i]:
                last = i
            right[i] = last
        left = [None] * n
        last = None
        for i in range(n):
            if non_special[i]:
                last = i
            left[i] = last
        model2disp: List[Optional[int]] = [None] * n
        for i in range(n):
            if non_special[i]:
                model2disp[i] = ps[i + 1] - 1
            else:
                j = right[i] if right[i] is not None else left[i]
                model2disp[i] = (ps[j + 1] - 1) if j is not None else None
        return model2disp

    def span_for(self, b: int, model_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._prepared:
            raise RuntimeError("SpanCache.prepare_batch() must be called first.")
        model2disp = self._batch_model2disp[b]
        disp_idx = None
        if 0 <= model_idx < len(model2disp):
            disp_idx = model2disp[model_idx]
        if disp_idx is None:
            return None, None, None
        offsets = self._batch_offsets[b]
        if not (0 <= disp_idx < len(offsets)):
            return None, None, disp_idx
        cs, ce = offsets[disp_idx]
        return int(cs), int(ce), int(disp_idx)

# ---------------------------
# export_topk_latents_csv
# ---------------------------

def export_topk_latents_csv(
    sae_state_dict, 
    cfg_in_ckpt, 
    tokenizer, 
    all_val_data, 
    args, 
    export_path, 
    filter_terms=None, 
    with_offsets=False, 
    feature_scale=None,
    global_batch_offset=0,  # NEW: Offset to convert local idx to global batch_idx
):
    """
    Export Top-K latent IDs and values per token to CSV.
    """
    k_main = cfg_in_ckpt.get("k", 128)
    relu_after_topk = cfg_in_ckpt.get("relu_after_topk", True)
    layer_index = args.layer_index
    model_id = args.model_id
    
    target_lemmas = set()
    if filter_terms:
        for term in filter_terms:
            target_lemmas.add(term.lower())
    
    device = all_val_data[0]["acts"].device
    dtype = all_val_data[0]["acts"].dtype
    
    W_enc = sae_state_dict['W_enc'].to(device=device, dtype=dtype)
    b_enc = sae_state_dict.get('b_enc', torch.zeros(W_enc.shape[0], device=device, dtype=dtype))
    
    scale_vals_gpu = None
    if feature_scale is not None:
        scale_vals_gpu = feature_scale.view(-1).to(device=device, dtype=dtype).clamp_min(1e-6)
    
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    
    offset_computer = CharOffsetComputer(tokenizer) if with_offsets else None
    
    rows = []
    pbar = tqdm(total=len(all_val_data), desc="[TopK Export] Processing batches", unit="batch", leave=False)
    
    for local_idx, val_batch in enumerate(all_val_data):
        global_batch_idx = global_batch_offset + local_idx
        
        acts = val_batch['acts'].to(device=device, dtype=dtype)
        mask_bt = val_batch['mask_bt']
        texts = val_batch.get('texts', [])
        input_ids_batch = val_batch.get('input_ids', [])
        doc_ids = val_batch.get('doc_id', None)
        
        if isinstance(input_ids_batch, torch.Tensor):
            input_ids_list = input_ids_batch.tolist()
        else:
            input_ids_list = list(input_ids_batch) if input_ids_batch is not None else []
        
        if not texts and input_ids_list:
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_list]
        
        char_offsets_batch = val_batch.get('char_offsets', None)
        
        needs_recompute = False
        if with_offsets:
            if char_offsets_batch is None:
                needs_recompute = True
            else:
                all_zeros = True
                for co_list in char_offsets_batch:
                    if co_list:
                        for (cs, ce) in co_list:
                            if cs != 0 or ce != 0:
                                all_zeros = False
                                break
                    if not all_zeros:
                        break
                if all_zeros:
                    needs_recompute = True
        
        if needs_recompute and offset_computer is not None:
            char_offsets_batch = []
            B = len(input_ids_list)
            for b in range(B):
                text_b = texts[b] if b < len(texts) else ""
                ids_b = input_ids_list[b] if b < len(input_ids_list) else []
                offsets_b = offset_computer.compute_offsets(text_b, ids_b)
                char_offsets_batch.append(offsets_b)
        
        pre = torch.einsum("btd,nd->btn", acts, W_enc) + b_enc
        k_eff = min(int(k_main), pre.shape[-1])
        topk_vals, topk_idx = torch.topk(pre, k=k_eff, dim=-1)
        z_active = F.relu(topk_vals) if relu_after_topk else topk_vals
        
        topk_vals_cpu = z_active.float().cpu()
        topk_idx_cpu = topk_idx.cpu()
        
        B, T = acts.shape[:2]
        
        for b in range(B):
            doc_id = None
            if doc_ids is not None:
                if isinstance(doc_ids, torch.Tensor):
                    doc_id = doc_ids[b].item() if b < doc_ids.shape[0] else None
                elif isinstance(doc_ids, list):
                    doc_id = doc_ids[b] if b < len(doc_ids) else None
                else:
                    doc_id = doc_ids
            if doc_id is None:
                doc_id = f"batch{global_batch_idx}_b{b}"
            
            input_ids_b = input_ids_list[b] if b < len(input_ids_list) else []
            char_offsets_b = char_offsets_batch[b] if char_offsets_batch and b < len(char_offsets_batch) else None
            
            for t in range(T):
                if not mask_bt[b, t]:
                    continue
                
                token_id = int(input_ids_b[t]) if t < len(input_ids_b) else 0
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                lemma = token_str.lower().strip().rstrip(".,;:!?")
                
                if target_lemmas:
                    matched = False
                    if lemma in target_lemmas:
                        matched = True
                    else:
                        for term in target_lemmas:
                            if term in lemma or lemma in term:
                                matched = True
                                break
                    if not matched:
                        continue
                
                t_vals = topk_vals_cpu[b, t].tolist()
                t_ids = topk_idx_cpu[b, t].tolist()
                
                row = {
                    "doc_id": doc_id,
                    "layer": layer_index,
                    "model": model_id,
                    "token_str": token_str,
                    "lemma": lemma,
                    "position": t,
                    "batch_idx": global_batch_idx,
                    "token_id": token_id,
                    "K": k_eff,
                    "latent_ids_json": json.dumps(t_ids),
                    "z_values_json": json.dumps(t_vals)
                }
                
                if scale_vals_gpu is not None:
                    scales = scale_vals_gpu[t_ids].float().cpu().tolist()
                    z_norm = [(v/s if s > 0 else 0.0) for v, s in zip(t_vals, scales)]
                    row["z_values_norm_json"] = json.dumps([max(0.0, min(1.0, z)) for z in z_norm])
                
                if with_offsets:
                    if char_offsets_b and t < len(char_offsets_b):
                        char_start, char_end = char_offsets_b[t]
                        row["char_start"] = char_start
                        row["char_end"] = char_end
                    else:
                        row["char_start"] = 0
                        row["char_end"] = 0
                
                rows.append(row)
        
        pbar.update(1)
        pbar.set_postfix({"rows": len(rows)})
        del acts, pre, topk_vals, topk_idx, z_active
        torch.cuda.empty_cache()
    
    pbar.close()
    
    if rows:
        fieldnames = [
            "doc_id", "layer", "model", "token_str", "lemma", "position", 
            "batch_idx", "token_id", "K", "latent_ids_json", "z_values_json"
        ]
        if feature_scale is not None:
            fieldnames.append("z_values_norm_json")
        if with_offsets:
            fieldnames.extend(["char_start", "char_end"])
        
        with open(export_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        
        # Print statistics about char offsets
        if with_offsets:
            n_valid_offsets = sum(1 for r in rows if r.get("char_start", 0) > 0 or r.get("char_end", 0) > 0)
            print(f"Exported {len(rows)} rows to {export_path}")
            print(f"  - Valid char offsets: {n_valid_offsets}/{len(rows)} ({100*n_valid_offsets/len(rows):.1f}%)")
        else:
            print(f"Exported {len(rows)} rows to {export_path}")
    else:
        print(f"Warning: No rows exported (filter may be too restrictive)")

def setup_logging(shard_id: int, total_shards: int):
    """Setup logging with shard identification"""
    log_format = f"[Shard {shard_id}/{total_shards}] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def str2dtype(s: str) -> torch.dtype:
    return dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[s]

def finite_or_none(x):
    try:
        xf = float(x)
    except Exception:
        return None
    return None if (math.isnan(xf) or math.isinf(xf)) else xf

def sanitize_scores_list(values):
    out = []
    for v in values:
        try:
            vf = float(v)
        except Exception:
            out.append(0.0)
            continue
        out.append(0.0 if (math.isnan(vf) or math.isinf(vf)) else vf)
    return out

def l2_normalize_per_token(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Explicit float32 cast for safety during normalization
    orig_dtype = x.dtype
    x_f = x.float()
    x_f = x_f - x_f.mean(dim=-1, keepdim=True)
    denom = x_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (x_f / denom).to(orig_dtype)

def center_unit_norm(x: torch.Tensor, mask_bt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    x_f = x_f - x_f.mean(dim=-1, keepdim=True)
    denom = x_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    if mask_bt is not None:
        mask_f = mask_bt.unsqueeze(-1).to(dtype=torch.float32)
        denom = denom * mask_f + (1.0 - mask_f)
    return torch.nan_to_num(x_f / denom, nan=0.0, posinf=0.0, neginf=0.0).to(orig_dtype)

@torch.no_grad()
def looks_normalized(x: torch.Tensor, mask_bt: torch.Tensor, tol: float = 0.05) -> bool:
    if mask_bt.sum() < 32:
        return False
    v = (x.float() - x.float().mean(dim=-1, keepdim=True)).norm(dim=-1)
    vals = v[mask_bt].flatten()
    med = torch.median(vals).item()
    return abs(med - 1.0) <= tol

def is_sharded_hf_model(m) -> bool:
    return hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict)

def route_inputs_for_hf(model, ids, att):
    """
    Route inputs to the same device as the model's embedding layer
    to avoid 'Expected all tensors to be on the same device' errors.
    """
    target_dev = None
    
    try:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            target_dev = model.model.embed_tokens.weight.device
        elif hasattr(model, "get_input_embeddings"):
            target_dev = model.get_input_embeddings().weight.device
    except Exception:
        pass

    if target_dev is None:
        try:
            target_dev = next(model.parameters()).device
        except Exception:
            target_dev = torch.device("cpu")

    # 3. Safety: If device is 'meta' (accelerate init), keep on CPU
    if target_dev.type == "meta":
        target_dev = torch.device("cpu")

    # 4. Move tensors
    if ids.device != target_dev:
        ids = ids.to(target_dev, non_blocking=True)
    if att.device != target_dev:
        att = att.to(target_dev, non_blocking=True)
        
    return ids, att

def module_device(mod):
    for p in mod.parameters():
        if p.device.type == "cuda":
            return p.device
    return torch.device("cpu")

def _get_state_tensor(state_dict, name: str):
    for prefix in ("", "module."):
        key = prefix + name
        if key in state_dict:
            return state_dict[key]
    return None

def _find_decoder_layers_module(model):
    candidates = ["model.layers", "model.decoder.layers", "transformer.h", "transformer.layers", "layers"]
    for path in candidates:
        mod = model
        ok = True
        for name in path.split("."):
            if not hasattr(mod, name):
                ok = False; break
            mod = getattr(mod, name)
        if ok:
            try: _ = mod[0]
            except Exception: continue
            return mod
    raise RuntimeError(f"Could not locate decoder layers (tried: {candidates}).")

@torch.no_grad()
def _get_layer_acts_hooked(model, input_ids, attention_mask, layer_index: int, normalize: bool = True):
    layers = _find_decoder_layers_module(model)
    if not (0 <= layer_index < len(layers)):
        raise IndexError(f"layer_index {layer_index} out of range (have {len(layers)})")
    target = layers[layer_index]
    cached = {"acts": None}
    
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        cached["acts"] = h.detach().float().contiguous().to("cpu")
    
    handle = target.register_forward_hook(hook)
    try:
        input_ids, attention_mask = route_inputs_for_hf(model, input_ids, attention_mask)
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    
    acts = cached["acts"]
    if acts is None:
        raise RuntimeError("Activation hook did not fire; model/layer path may differ.")
    
    if normalize:
        acts = l2_normalize_per_token(acts)
    
    return acts

@torch.no_grad()
def llm_acts_in_chunks(model,
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       layer_index: int,
                       normalize: bool,
                       chunk_bsz: int,
                       to_device: torch.device,
                       to_dtype: torch.dtype):
    if chunk_bsz is None or chunk_bsz <= 0:
        chunk_bsz = input_ids.size(0)
    outs_acts, outs_masks = [], []
    B = input_ids.size(0)
    use_hook = True
    for s in range(0, B, chunk_bsz):
        ids_chunk = input_ids[s:s+chunk_bsz]
        att_chunk = attention_mask[s:s+chunk_bsz]
        
        ids, att = route_inputs_for_hf(model, ids_chunk, att_chunk)
        
        try:
            if use_hook:
                acts = _get_layer_acts_hooked(model, ids, att, layer_index, normalize=normalize)
            else:
                out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                idx = max(0, min(layer_index + 1, len(hs) - 1))
                acts = hs[idx]
                if normalize:
                    acts = l2_normalize_per_token(acts)
        except Exception:
            if use_hook:
                use_hook = False
                out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                idx = max(0, min(layer_index + 1, len(hs) - 1))
                acts = hs[idx]
                if normalize:
                    acts = l2_normalize_per_token(acts)
            else:
                raise
        
        outs_acts.append(acts.to(device=to_device, dtype=to_dtype, non_blocking=True))
        outs_masks.append(att_chunk.to(device=to_device, dtype=torch.bool, non_blocking=True))
        del ids, att, acts
        torch.cuda.empty_cache()
    acts_all  = torch.cat(outs_acts,  dim=0) if len(outs_acts)  > 1 else outs_acts[0]
    masks_all = torch.cat(outs_masks, dim=0) if len(outs_masks) > 1 else outs_masks[0]
    return acts_all, masks_all

@torch.no_grad()
def iter_layer_token_reps(llm, tokenizer, token_ids: List[int], layer_index: int, batch_size: int,
                          device: torch.device, dtype: torch.dtype, add_bos: bool) -> Tuple[List[int], torch.Tensor]:
    bos_id = tokenizer.bos_token_id
    if bos_id is None: bos_id = tokenizer.eos_token_id
    if bos_id is None: bos_id = tokenizer.pad_token_id
    for s in range(0, len(token_ids), batch_size):
        ids_batch = token_ids[s:s+batch_size]
        if add_bos and bos_id is not None:
            seq = [[bos_id, tid] for tid in ids_batch]
        else:
            seq = [[tid] for tid in ids_batch]
        input_ids = torch.tensor(seq, dtype=torch.long)
        attn = torch.ones_like(input_ids, dtype=torch.long)
        acts, _mask = llm_acts_in_chunks(
            llm, input_ids=input_ids, attention_mask=attn,
            layer_index=layer_index, normalize=False, chunk_bsz=None,
            to_device=device, to_dtype=dtype
        )
        reps = acts[:, -1, :].float()
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        yield ids_batch, reps

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def get_local_model_path(model_id: str) -> str:
    """Map HuggingFace model IDs to local paths if models are available locally.
    
    Checks environment variables first (e.g., MEDGEMMA_PATH, OPENBIOLLM_PATH),
    then falls back to model_id for HuggingFace Hub download.
    """
    env_map = {
        "google/medgemma-27b-text-it": os.environ.get("MEDGEMMA_PATH"),
        "aaditya/OpenBioLLM-Llama3-70B": os.environ.get("OPENBIOLLM_PATH"),
    }
    local_path = env_map.get(model_id)
    if local_path and os.path.exists(local_path):
        return local_path
    return model_id

def get_tokenizer_only(model_id: str, ctx_len: int = 64, revision: str = None):
    local_model_path = get_local_model_path(model_id)
    tok = AutoTokenizer.from_pretrained(local_model_path, revision=revision, use_fast=True)
    tok.padding_side = "left"
    tok.model_max_length = ctx_len
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def get_model_only(model_id: str, dtype=torch.bfloat16, offload_dir="hf_offload", revision=None):
    local_model_path = get_local_model_path(model_id)
    os.makedirs(offload_dir, exist_ok=True)
    cfg = AutoConfig.from_pretrained(local_model_path, revision=revision)
    cfg.attn_implementation = "sdpa"
    lm = AutoModelForCausalLM.from_pretrained(
        local_model_path, revision=revision, dtype=dtype,
        device_map="auto",
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
        config=cfg,
    )
    if hasattr(lm, "tie_weights"): lm.tie_weights()
    lm.config.use_cache = False
    return lm

# ---------------------------
# Heaps & Utils
# ---------------------------

def _update_heap_maxk(heap: List[Tuple], k: int, score: float, item: any, unique_id: Optional[int] = None):
    heap_item = (score, unique_id, item) if unique_id is not None else (score, item)
    if len(heap) < k:
        heapq.heappush(heap, heap_item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, heap_item)

def _sae_params_to_x(sae_W_enc, sae_b_enc, sae_W_dec, sae_b_dec, x: torch.Tensor):
    dev, dt = x.device, x.dtype
    W_enc = sae_W_enc.to(dev, dt)
    W_dec = sae_W_dec.to(dev, dt)
    b_enc = sae_b_enc.to(dev, dt) if sae_b_enc is not None else torch.zeros(W_enc.shape[0], device=dev, dtype=dt)
    b_dec = sae_b_dec.to(dev, dt) if sae_b_dec is not None else None
    return W_enc, b_enc, W_dec, b_dec

def _finalize_heap_desc(heap: List[Tuple]) -> List[any]:
    return [item[-1] for item in sorted(heap, key=lambda x: x[0], reverse=True)]

def _loss_shifted_ce(logits, input_ids, attn_mask):
    device = logits.device
    labels = input_ids[:, 1:].to(device)
    logits_sliced = logits[:, :-1, :]
    mask = (attn_mask[:, 1:].float() if attn_mask.dtype != torch.float else attn_mask[:, 1:]).to(device)
    
    B_logits, T_logits, V = logits_sliced.shape
    B_labels, T_labels = labels.shape
    B_mask, T_mask = mask.shape
    
    min_T = min(T_logits, T_labels, T_mask)
    if min_T < max(T_logits, T_labels, T_mask):
        logits_sliced = logits_sliced[:, :min_T, :]
        labels = labels[:, :min_T]
        mask = mask[:, :min_T]
    
    logits = logits_sliced.reshape(-1, V)
    labels = labels.reshape(-1)
    mask   = mask.reshape(-1)
    loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
    if (labels == -100).any():
        keep = (labels != -100).float()
        return (loss * keep).sum() / keep.sum().clamp_min(1.0)
    else:
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

def _kl_shifted(logits_base, logits_swap, mask=None, v_chunk: int = 65536):
    p = logits_base[:, :-1, :]
    q = logits_swap[:, :-1, :]
    B, Tm1, V = p.shape
    if not v_chunk or v_chunk <= 0:
        target_bytes = 96 * 1024 * 1024
        v_chunk = max(8192, (target_bytes // (B * Tm1 * 4)) if B*Tm1 else 65536)

    p_max = torch.full((B, Tm1, 1), -float("inf"), device=p.device, dtype=p.dtype)
    q_max = torch.full_like(p_max, -float("inf"))
    for s in range(0, V, v_chunk):
        p_max = torch.maximum(p_max, p[..., s:s+v_chunk].max(dim=-1, keepdim=True).values)
        q_max = torch.maximum(q_max, q[..., s:s+v_chunk].max(dim=-1, keepdim=True).values)

    sum_exp_p = torch.zeros((B, Tm1, 1), device=p.device, dtype=torch.float32)
    sum_exp_q = torch.zeros_like(sum_exp_p)
    num_Ep_p  = torch.zeros_like(sum_exp_p)
    num_Ep_q  = torch.zeros_like(sum_exp_p)

    for s in range(0, V, v_chunk):
        p_chunk = (p[..., s:s+v_chunk] - p_max).float()
        q_chunk = (q[..., s:s+v_chunk] - q_max).float()
        exp_p = torch.exp(p_chunk)
        sum_exp_p += exp_p.sum(dim=-1, keepdim=True)
        sum_exp_q += torch.exp(q_chunk).sum(dim=-1, keepdim=True)
        num_Ep_p += (exp_p * p[..., s:s+v_chunk].float()).sum(dim=-1, keepdim=True)
        num_Ep_q += (exp_p * q[..., s:s+v_chunk].float()).sum(dim=-1, keepdim=True)
        del p_chunk, q_chunk, exp_p
        torch.cuda.empty_cache()

    lse_p = p_max.float() + torch.log(sum_exp_p.clamp_min(1e-30))
    lse_q = q_max.float() + torch.log(sum_exp_q.clamp_min(1e-30))
    Ep_p  = num_Ep_p / sum_exp_p.clamp_min(1e-30)
    Ep_q  = num_Ep_q / sum_exp_p.clamp_min(1e-30)

    kl_bt = (lse_q - lse_p + (Ep_p - Ep_q)).squeeze(-1)
    if mask is not None:
        m = mask
        if m.device != kl_bt.device:
            m = m.to(kl_bt.device, non_blocking=True)
        if m.ndim == 3:
            m = m.squeeze(-1)
        m = m[:, :kl_bt.shape[1]]
        if m.dtype != torch.float32:
            m = m.float()
        return (kl_bt * m).sum() / m.sum().clamp_min(1e-12)
    return kl_bt.mean()

def _get_llm_device_and_dtype(llm):
    device, dtype = None, None
    if hasattr(llm, "hf_device_map") and isinstance(llm.hf_device_map, dict):
        for key, value in llm.hf_device_map.items():
            if isinstance(value, (int, torch.device)):
                device = torch.device(f"cuda:{value}") if isinstance(value, int) else value
                break
    if device is None:
        try:
            param = next(llm.parameters())
            device = param.device
            dtype = param.dtype
        except Exception:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16
    if dtype is None: dtype = torch.bfloat16
    return device, dtype

def _iter_batches_for_delta_ce(args, llm, val_loader, acts_root, acts_split, acts_max_files):
    device, dtype = _get_llm_device_and_dtype(llm)
    micro_bsz = getattr(args, "llm_micro_bsz", 8)
    def _recompute(ids, attn):
        return llm_acts_in_chunks(llm, ids, attn, args.layer_index, False, micro_bsz, device, dtype)

    if val_loader is not None:
        for batch in val_loader:
            ids = batch["input_ids"]
            attn = batch.get("attention_mask", (ids!=0).long())
            acts, mask_bt = _recompute(ids, attn)
            if acts.numel() == 0: continue
            yield {"input_ids": ids, "attention_mask": attn.long(), "acts": acts, "mask_bt": mask_bt}
    else:
        saved_ds = SavedActsBatches(acts_root, split=acts_split, max_files=acts_max_files)
        loader = torch.utils.data.DataLoader(saved_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_saved_batches)
        for batch in loader:
            ids = batch["input_ids"]
            mask_bt = batch["mask_bt"].bool()
            attn = mask_bt.long()
            acts, mask_bt_re = _recompute(ids, attn)
            if acts.numel() == 0: continue
            yield {"input_ids": ids, "attention_mask": attn.long(), "acts": acts, "mask_bt": mask_bt_re}

class ResidualSwap:
    def __init__(self, model, layer_index, new_resid, mask=None):
        self.model = model
        self.layer_index = layer_index
        self.new_resid = new_resid
        self.mask = mask
        self.handle = None

    def _post_hook(self, module, inputs, output):
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output
        if x.is_cuda: torch.cuda.synchronize(device=x.device)
        x = x.contiguous()
        
        Bx, Tx, H = x.shape
        new_resid_contig = self.new_resid.contiguous()
        
        if new_resid_contig.dim() != 3 or new_resid_contig.shape[0] != Bx: return output
        
        T_new = new_resid_contig.shape[1]
        if T_new > Tx: resid_swapped = new_resid_contig[:, :Tx, :]
        elif T_new < Tx: 
            pad = torch.zeros(Bx, Tx-T_new, H, device=x.device, dtype=x.dtype)
            resid_swapped = torch.cat([new_resid_contig, pad], dim=1)
        else: resid_swapped = new_resid_contig
        
        resid_swapped = resid_swapped.to(x.device, dtype=x.dtype)
        
        if self.mask is not None:
            m = self.mask.to(x.device).bool()
            if m.ndim == 2: m = m.unsqueeze(-1)
            if m.shape[1] > Tx: m = m[:, :Tx, :]
            elif m.shape[1] < Tx: 
                 pad = torch.zeros(Bx, Tx-m.shape[1], 1, device=x.device, dtype=torch.bool)
                 m = torch.cat([m, pad], dim=1)
            resid_swapped = torch.where(m, resid_swapped, x)
        
        if resid_swapped.is_cuda: torch.cuda.synchronize(device=resid_swapped.device)
        if is_tuple: return (resid_swapped, *output[1:])
        return resid_swapped

    def __enter__(self):
        layers = _find_decoder_layers_module(self.model)
        target = layers[self.layer_index]
        self.handle = target.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle: self.handle.remove()
        self.handle = None

@torch.no_grad()
def delta_ce_for_batch(model, sae_W_enc, sae_b_enc, sae_W_dec, sae_b_dec,
                       acts_bth, mask_bt, input_ids, alpha=1.0,
                       k_main=128, device=None, layer_index=None):
    if device is None:     device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    ids = input_ids.to(device)
    attn = mask_bt.to(device).long()
    m_bt_full = (attn > 0).bool()
    
    ids_base, attn_base = route_inputs_for_hf(model, ids, attn)
    out_base = model(input_ids=ids_base, attention_mask=attn_base, use_cache=False)
    logits_base = out_base.logits
    ce_base = _loss_shifted_ce(logits_base, ids, attn[:, 1:] > 0)
    
    x_raw = acts_bth.to(device, dtype=dtype)
    mu = x_raw.mean(dim=-1, keepdim=True)
    centered = x_raw - mu
    sigma = centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x_norm = centered / sigma

    W_enc, b_enc, W_dec, b_dec = _sae_params_to_x(sae_W_enc, sae_b_enc, sae_W_dec, sae_b_dec, x_norm)
    pre = torch.einsum("btd,nd->btn", x_norm, W_enc) + b_enc
    k_eff = min(int(k_main), pre.shape[-1])
    _, idx_top = torch.topk(pre, k=k_eff, dim=-1)
    mask_top = torch.zeros_like(pre).scatter_(-1, idx_top, 1.0)
    z = torch.relu(pre) * mask_top
    
    xh_norm = torch.einsum("btn,dn->btd", z, W_dec)
    if b_dec is not None: xh_norm += b_dec
    xh_raw = xh_norm * sigma + mu
    x_mix = (1.0 - float(alpha)) * x_raw + float(alpha) * xh_raw
    
    layers = _find_decoder_layers_module(model)
    target_dev = module_device(layers[layer_index])
    x_mix_target = x_mix.to(target_dev, non_blocking=True)
    m_target = m_bt_full.to(target_dev, non_blocking=True)
    if x_mix_target.is_cuda: torch.cuda.synchronize(device=target_dev)
    
    with ResidualSwap(model, layer_index, x_mix_target, m_target):
        out_swap = model(input_ids=ids_base, attention_mask=attn_base, use_cache=False)
        logits_swap = out_swap.logits
    
    ce_swap = _loss_shifted_ce(logits_swap, ids, attn[:, 1:] > 0)
    dce = (ce_swap - ce_base).detach()
    kl = _kl_shifted(logits_base, logits_swap, mask=(attn[:, 1:] > 0))
    
    return float(dce.item()), float(kl.item()), float(ce_base.item()), float(ce_swap.item())

class SavedActsBatches(torch.utils.data.Dataset):
    def __init__(self, acts_root: str, split: str = "val", max_files: int = 0, preload: bool = False):
        patt1 = os.path.join(acts_root, "shard_*", split, "batch_*.pt")
        patt2 = os.path.join(acts_root, "shard_*", split, "rank*_batch_*.pt")
        files = sorted(glob.glob(patt1) + glob.glob(patt2))
        if max_files and max_files > 0:
            random.seed(42)
            random.shuffle(files)
            files = files[:max_files]
            files.sort()
        if not files:
            raise FileNotFoundError(f"No activation batch files under {acts_root}/shard_XX/{split}")
        self.files = files
        self.preload = preload
        self.data = [None] * len(self.files)
        if self.preload:
            print(f"[SavedActsBatches] Preloading {len(self.files)} files into RAM...")
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = executor.map(lambda i: (i, self._load_disk(i)), range(len(self.files)))
                for i, content in results: self.data[i] = content
    def _load_disk(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        acts = d.get("hidden_states", d.get("activations", None))
        return {"acts": acts, "mask_bt": (d["attention_mask"]!=0), "input_ids": d["input_ids"], "char_offsets": d.get("char_offsets"), "doc_id": d.get("doc_id")}
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.data[idx] if self.preload else self._load_disk(idx)

def collate_saved_batches(batches):
    out = {
        "acts": torch.cat([b["acts"] for b in batches], dim=0),
        "mask_bt": torch.cat([b["mask_bt"] for b in batches], dim=0),
        "input_ids": torch.cat([b["input_ids"] for b in batches], dim=0),
    }
    
    if len(batches) > 0:
        if "doc_id" in batches[0] and batches[0]["doc_id"] is not None:
            val = batches[0]["doc_id"]
            if isinstance(val, torch.Tensor):
                out["doc_id"] = torch.cat([b["doc_id"] for b in batches], dim=0)
            elif isinstance(val, list):
                out["doc_id"] = [item for b in batches for item in b["doc_id"]]
        
        if "char_offsets" in batches[0] and batches[0]["char_offsets"] is not None:
            out["char_offsets"] = [item for b in batches for item in b["char_offsets"]]
    
    return out

def _decode_batch_texts(tokenizer, input_ids_batch):
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_batch]

def load_gen_meta(acts_root: str) -> Dict:
    metas = glob.glob(os.path.join(acts_root, "shard_*", "metadata.json"))
    if not metas: return {}
    with open(metas[0]) as f: return json.load(f)

def _load_token_labels_npz(labels_path, batch_files):
    import numpy as np
    data = np.load(labels_path)
    labels = []
    for f in batch_files:
        key = os.path.basename(f)
        if key not in data: raise KeyError(f"labels NPZ missing {key}")
        labels.append(torch.from_numpy(data[key]).float())
    return labels

class LineTextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.samples = []
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f: self.samples = [s.strip() for s in f if s.strip()]
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            s = obj.get("text") or obj.get("content") or obj.get("prompt")
                            if s: self.samples.append(s)
                        except json.JSONDecodeError: continue
        if not self.samples: raise ValueError("No samples loaded.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

class CSVClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, text_column, mode, strip_bhc_section=False, min_len=1):
        if pd is None: raise RuntimeError("pandas required")
        df = pd.read_csv(csv_path, low_memory=False)
        col = text_column or ("input" if mode == "bhc" else "text")
        if col not in df.columns: raise ValueError(f"Column '{col}' not found.")
        self.samples = []
        if strip_bhc_section:
            patt = r"(?im)^\s*(brief\s+hospital\s+course|hospital\s+course|bhc)\s*:?\s*$"
            nextp = r"(?im)^\s*[A-Z][A-Z0-9 /\-]{3,}\s*:?\s*$"
            heading_re, next_heading_re = re.compile(patt), re.compile(nextp)
        else:
            heading_re, next_heading_re = None, None
        for s in df[col].astype(str):
            s = s.strip()
            if not s: continue
            if strip_bhc_section and heading_re:
                s = _strip_section(s, heading_re, next_heading_re)
            if len(s) >= min_len:
                self.samples.append(s)
        if not self.samples: raise ValueError("No usable texts from CSV.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def _strip_section(text: str, heading_re: re.Pattern, next_heading_re: re.Pattern) -> str:
    if not text: return text
    m = heading_re.search(text)
    if not m: return text
    start = m.start()
    m2 = next_heading_re.search(text, pos=m.end())
    end = m2.start() if m2 else len(text)
    return text[:start] + text[end:]

def collate_to_token_windows(batch_texts, tokenizer, ctx_len, stride):
    ids_all, attn_all = [], []
    for text in batch_texts:
        enc = tokenizer(text, return_attention_mask=True)
        for start in range(0, len(enc['input_ids']), stride):
            ids = enc['input_ids'][start:start+ctx_len]
            att = enc['attention_mask'][start:start+ctx_len]
            if len(ids) < ctx_len:
                pad_len = ctx_len - len(ids)
                ids.extend([tokenizer.pad_token_id] * pad_len)
                att.extend([0] * pad_len)
            ids_all.append(ids)
            attn_all.append(att)
    if not ids_all:
        return {"input_ids": torch.empty(0, dtype=torch.long), "attention_mask": torch.empty(0, dtype=torch.long)}
    return {"input_ids": torch.tensor(ids_all, dtype=torch.long), "attention_mask": torch.tensor(attn_all, dtype=torch.long)}

@torch.no_grad()
def build_corpus_token_reps(all_val_data, vocab_size, sae_device, llm, layer_index, max_per_token, micro_bsz, dtype, normalized):
    sums = torch.zeros((vocab_size, all_val_data[0]["acts"].shape[-1]), device=sae_device, dtype=torch.float32)
    counts = torch.zeros((vocab_size,), device=sae_device, dtype=torch.int32)
    
    use_raw_recompute = normalized and (llm is not None)
    
    pbar = tqdm(total=len(all_val_data), desc="[Corpus Reps] Building", unit="batch", leave=False)
    for vb in all_val_data:
        ids = torch.tensor(vb["input_ids"], dtype=torch.long)
        mask = vb["mask_bt"]
        
        if use_raw_recompute:
            llm_dev, llm_dt = _get_llm_device_and_dtype(llm)
            acts_raw, _ = llm_acts_in_chunks(llm, ids.to(llm_dev), mask.to(llm_dev).long(), layer_index, False, micro_bsz, sae_device, dtype)
            acts = acts_raw
        else:
            acts = vb["acts"].to(sae_device, dtype=dtype)
            
        B, T = acts.shape[:2]
        ids = ids.to(sae_device)
        mask = mask.to(sae_device)
        
        for b in range(B):
            for t in range(T):
                if mask[b, t]:
                    tid = int(ids[b, t])
                    if tid < vocab_size and counts[tid] < max_per_token:
                        sums[tid] += acts[b, t].float()
                        counts[tid] += 1
        
        pbar.update(1)
        pbar.set_postfix({"valid_tokens": int(counts.sum().item())})
    pbar.close()
                        
    valid = (counts > 0)
    reps = torch.zeros_like(sums)
    if valid.any():
        avg = sums[valid] / counts[valid].unsqueeze(-1).float()
        centered = avg - avg.mean(dim=-1, keepdim=True)
        norms = centered.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        reps[valid] = centered / norms
        
    return reps, valid

@torch.no_grad()
def _metrics_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    labels = labels.to(scores.device)
    n_pos = float(labels.sum().item())
    n_neg = float((1.0 - labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    order = torch.argsort(scores, descending=True)
    y_sorted = labels[order]
    ranks = torch.arange(1, len(y_sorted) + 1, device=scores.device, dtype=torch.float32)
    U = float(ranks[y_sorted > 0.5].sum().item())
    auc = (U - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1.0 - y_sorted, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp[-1], min=1.0)
    ap = torch.trapz(precision, recall).item()
    return float(auc), float(ap)

@torch.no_grad()
def _select_probe_latents_by_corr(all_val_data, labels_by_batch, W_enc, b_enc, use_pre, feature_chunk, max_latents):
    device = W_enc.device
    n_latents = W_enc.shape[0]
    sum_z = torch.zeros(n_latents, device=device)
    sum_z2 = torch.zeros(n_latents, device=device)
    sum_zy = torch.zeros(n_latents, device=device)
    sum_y, sum_y2, total = 0.0, 0.0, 0
    for vb, lbl in zip(all_val_data, labels_by_batch):
        acts = vb["acts"].to(device, dtype=W_enc.dtype)
        mask = vb["mask_bt"].to(device)
        ybt = lbl.to(device, dtype=torch.float32)
        keep = mask & torch.isfinite(ybt)
        if keep.sum() == 0: continue
        for start in range(0, n_latents, feature_chunk):
            end = min(start + feature_chunk, n_latents)
            W_chunk = W_enc[start:end, :]
            b_chunk = b_enc[start:end]
            pre = torch.einsum("btd,nd->btn", acts, W_chunk) + b_chunk
            scalars = pre if use_pre else F.relu(pre)
            zf = scalars[keep]
            yf = ybt[keep].view(-1)
            sum_z[start:end] += zf.sum(dim=0)
            sum_z2[start:end] += (zf * zf).sum(dim=0)
            sum_zy[start:end] += (zf * yf.unsqueeze(-1)).sum(dim=0)
        y_keep = ybt[keep].view(-1)
        sum_y += float(y_keep.sum().item())
        sum_y2 += float((y_keep * y_keep).sum().item())
        total += int(y_keep.numel())
    if total == 0: return []
    mean_y = sum_y / total
    var_y = max(sum_y2 / total - mean_y * mean_y, 1e-12)
    mean_z = sum_z / total
    var_z = (sum_z2 / total) - (mean_z * mean_z)
    cov_zy = (sum_zy / total) - (mean_z * mean_y)
    denom = torch.sqrt(torch.clamp(var_z, min=1e-12) * var_y)
    corr = torch.where(denom > 0, torch.abs(cov_zy / denom), torch.zeros_like(denom))
    if max_latents is None or max_latents >= len(corr): idx = torch.arange(len(corr), device=device)
    else: _, idx = torch.topk(corr, k=min(int(max_latents), len(corr)))
    return idx.cpu().tolist()

def run_single_latent_probes(
    all_val_data, labels_by_batch, W_enc_cpu, b_enc_cpu, use_pre, out_csv, 
    feature_chunk, max_latents, max_steps, lr=0.001, l2=1e-4, val_stride=5, seed=13, device=None,
    shard_range: Tuple[int, int] = None, correlation_chunk: int = None
):
    if not all_val_data: raise ValueError("No validation data")
    if device is None: device = all_val_data[0]["acts"].device
    torch.manual_seed(seed)
    
    W_enc_full = W_enc_cpu.to(device)
    b_enc_full = b_enc_cpu.to(device)
    
    corr_chunk = correlation_chunk if correlation_chunk is not None else feature_chunk
    
    print("[probes] Selecting latents by correlation (Global)...")
    selected_latents_all = _select_probe_latents_by_corr(
        all_val_data, labels_by_batch, W_enc_full, b_enc_full, use_pre, corr_chunk, max_latents
    )
    
    if shard_range:
        s_start, s_end = shard_range
        selected_latents = [idx for idx in selected_latents_all if s_start <= idx < s_end]
        print(f"[probes] Shard {s_start}-{s_end}: Processing {len(selected_latents)} latents (filtered from {len(selected_latents_all)} globally selected).")
    else:
        selected_latents = selected_latents_all

    if not selected_latents:
        print("[probes] No latents selected for this shard.")
        return

    # 2. Setup CSV & Resume Logic
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latent_id", "w", "b", "ce_val", "auroc", "ap"])
        if not file_exists: writer.writeheader()
    
    existing_latent_ids = set()
    if os.path.exists(out_csv):
        try:
            with open(out_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader: existing_latent_ids.add(int(row["latent_id"]))
        except: pass

    n_chunks = int(math.ceil(len(selected_latents) / feature_chunk))
    
    with torch.enable_grad():
        chunk_pbar = tqdm(total=n_chunks, desc="[Probes] Chunks", unit="chunk", position=0)
        
        for start in range(0, len(selected_latents), feature_chunk):
            end = min(start + feature_chunk, len(selected_latents))
            chunk_ids = selected_latents[start:end]
            
            if all(lat_id in existing_latent_ids for lat_id in chunk_ids):
                chunk_pbar.update(1)
                continue

            idx_tensor = torch.tensor(chunk_ids, device=device, dtype=torch.long)
            W_chunk = W_enc_full.index_select(0, idx_tensor)
            b_chunk = b_enc_full.index_select(0, idx_tensor)
            
            w_probe = nn.Parameter(torch.zeros(len(chunk_ids), device=device, requires_grad=True))
            b_probe = nn.Parameter(torch.zeros(len(chunk_ids), device=device, requires_grad=True))
            optimizer = torch.optim.Adam([w_probe, b_probe], lr=lr)
            
            n_epochs = max(1, int(max_steps))
            train_pbar = tqdm(total=len(all_val_data) * n_epochs, desc="Training", unit="batch", leave=False, position=1)
            
            for epoch in range(n_epochs):
                for vb, lbl in zip(all_val_data, labels_by_batch):
                    optimizer.zero_grad(set_to_none=True)
                    acts = vb["acts"].to(device, dtype=W_enc_full.dtype)
                    mask = vb["mask_bt"].to(device)
                    lbl_dev = lbl.to(device, dtype=torch.float32)
                    keep = mask & torch.isfinite(lbl_dev)
                    if keep.sum() == 0: 
                        train_pbar.update(1)
                        continue
                    
                    pre = torch.einsum("btd,nd->btn", acts, W_chunk) + b_chunk
                    z_flat = (pre if use_pre else F.relu(pre))[keep]
                    y_flat = lbl_dev[keep].view(-1)
                    
                    idx = torch.arange(z_flat.shape[0], device=device)
                    train_mask = (idx % val_stride) != 0
                    if train_mask.sum() > 0:
                        z_train = z_flat[train_mask]
                        y_train = y_flat[train_mask].unsqueeze(1).expand(-1, len(chunk_ids))
                        logits = z_train * w_probe.view(1, -1) + b_probe.view(1, -1)
                        loss = F.binary_cross_entropy_with_logits(logits, y_train, reduction="mean") + l2 * (w_probe**2).mean()
                        loss.backward()
                        optimizer.step()
                        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    train_pbar.update(1)
            train_pbar.close()
            
            w_eval = w_probe.detach()
            b_eval = b_probe.detach()
            eval_sub_bsz = 1024
            
            total_val_samples = 0
            for vb, lbl in zip(all_val_data, labels_by_batch):
                mask = vb["mask_bt"]
                total_val_samples += mask.sum().item()
            total_eval_samples = int(total_val_samples / val_stride) + len(all_val_data) * 2
            
            eval_pbar = tqdm(total=len(chunk_ids), desc="Evaluating Latents", unit="latents", leave=False, position=1)
            
            for sub_start in range(0, len(chunk_ids), eval_sub_bsz):
                sub_end = min(sub_start + eval_sub_bsz, len(chunk_ids))
                n_sub = sub_end - sub_start
                W_sub = W_chunk[sub_start:sub_end]
                b_sub = b_chunk[sub_start:sub_end]
                w_p_sub = w_eval[sub_start:sub_end]
                b_p_sub = b_eval[sub_start:sub_end]
                
                all_logits = torch.zeros(total_eval_samples, n_sub, dtype=torch.float32)
                all_labels = torch.zeros(total_eval_samples, dtype=torch.float32)
                ptr = 0
                
                ce_sum = torch.zeros(n_sub, device=device)
                ce_count = torch.zeros(n_sub, device=device)
                
                for vb, lbl in zip(all_val_data, labels_by_batch):
                    acts = vb["acts"].to(device, dtype=W_chunk.dtype)
                    mask = vb["mask_bt"].to(device)
                    lbl_dev = lbl.to(device, dtype=torch.float32)
                    keep = mask & torch.isfinite(lbl_dev)
                    if keep.sum() == 0: continue
                    
                    pre = torch.einsum("btd,nd->btn", acts, W_sub) + b_sub
                    z = (pre if use_pre else F.relu(pre))[keep]
                    y = lbl_dev[keep].view(-1)
                    
                    idx = torch.arange(z.shape[0], device=device)
                    val_mask = (idx % val_stride) == 0
                    
                    if val_mask.sum() > 0:
                        z_val = z[val_mask]
                        y_val = y[val_mask]
                        
                        logits = z_val * w_p_sub.view(1, -1) + b_p_sub.view(1, -1)
                        
                        y_expanded = y_val.unsqueeze(1).expand(-1, n_sub)
                        ce_loss = F.binary_cross_entropy_with_logits(logits, y_expanded, reduction="none")
                        ce_sum += ce_loss.sum(dim=0)
                        ce_count += ce_loss.shape[0]
                        
                        n_batch = z_val.shape[0]
                        if ptr + n_batch > total_eval_samples:
                            new_size = ptr + n_batch + 10000
                            all_logits.resize_(new_size, n_sub)
                            all_labels.resize_(new_size)
                            total_eval_samples = new_size
                            
                        all_logits[ptr : ptr+n_batch] = logits.float().cpu()
                        all_labels[ptr : ptr+n_batch] = y_val.float().cpu()
                        ptr += n_batch
                
                if ptr > 0:
                    flat_logits_all = all_logits[:ptr]
                    flat_labels = all_labels[:ptr]
                    
                    for i in range(n_sub):
                        col_logits = flat_logits_all[:, i]
                        auc, ap = _metrics_from_scores(col_logits, flat_labels)
                        
                        ce_val = (ce_sum[i] / ce_count[i].clamp_min(1.0)).item()
                        lat_idx_global = chunk_ids[sub_start + i]
                        
                        row = {
                            "latent_id": int(lat_idx_global),
                            "w": float(w_p_sub[i]), "b": float(b_p_sub[i]),
                            "ce_val": ce_val, "auroc": auc, "ap": ap
                        }
                        
                        with open(out_csv, "a", newline="") as f:
                            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
                
                eval_pbar.update(n_sub)
            
            eval_pbar.close()
            chunk_pbar.update(1)
            
            del W_chunk, b_chunk, w_probe, b_probe, optimizer, all_logits, all_labels
            torch.cuda.empty_cache()
            
        chunk_pbar.close()

def export_sae_for_viewer_sharded(
    sae_state_dict, cfg_in_ckpt, tokenizer, llm, preloaded_data, args, export_dir, 
    sae_device, sae_dtype, feature_scale=None, shard_range=None, precomputed_corpus_reps=None
):
    os.makedirs(export_dir, exist_ok=True)
    # Load all Latents on CPU, but only move Shard to GPU
    W_dec_cpu = sae_state_dict['W_dec'].float().cpu()
    d_model, n_latents = W_dec_cpu.shape
    feature_scale_cpu = feature_scale.view(-1).float().cpu().clamp_min(1e-6) if feature_scale is not None else None
    
    if shard_range:
        s_start, s_end = shard_range
        s_start = max(0, s_start)
        s_end = min(n_latents, s_end)
        print(f"Exporting viewer shard: features {s_start} to {s_end}...")
    else:
        s_start, s_end = 0, n_latents

    feature_ids = list(range(s_start, s_end))
    token_heaps = {i:[] for i in feature_ids}
    neighbor_heaps = {i:[] for i in feature_ids}
    example_heaps = {i:[] for i in feature_ids}

    W_dec_shard = W_dec_cpu[:, s_start:s_end]
    if feature_scale_cpu is not None:
        dec_viz = W_dec_shard * feature_scale_cpu[s_start:s_end].unsqueeze(0)
    else:
        dec_viz = W_dec_shard
    
    W_dec_unit_shard = dec_viz / dec_viz.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
    W_dec_unit_shard = W_dec_unit_shard.to(device=sae_device, dtype=sae_dtype)

    if args.token_probe == "input":
        if llm is None: raise RuntimeError("LLM model required for 'input' probe.")
        emb_gpu = F.normalize(llm.get_input_embeddings().weight.detach().float(), p=2, dim=1).to(device=sae_device, dtype=sae_dtype)
        sim = emb_gpu @ W_dec_unit_shard
        for j, fid in enumerate(feature_ids):
            s, i = torch.topk(sim[:, j], k=args.top_tokens_k)
            for score, idx in zip(s.tolist(), i.tolist()): 
                _update_heap_maxk(token_heaps[fid], args.top_tokens_k, score, int(idx))
            del emb_gpu, sim
        
    elif args.token_probe == "corpus":
        corpus_reps = None
        if precomputed_corpus_reps:
            corpus_reps, token_valid = precomputed_corpus_reps
            corpus_reps = corpus_reps.to(device=sae_device, dtype=sae_dtype)
            token_valid = token_valid.to(sae_device)
        
        if corpus_reps is not None:
            valid_ids = token_valid.nonzero(as_tuple=False).view(-1)
            if valid_ids.numel() > 0:
                sim = corpus_reps[valid_ids] @ W_dec_unit_shard
                for j, fid in enumerate(feature_ids):
                    s, i = torch.topk(sim[:, j], k=args.top_tokens_k)
                    for score, loc_idx in zip(s.tolist(), i.tolist()): 
                        _update_heap_maxk(token_heaps[fid], args.top_tokens_k, score, int(valid_ids[loc_idx]))
            del corpus_reps, token_valid

    if args.neighbors_k and args.neighbors_k > 0:
        print(f"[Viewer] Computing neighbors for shard {s_start}-{s_end}...")
        query = W_dec_unit_shard.t()
        
        n_search = n_latents
        if args.neighbors_max_search_range > 0:
            n_search = min(n_latents, args.neighbors_max_search_range)
            print(f"  Limiting neighbor search to first {n_search} latents.")

        chunk_size_inner = 8192
        W_dec_all_unit_cpu = W_dec_cpu[:, :n_search] / W_dec_cpu[:, :n_search].norm(p=2, dim=0, keepdim=True).clamp_min(1e-8) # Re-normalize full on CPU
        
        n_chunks = int(math.ceil(n_search / chunk_size_inner))
        pbar = tqdm(total=n_chunks, desc=f"[Viewer] Neighbors ({s_start}-{s_end})", unit="chunk", leave=False)
        for j_inner in range(0, n_search, chunk_size_inner):
            f_start_inner = j_inner
            f_end_inner = min(j_inner + chunk_size_inner, n_search)
            
            W_target_chunk = W_dec_all_unit_cpu[:, f_start_inner:f_end_inner].to(device=sae_device, dtype=sae_dtype)
            
            sims = query @ W_target_chunk
            
            for row_idx in range(sims.shape[0]):
                fid = s_start + row_idx
                for col_idx in range(sims.shape[1]):
                    target_fid = f_start_inner + col_idx
                    if fid == target_fid: continue
                    
                    score = sims[row_idx, col_idx].item()
                    _update_heap_maxk(neighbor_heaps[fid], args.neighbors_k, score, target_fid)
            
            pbar.update(1)
            del W_target_chunk, sims
        pbar.close()
        del query, W_dec_all_unit_cpu
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"[Viewer] Mining examples for shard {s_start}-{s_end}...")
    W_enc_cpu = sae_state_dict['W_enc'].float().cpu()
    b_enc_cpu = sae_state_dict.get('b_enc', torch.zeros(n_latents)).float().cpu()
    
    W_enc_shard = W_enc_cpu[s_start:s_end].to(sae_device, dtype=sae_dtype)
    b_enc_shard = b_enc_cpu[s_start:s_end].to(sae_device, dtype=sae_dtype)
    scale_shard = feature_scale_cpu[s_start:s_end].to(sae_device) if feature_scale_cpu is not None else None
    
    caches = [SpanCache(tokenizer) for _ in preloaded_data]
    for c, d in zip(caches, preloaded_data): c.prepare_batch(d['texts'], d['input_ids'])

    ex_ctr = 0
    pbar = tqdm(total=len(preloaded_data), desc=f"[Viewer] Examples ({s_start}-{s_end})", unit="batch", leave=False)
    for vb, sc in zip(preloaded_data, caches):
        acts = vb['acts'].to(device=sae_device, dtype=W_enc_shard.dtype)
        mask = vb['mask_bt'].to(sae_device)
        
        pre = torch.einsum("btd,nd->btn", acts, W_enc_shard) + b_enc_shard
        z = F.relu(pre) if args.relu_after_topk else pre 
        z_masked = z * mask.unsqueeze(-1)
        
        for b in range(acts.shape[0]):
            vals, idxs = torch.max(z_masked[b], dim=0)
            
            for j, fid in enumerate(feature_ids):
                score = float(vals[j])
                if score < 1e-6: continue
                
                if len(example_heaps[fid]) >= args.top_examples_k:
                    if score <= example_heaps[fid][0][0]:
                        continue

                z_line = z_masked[b, :, j]
                z_norm = (z_line / scale_shard[j]).clamp(0,1) if scale_shard is not None else z_line
                
                acts_disp = sanitize_scores_list(sc.aggregate_acts_to_display(b, z_line))
                acts_norm = sanitize_scores_list(sc.aggregate_acts_to_display(b, z_norm)) if scale_shard is not None else None
                peak_disp = None
                if acts_disp:
                    try: peak_disp = int(max(range(len(acts_disp)), key=lambda i: acts_disp[i]))
                    except: pass
                
                cs = ce = None
                if peak_disp is not None:
                    offsets = sc._batch_offsets[b]
                    if 0 <= peak_disp < len(offsets):
                        _cs, _ce = offsets[peak_disp]
                        cs, ce = int(_cs), int(_ce)

                payload = {
                    "text": vb['texts'][b], 
                    "tokens": sc.display_tokens(b),   
                    "char_start": cs,                 
                    "char_end": ce,                   
                    "peak_token_index": int(idxs[j]), 
                    "score": finite_or_none(score),
                    "tokenScores": acts_disp, 
                    "display_peak_token_index": peak_disp
                }
                if acts_norm: payload["tokenScoresNormalized"] = acts_norm
                
                ex_ctr += 1
                _update_heap_maxk(example_heaps[fid], args.top_examples_k, score, payload, unique_id=-ex_ctr)
        
        pbar.update(1)
        pbar.set_postfix({"examples": ex_ctr})
    pbar.close()
    
    del W_enc_shard, b_enc_shard
    torch.cuda.empty_cache()

    suffix = f"_shard_{args.shard_id}" if args.shard_id >= 0 else ""
    if token_heaps:
        with open(os.path.join(export_dir, f"top_tokens{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["latent_id", "rank", "token_str", "token_id", "score"])
            w.writeheader()
            unique_ids = sorted({x[1] for h in token_heaps.values() for x in h})
            id2str = dict(zip(unique_ids, tokenizer.convert_ids_to_tokens(unique_ids)))
            for fid in sorted(token_heaps):
                for r, (s, tid) in enumerate(sorted(token_heaps[fid], reverse=True), 1):
                    w.writerow({"latent_id": fid, "rank": r, "token_str": id2str.get(tid, str(tid)), "token_id": tid, "score": finite_or_none(s)})

    pack = {
        "features": {
            str(fid): {
                "top_tokens": [[id2str.get(t, str(t)), s] for s, t in sorted(token_heaps[fid], reverse=True)],
                "neighbors": [[item, finite_or_none(score)] for score, item in sorted(neighbor_heaps[fid], reverse=True, key=lambda x: x[0])],
                "examples": _finalize_heap_desc(example_heaps[fid])
            } for fid in feature_ids
        }
    }
    with open(os.path.join(export_dir, f"viewer_pack{suffix}.json"), "w") as f:
        json.dump(pack, f)
    print(f"Viewer Export for Shard {args.shard_id} complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--model_id", default="google/medgemma-27b-text-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer_index", type=int, default=52)
    ap.add_argument("--reserve_gpu_for_sae", type=int, default=0)
    ap.add_argument("--llm_mem_other_gib", type=int, default=60)
    ap.add_argument("--llm_mem_reserved_gib", type=int, default=10)
    ap.add_argument("--ctx_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--llm_micro_bsz", type=int, default=32)
    ap.add_argument("--val_batches", type=int, default=64)
    ap.add_argument("--max_val_batches", type=int, default=64)
    ap.add_argument("--relu_after_topk", action="store_true", default=True) # Default true for TopK SAEs
    
    ap.add_argument("--max_viewer_latents", type=int, default=0, help="Limit for heavy viewer export (0=all)")
    ap.add_argument("--probe_max_latents", type=int, default=65536, help="Limit for logistic probes (0=all)")
    ap.add_argument("--feature_shard_size", type=int, default=16384)
    ap.add_argument("--neighbors_k", type=int, default=15)
    ap.add_argument("--neighbors_max_search_range", type=int, default=0, help="Limit neighbors search to first N latents (0=all)")
    ap.add_argument("--top_tokens_k", type=int, default=50)
    ap.add_argument("--top_examples_k", type=int, default=10)
    ap.add_argument("--use_auxk_for_examples", action="store_true")
    ap.add_argument("--token_probe", default="corpus")
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--probe_batch_size", type=int, default=2048)
    ap.add_argument("--probe_include_specials", action="store_true")
    ap.add_argument("--probe_vocab_limit", type=int, default=0)
    ap.add_argument("--probe_add_bos", action="store_true")
    ap.add_argument("--corpus_probe_max_per_token", type=int, default=128)
    ap.add_argument("--dataset_format", required=True)
    ap.add_argument("--dataset_path")
    ap.add_argument("--csv_path")
    ap.add_argument("--text_column")
    ap.add_argument("--strip_bhc_section", action="store_true")
    ap.add_argument("--val_percent", type=float, default=0.01)
    ap.add_argument("--use_saved_acts", action="store_true")
    ap.add_argument("--acts_root", type=str)
    ap.add_argument("--acts_split", type=str, default="val")
    ap.add_argument("--acts_max_files", type=int, default=0)
    ap.add_argument("--acts_normalized", type=str, default="auto")
    ap.add_argument("--run_delta_ce", action="store_true")
    ap.add_argument("--delta_ce_max_batches", type=int, default=None)
    ap.add_argument("--alpha_swap", type=float, default=1.0)
    ap.add_argument("--kl_topk_only", action="store_true")
    ap.add_argument("--kl_vocab_chunk", type=int, default=65536)
    ap.add_argument("--run_probes", action="store_true")
    ap.add_argument("--labels_path", type=str, default="")
    ap.add_argument("--probe_start_latent", type=int, default=None, 
                    help="Start processing probes from this latent ID (skips lower IDs). If None, checks merged probes CSV.")
    ap.add_argument("--merged_probes_csv", type=str, default=None,
                    help="Path to merged probes CSV to check for already-processed latents")
    ap.add_argument("--probe_use_pre", action="store_true")
    ap.add_argument("--probe_max_steps", type=int, default=200)
    ap.add_argument("--run_n2g", action="store_true")
    ap.add_argument("--n2g_max_ngram", type=int, default=6)
    ap.add_argument("--n2g_thresh_quantile", type=float, default=0.98)
    ap.add_argument("--n2g_min_hits", type=int, default=25)
    ap.add_argument("--export_topk_latents_csv", type=str, default="")
    ap.add_argument("--filter_terms_csv", type=str, default="")
    ap.add_argument("--with_offsets", action="store_true")
    ap.add_argument("--total_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)

    args = ap.parse_args()
    
    # Setup logging
    logger = setup_logging(args.shard_id, args.total_shards)

    # GPU Assignment
    if torch.cuda.is_available():
        sae_device = torch.device(f"cuda:0") 
    else:
        sae_device = torch.device("cpu")

    logger.info(f"Initializing on {sae_device} (Step 3d: Viewer/Probes/TopK Export)")

    logger.info("Loading tokenizer...")
    tok = get_tokenizer_only(args.model_id, args.ctx_len)
    logger.info("Tokenizer loaded")

    logger.info("Loading validation data...")
    all_val_data = []
    saved_ds = None
    saved_acts_are_normalized = None
    val_loader = None

    if args.use_saved_acts:
        gen_meta = load_gen_meta(args.acts_root)
        if args.acts_normalized == "auto":
            if "normalize" in gen_meta:
                saved_acts_are_normalized = gen_meta["normalize"]
            else:
                saved_acts_are_normalized = None
        else:
            saved_acts_are_normalized = (args.acts_normalized == "yes")
            
        saved_ds = SavedActsBatches(args.acts_root, split=args.acts_split, max_files=args.acts_max_files, preload=args.preload)
        loader = torch.utils.data.DataLoader(saved_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_saved_batches)
        
        auto_probe_done = False
        sae_dtype = str2dtype(args.dtype)
        
        max_batches = min(args.max_val_batches, len(loader)) if args.max_val_batches > 0 else len(loader)
        pbar = tqdm(total=max_batches, desc=f"[Shard {args.shard_id}] Loading batches", unit="batch", leave=False)
        
        for i, batch in enumerate(loader):
            va, vm, ids = batch["acts"].to(sae_dtype), batch["mask_bt"].bool(), batch["input_ids"]
            
            if torch.is_floating_point(va):
                nan_mask = torch.isnan(va).any(dim=-1)
                if nan_mask.any():
                    vm = vm & (~nan_mask)
                    va = torch.nan_to_num(va, nan=0.0, posinf=0.0, neginf=0.0)
            
            if saved_acts_are_normalized is None:
                if not auto_probe_done:
                    saved_acts_are_normalized = looks_normalized(va, vm)
                    logger.info(f"Auto-detected normalization: {saved_acts_are_normalized}")
                    auto_probe_done = True
            
            if not saved_acts_are_normalized: 
                va = center_unit_norm(va, vm)
            
            meta_payload = {
                "acts": va.cpu(), 
                "mask_bt": vm.cpu(), 
                "texts": _decode_batch_texts(tok, ids.tolist()), 
                "input_ids": ids.tolist()
            }
            if "char_offsets" in batch: meta_payload["char_offsets"] = batch["char_offsets"]
            if "doc_id" in batch: meta_payload["doc_id"] = batch["doc_id"]
            
            all_val_data.append(meta_payload)
            pbar.update(1)
            if i+1 >= args.max_val_batches: break
        
        pbar.close()
        if saved_acts_are_normalized is None: saved_acts_are_normalized = True
        logger.info(f"Loaded {len(all_val_data)} validation batches")

    else:
        saved_acts_are_normalized = True
        requires_llm = True

    requires_llm = False
    if not args.use_saved_acts: requires_llm = True
    elif args.run_delta_ce and args.shard_id == 0: requires_llm = True
    elif args.token_probe in ["layer", "input"]: requires_llm = True
    elif args.token_probe == "corpus" and saved_acts_are_normalized: requires_llm = True
    
    lm = None
    if requires_llm:
        logger.info("Loading LLM (required for requested tasks)...")
        lm = get_model_only(args.model_id, str2dtype(args.dtype), os.path.join(args.export_dir, "hf_offload"))
        logger.info("LLM loaded")
    else:
        logger.info("Skipping LLM load (using saved activations / raw features only)")

    logger.info(f"Loading SAE checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    sae_state_dict = ckpt.get("model", ckpt.get("sae_state_dict", ckpt))
    cfg_in_ckpt = ckpt.get("config", {})
    feature_scale = _get_state_tensor(sae_state_dict, "feature_scale")
    sae_dtype = str2dtype(args.dtype)
    total_latents = sae_state_dict["W_enc"].shape[0]
    logger.info(f"SAE loaded: {total_latents} latents, d_model={sae_state_dict['W_dec'].shape[0]}")

    if args.run_delta_ce and args.shard_id == 0:
        logger.info("=" * 60)
        logger.info("Task 1: Delta CE (Shard 0 only)")
        logger.info("=" * 60)
        batch_iter = _iter_batches_for_delta_ce(args, lm, val_loader, args.acts_root, args.acts_split, args.acts_max_files)
        
        W_enc_full = sae_state_dict["W_enc"].float()
        W_dec_full = sae_state_dict["W_dec"].float()
        b_enc_full = sae_state_dict.get("b_enc", torch.zeros(W_enc_full.shape[0])).float()
        b_dec_full = sae_state_dict.get("b_dec", torch.zeros(W_dec_full.shape[0])).float()
        k_sys = cfg_in_ckpt.get("k", 128)
        
        max_batches = args.delta_ce_max_batches or float('inf')
        dce_stats = []
        with torch.set_grad_enabled(False):
            pbar = tqdm(desc=f"[Shard {args.shard_id}] Delta CE", unit="batch", leave=False)
            for i, vb in enumerate(batch_iter):
                res = delta_ce_for_batch(
                    lm, W_enc_full, b_enc_full, W_dec_full, b_dec_full,
                    vb["acts"], vb["mask_bt"], vb["input_ids"],
                    alpha=args.alpha_swap, k_main=k_sys, device=sae_device, layer_index=args.layer_index
                )
                dce_stats.append({
                    "batch": i,
                    "delta_ce": res[0],
                    "kl": res[1],
                    "ce_base": res[2],
                    "ce_swap": res[3]
                })
                pbar.update(1)
                pbar.set_postfix({"dce": f"{res[0]:.4f}", "kl": f"{res[1]:.4f}"})
                if (i+1) >= max_batches: break
            pbar.close()
                
        if dce_stats:
            avg_dce = sum(s["delta_ce"] for s in dce_stats) / len(dce_stats)
            avg_kl = sum(s["kl"] for s in dce_stats) / len(dce_stats)
            avg_base = sum(s["ce_base"] for s in dce_stats) / len(dce_stats)
            avg_swap = sum(s["ce_swap"] for s in dce_stats) / len(dce_stats)
            logger.info(f"Delta CE completed: avg_dce={avg_dce:.6f}, avg_kl={avg_kl:.6f} ({len(dce_stats)} batches)")
            print(json.dumps({"metric": "delta_ce", "value": avg_dce, "kl": avg_kl}))
            
            metrics_output = {
                "summary": {
                    "delta_ce": {
                        "mean": avg_dce,
                        "n_batches": len(dce_stats)
                    },
                    "kl": {
                        "mean": avg_kl,
                        "n_batches": len(dce_stats)
                    },
                    "ce_base": {
                        "mean": avg_base,
                        "n_batches": len(dce_stats)
                    },
                    "ce_swap": {
                        "mean": avg_swap,
                        "n_batches": len(dce_stats)
                    }
                },
                "per_batch": dce_stats
            }
            
            metrics_path = os.path.join(args.export_dir, "delta_ce_metrics.json")
            os.makedirs(args.export_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics_output, f, indent=2)
            logger.info(f"[delta-ce] Metrics saved to: {metrics_path}")
        
        del W_enc_full, W_dec_full, batch_iter
        gc.collect()

    if args.export_topk_latents_csv and not os.path.exists(args.export_topk_latents_csv):
        logger.info("=" * 60)
        logger.info("Task 2: Top-K Export (Data Sharded)")
        logger.info("=" * 60)
        total_batches = len(all_val_data)
        per_shard = int(math.ceil(total_batches / args.total_shards))
        d_start = args.shard_id * per_shard
        d_end = min(d_start + per_shard, total_batches)
        
        if d_start < d_end:
            logger.info(f"Exporting Top-K CSV for data batches {d_start}-{d_end} (total: {total_batches})")
            data_subset = all_val_data[d_start:d_end]
            
            filter_terms = None
            if args.filter_terms_csv and os.path.exists(args.filter_terms_csv):
                try:
                    df_lex = pd.read_csv(args.filter_terms_csv)
                    filter_terms = df_lex["term"].unique().tolist()
                    logger.info(f"Loaded {len(filter_terms)} filter terms")
                except Exception as e:
                    logger.warning(f"Failed to load filter terms: {e}")

            base, ext = os.path.splitext(args.export_topk_latents_csv)
            shard_filename = f"{base}_shard_{args.shard_id}{ext}"
            
            export_topk_latents_csv(
                sae_state_dict=sae_state_dict, 
                cfg_in_ckpt=cfg_in_ckpt, 
                tokenizer=tok,
                all_val_data=data_subset, 
                args=args, 
                export_path=shard_filename,
                filter_terms=filter_terms, 
                with_offsets=args.with_offsets, 
                feature_scale=feature_scale,
                global_batch_offset=d_start,
            )
            logger.info(f"Top-K export completed: {shard_filename}")
        else:
            logger.info("Data shard empty for Top-K export.")

    limit_view = args.max_viewer_latents if args.max_viewer_latents > 0 else total_latents
    limit_probe = args.probe_max_latents if args.probe_max_latents > 0 else total_latents
    
    probe_start_latent = args.probe_start_latent if args.probe_start_latent is not None else 0
    if args.run_probes and args.merged_probes_csv and os.path.exists(args.merged_probes_csv):
        try:
            max_latent_id = -1
            with open(args.merged_probes_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    latent_id = int(row.get('latent_id', -1))
                    if latent_id > max_latent_id:
                        max_latent_id = latent_id
            if max_latent_id >= 0:
                probe_start_latent = max_latent_id + 1
                logger.info(f"Found merged probes CSV with max latent_id={max_latent_id}, starting probes from {probe_start_latent}")
        except Exception as e:
            logger.warning(f"Could not read merged probes CSV {args.merged_probes_csv}: {e}. Starting from {probe_start_latent}")
    
    target_latents = max(limit_view, limit_probe)
    target_latents = min(target_latents, total_latents)
    
    per_shard_lat = int(math.ceil(target_latents / args.total_shards))
    s_start = args.shard_id * per_shard_lat
    s_end = min(s_start + per_shard_lat, target_latents)
    

    shard_range_size = s_end - s_start
    dynamic_feature_chunk = min(args.feature_shard_size, max(shard_range_size, 1))
    
    if s_start < s_end:
        logger.info("=" * 60)
        logger.info("Task 3: Viewer & Probes (Feature Sharded)")
        logger.info("=" * 60)
        logger.info(f"Feature processing range: {s_start}-{s_end} (out of {target_latents} total)")
        logger.info(f"Dynamic feature_chunk: {dynamic_feature_chunk} (shard range: {shard_range_size}, configured: {args.feature_shard_size})")
        
        if args.run_probes:
            p_start = max(probe_start_latent, s_start)
            p_end = min(s_end, limit_probe)
            
            if p_start < p_end and p_start < limit_probe:
                logger.info(f"Running probes for latents {p_start}-{p_end} ({p_end - p_start} latents)...")
                W_enc_cpu = sae_state_dict['W_enc'].float()
                b_enc_cpu = sae_state_dict.get('b_enc', torch.zeros(total_latents)).float()
                
                if all_val_data:
                    logger.info("Loading probe labels...")
                    labels = _load_token_labels_npz(args.labels_path, saved_ds.files[:len(all_val_data)])
                    logger.info(f"Loaded labels for {len(labels)} batches")
                    
                    suffix = f"_shard_{args.shard_id}"
                    probes_csv = os.path.join(args.export_dir, "probes", f"single_latent_probes{suffix}.csv")
                    
                    run_single_latent_probes(
                        all_val_data=all_val_data, labels_by_batch=labels, 
                        W_enc_cpu=W_enc_cpu, b_enc_cpu=b_enc_cpu,
                        use_pre=args.probe_use_pre, out_csv=probes_csv, 
                        feature_chunk=dynamic_feature_chunk,
                        max_latents=total_latents,
                        max_steps=args.probe_max_steps, device=sae_device, 
                        shard_range=(p_start, p_end),
                        correlation_chunk=args.feature_shard_size
                    )
                    logger.info(f"Probes completed: {probes_csv}")
            else:
                    if args.run_probes:
                        logger.info(f"Skipping probes for shard {args.shard_id}: range {p_start}-{p_end} is invalid or already processed")

        if not os.path.exists(os.path.join(args.export_dir, f"viewer_pack.json")):
            v_end = min(s_end, limit_view)
            if s_start < v_end:
                logger.info(f"Running viewer export for latents {s_start}-{v_end} ({v_end - s_start} latents)...")
                
                corpus_reps_tuple = None
                if args.token_probe == "corpus":
                    logger.info("Building corpus reps for viewer...")
                    vocab_size = tok.vocab_size 
                    if lm is not None: vocab_size = lm.config.vocab_size
                    reps, valid = build_corpus_token_reps(
                        all_val_data, vocab_size, sae_device, lm, 
                        args.layer_index, args.corpus_probe_max_per_token, 
                        args.llm_micro_bsz, sae_dtype, saved_acts_are_normalized
                    )
                    corpus_reps_tuple = (reps.float().cpu(), valid.cpu())
                    logger.info(f"Corpus reps built: {valid.sum().item()}/{vocab_size} valid tokens")

                export_sae_for_viewer_sharded(
                    sae_state_dict=sae_state_dict, cfg_in_ckpt=cfg_in_ckpt, tokenizer=tok, llm=lm,
                    preloaded_data=all_val_data, args=args, export_dir=args.export_dir,
                    sae_device=sae_device, sae_dtype=sae_dtype, feature_scale=feature_scale,
                    shard_range=(s_start, v_end), precomputed_corpus_reps=corpus_reps_tuple
                )
                logger.info("Viewer export completed")

    logger.info("=" * 60)
    logger.info(f"Shard {args.shard_id}/{args.total_shards} completed successfully")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
