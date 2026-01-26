#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch
import torch.nn.functional as F


def load_model_and_sae(model_id: str, ckpt_path: str, layer_index: int, 
                       dtype: torch.dtype = torch.bfloat16):
    """Load LLM and SAE checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_viewer_comparisons_fixed import prepare_hf_with_reservation, str2dtype
    
    # Map model IDs to local paths if available
    def get_local_model_path(model_id: str) -> str:
        """Map HuggingFace model IDs to local paths if models are available locally."""
        model_path_map = {
            "google/medgemma-27b-text-it": "/bmbl_data/mirage/MedGemma",
            "aaditya/OpenBioLLM-Llama3-70B": "/bmbl_data/mirage/OpenBioLLM",
        }
        return model_path_map.get(model_id, model_id)
    
    local_model_path = get_local_model_path(model_id)
    print(f"[Info] Loading model from local path: {local_model_path} (from {model_id})")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    
    # Load SAE
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sae_state_dict = ckpt["model"]
        cfg_in_ckpt = {}
    else:
        sae_state_dict = ckpt.get("sae_state_dict", ckpt)
        cfg_in_ckpt = ckpt.get("config", {})
    
    return model, tokenizer, sae_state_dict, cfg_in_ckpt


def compute_delta_ce_ablation(model, tokenizer, sae_state_dict, cfg_in_ckpt,
                             acts: torch.Tensor, input_ids: torch.Tensor,
                             attention_mask: torch.Tensor, latent_id: int,
                             layer_index: int, device: torch.device) -> float:
    """
    Compute ΔCE when ablating a specific latent.
    
    Returns:
        ΔCE = CE(ablated) - CE(baseline)
    """
    from sae_viewer_comparisons_fixed import _loss_shifted_ce, delta_ce_for_batch
    
    k_main = cfg_in_ckpt.get("k", 128)
    relu_after_topk = cfg_in_ckpt.get("relu_after_topk", True)
    
    # Move to device
    acts_dev = acts.to(device, dtype=next(model.parameters()).dtype)
    ids_dev = input_ids.to(device)
    attn_dev = attention_mask.to(device)
    mask_bt = (attn_dev > 0).bool()
    
    # 1) Baseline forward
    with torch.no_grad():
        out_base = model(input_ids=ids_dev, attention_mask=attn_dev, use_cache=False)
        logits_base = out_base.logits
        ce_base = _loss_shifted_ce(logits_base, ids_dev, mask_bt[:, 1:])
    
    # 2) SAE encode/decode
    from sae_viewer_comparisons_fixed import _sae_params_to_x
    
    mu = acts_dev.mean(dim=-1, keepdim=True)
    centered = acts_dev - mu
    sigma = centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x_norm = centered / sigma

    W_enc, b_enc, W_dec, b_dec = _sae_params_to_x(
        sae_state_dict["W_enc"],
        sae_state_dict.get("b_enc", None),
        sae_state_dict["W_dec"],
        sae_state_dict.get("b_dec", None),
        x_norm
    )
    
    # Encode
    pre = torch.einsum("btd,nd->btn", x_norm, W_enc) + b_enc
    k_eff = min(k_main, pre.shape[-1])
    _, idx_top = torch.topk(pre, k=k_eff, dim=-1)
    mask_top = torch.zeros_like(pre).scatter_(-1, idx_top, 1.0)
    z = (F.relu(pre) if relu_after_topk else pre) * mask_top

    # Ablate latent ℓ: set z[..., ℓ] = 0
    z_ablate = z.clone()
    z_ablate[..., latent_id] = 0.0
    
    # Decode
    x_hat_base_norm = torch.einsum("btn,dn->btd", z, W_dec)
    if b_dec is not None:
        x_hat_base_norm = x_hat_base_norm + b_dec
    
    x_hat_ablate_norm = torch.einsum("btn,dn->btd", z_ablate, W_dec)
    if b_dec is not None:
        x_hat_ablate_norm = x_hat_ablate_norm + b_dec

    x_hat_ablate = x_hat_ablate_norm * sigma + mu
    
    # 3) Swap reconstruction at target layer
    from sae_viewer_comparisons_fixed import ResidualSwap, _find_decoder_layers_module, route_inputs_for_hf, module_device
    
    layers = _find_decoder_layers_module(model)
    target_layer = layers[layer_index]
    target_dev = module_device(target_layer)
    x_hat_target = x_hat_ablate.to(device=target_dev, non_blocking=True)
    mask_target = mask_bt.to(device=target_dev, non_blocking=True)
    
    with ResidualSwap(model, layer_index=layer_index, new_resid=x_hat_target, mask=mask_target):
        ids_swap, attn_swap = route_inputs_for_hf(model, ids_dev, attn_dev)
        out_swap = model(input_ids=ids_swap, attention_mask=attn_swap, use_cache=False)
        logits_swap = out_swap.logits
    
    # 4) Compute ΔCE
    ce_swap = _loss_shifted_ce(logits_swap, ids_dev, mask_bt[:, 1:])
    delta_ce = float((ce_swap - ce_base).item())
    
    return delta_ce


def main():
    parser = argparse.ArgumentParser(description="Run context-matched ablation effects")
    # ... (Keep arguments the same) ...
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--layer_index", type=int, required=True)
    parser.add_argument("--sense_linked_csv", type=str, required=True)
    parser.add_argument("--topk_csv", type=str, required=True)
    parser.add_argument("--acts_root", type=str, required=True)
    parser.add_argument("--acts_split", type=str, default="val")
    parser.add_argument("--acts_max_files", type=int, default=0) # Set to 0 in bash!
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_ablations", type=int, default=100, 
                        help="Total global limit (safety stop)")
    parser.add_argument("--max_ablations_per_sense", type=int, default=10, 
                        help="Maximum number of ablations to run per unique lemma-sense pair")
    
    args = parser.parse_args()
    
    from sae_viewer_comparisons_fixed import str2dtype
    dtype = str2dtype(args.dtype)
    
    # Load model and SAE
    print(f"Loading model {args.model_id} and SAE from {args.ckpt_path}...")
    model, tokenizer, sae_state_dict, cfg_in_ckpt = load_model_and_sae(
        args.model_id, args.ckpt_path, args.layer_index, dtype
    )
    device = next(model.parameters()).device
    
    # Load sense-linked latents
    print(f"Loading sense-linked latents from {args.sense_linked_csv}...")
    df_sense_linked = pd.read_csv(args.sense_linked_csv)
    
    # Sort by strength to test best candidates first per sense
    if "presence_rate_diff" in df_sense_linked.columns:
        df_sense_linked = df_sense_linked.sort_values("presence_rate_diff", ascending=False)
    
    # Load Top-K latents
    print(f"Loading Top-K latents from {args.topk_csv}...")
    df_topk = pd.read_csv(args.topk_csv)
    if "latent_ids_json" in df_topk.columns:
        import json
        df_topk["latent_ids"] = df_topk["latent_ids_json"].apply(json.loads)
    
    # Load activation batches
    from sae_viewer_comparisons_fixed import SavedActsBatches, collate_saved_batches, load_gen_meta, center_unit_norm
    
    print(f"Loading activation batches from {args.acts_root}...")
    saved_ds = SavedActsBatches(args.acts_root, split=args.acts_split, max_files=args.acts_max_files)
    
    
    gen_meta = load_gen_meta(args.acts_root)
    acts_norm_flag = "yes" if gen_meta.get("normalize", False) else "no"
    
    # Run ablations
    results = []
    global_count = 0
    sense_counts = {} # Track count per (lemma, sense)
    
    # Iterate through candidates
    for _, row in df_sense_linked.iterrows():
        if global_count >= args.max_ablations:
            print("Global ablation limit reached.")
            break
            
        lemma = row["lemma"]
        sense = row["sense"]
        latent_id = int(row["latent_id"])
        
        # Check per-sense quota
        key = (lemma, sense)
        current_sense_count = sense_counts.get(key, 0)
        if current_sense_count >= args.max_ablations_per_sense:
            continue # Skip to next candidate, we have enough for this sense
            
        print(f"Processing: {lemma}/{sense} (Latent {latent_id}) [{current_sense_count+1}/{args.max_ablations_per_sense}]")
        
        # Find matching occurrences in Top-K data
        matches = df_topk[
            (df_topk["lemma"].str.lower() == lemma.lower()) &
            (df_topk["latent_ids"].apply(lambda x: latent_id in x if isinstance(x, list) else False))
        ]
        
        if len(matches) == 0:
            continue
        
        match_row = matches.sample(n=1, random_state=42).iloc[0]
        
        batch_idx = int(match_row.get("batch_idx", 0))
        doc_id = match_row.get("doc_id", None)
        
        # Safety check for file existence
        if batch_idx >= len(saved_ds.files):
            print(f"  Skipping: batch_idx {batch_idx} out of range (max {len(saved_ds.files)})")
            continue
        
        try:
            # Load batch
            batch_data = torch.load(saved_ds.files[batch_idx], map_location="cpu")
            acts = batch_data.get("hidden_states", batch_data.get("activations"))
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data.get("attention_mask", (input_ids != 0).long())
            
            if acts is None: continue
            
            # Handle normalization
            if acts_norm_flag == "no":
                mask_bt = (attention_mask != 0) if attention_mask.dtype != torch.bool else attention_mask
                acts = center_unit_norm(acts, mask_bt)
            
            # Compute ΔCE
            delta_ce = compute_delta_ce_ablation(
                model, tokenizer, sae_state_dict, cfg_in_ckpt,
                acts, input_ids, attention_mask,
                latent_id, args.layer_index, device
            )
            
            results.append({
                "lemma": lemma,
                "sense": sense,
                "latent_id": latent_id,
                "doc_id": doc_id,
                "batch_idx": batch_idx,
                "delta_ce": delta_ce,
                "layer": args.layer_index,
                "model": args.model_id,
            })
            
            # Update counts
            global_count += 1
            sense_counts[key] = current_sense_count + 1
            
        except Exception as e:
            print(f"  Warning: Ablation failed for {lemma}/{sense}: {e}")
            continue
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output, index=False)
        print(f"Saved {len(results)} ablation results to {args.output}")
        
        summary = df_results.groupby(["lemma", "sense"]).agg({
            "delta_ce": ["mean", "count"]
        }).reset_index()
        print("\nSummary by lemma-sense:")
        print(summary)
    else:
        print("Warning: No ablation results computed")

if __name__ == "__main__":
    main()
