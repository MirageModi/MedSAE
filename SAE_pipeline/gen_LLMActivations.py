
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_LLMActivations_fixed.py — Multi-process (1 process/GPU) activation generator with Option A sharding.

This version mirrors the user's original activation logic:
- Batching/windowing: windows are left-padded to exactly `ctx_len` (like the original collate).
- Capture path: try forward-hook on the decoder block; if it fails, fall back to `output_hidden_states=True`
  with index = clamp(layer_index + 1, 0, len(hidden_states)-1).
- Normalization: optional `--normalize` flag applies *per-token* mean-centering + L2-normalization with eps=1e-8.
- No zeroing of padded activations (mask is saved for downstream handling).
- Saves batch files containing: hidden_states [B, T, H], input_ids [B, T], attention_mask [B, T],
  and per-window metadata (global_row_index, window_index, token_start, token_end).

Sharding: Option-A (modulo of a global permutation) across ranks; outputs under <output_dir>/shard_{rank}/train|val/
Also writes indices_train.npy / indices_val.npy per shard for verification.

Run example (single node, 4 GPUs):
  torchrun --nproc-per-node=4 gen_LLMActivations_fixed.py \
    --csv_path /path/to/data.csv \
    --text_column note_text \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --attn_impl sdpa \
    --layer_index 20 \
    --ctx_len 2048 \
    --stride 2048 \
    --batch_texts 8 \
    --output_dir /path/to/activations \
    --val_percent 0.1 \
    --shuffle --shuffle_seed 42
"""

import os, json, argparse, time, gc
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
except ImportError as e:
    raise RuntimeError("Please install pandas to load CSVs") from e

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    raise RuntimeError("Please install transformers") from e


def ddp_setup_if_needed(gpus_per_rank: int = None) -> Tuple[int, int, int]:
    """
    Setup distributed training environment.
    
    Args:
        gpus_per_rank: If specified, partition CUDA_VISIBLE_DEVICES so each rank sees
                       only its own subset of GPUs. This is required when using device_map="auto"
                       with multiple ranks, since each process would otherwise try to use all GPUs.
    
    Returns:
        (rank, world_size, local_rank)
    """
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if gpus_per_rank is not None and gpus_per_rank > 0 and world > 1:
        orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if orig_visible:
            all_gpus = [int(g.strip()) for g in orig_visible.split(",")]
        else:
            all_gpus = list(range(torch.cuda.device_count()))
        
        start_idx = local_rank * gpus_per_rank
        end_idx = start_idx + gpus_per_rank
        my_gpus = all_gpus[start_idx:end_idx]
        
        if len(my_gpus) == 0:
            raise RuntimeError(
                f"[rank {rank}] No GPUs available! Requested gpus_per_rank={gpus_per_rank}, "
                f"but with world={world} and {len(all_gpus)} total GPUs, "
                f"rank {local_rank} would need GPUs [{start_idx}:{end_idx}] which is out of range. "
                f"Reduce --nproc-per-node or --gpus_per_rank."
            )
        
        new_visible = ",".join(str(g) for g in my_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = new_visible
        print(f"[Info][rank {rank}] Partitioned GPUs: using devices {new_visible} (from original {orig_visible})")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world, local_rank


def shard_indices(n_total: int, rank: int, world: int, perm: torch.Tensor) -> torch.Tensor:
    # Option A sharding: indices i where i % world == rank, on a (maybe shuffled) permutation
    return perm[rank::world]

def l2_normalize_per_token(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    denom = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / denom

class WindowedTextDataset(Dataset):
    """
    Tokenizes texts, splits into windows of length ctx_len with step 'stride', and
    tracks per-window metadata (source global row index, window index within note,
    and token start/end offsets). Window IDs are not padded here; padding happens in collate.
    """
    def __init__(
        self,
        texts: List[str],
        src_rows: List[int],   # global row indices corresponding to each text
        doc_ids: List[Any],    # actual document IDs from CSV (e.g., note_id)
        tokenizer,
        ctx_len: int,
        stride: int,
        tokenizer_truncation_side: str = "right",
        add_bos_eos: bool = True,
    ):
        assert len(texts) == len(src_rows), "texts and src_rows must align"
        assert len(texts) == len(doc_ids), "texts and doc_ids must align"
        self.tokenizer = tokenizer
        self.ctx_len = int(ctx_len)
        self.stride = int(stride) if stride is not None else int(ctx_len)
        self.add_bos_eos = add_bos_eos

        if hasattr(self.tokenizer, "truncation_side"):
            self.tokenizer.truncation_side = tokenizer_truncation_side

        self.windows: List[Dict[str, Any]] = []
        for txt, global_row, doc_id in zip(texts, src_rows, doc_ids):
            enc = tokenizer(
                txt,
                add_special_tokens=add_bos_eos,
                return_attention_mask=True,
                return_tensors=None,
            )
            ids = enc["input_ids"]
            att = enc["attention_mask"]
            if isinstance(ids[0], list):
                ids = ids[0]; att = att[0]

            if len(ids) <= self.ctx_len:
                self.windows.append({
                    "ids": ids,
                    "att": att,
                    "global_row_index": int(global_row),
                    "doc_id": doc_id,
                    "window_index_within_note": 0,
                    "token_start": 0,
                    "token_end": len(ids),
                    "text": txt,
                })
            else:
                start = 0
                widx = 0
                limit = max(0, len(ids) - 1)
                while start <= limit:
                    end = min(start + self.ctx_len, len(ids))
                    window_ids = ids[start:end]
                    window_att = att[start:end]
                    self.windows.append({
                        "ids": window_ids,
                        "att": window_att,
                        "global_row_index": int(global_row),
                        "doc_id": doc_id,
                        "window_index_within_note": int(widx),
                        "token_start": int(start),
                        "token_end": int(end),
                        "text": txt,
                    })
                    if end == len(ids):
                        break
                    start += self.stride
                    widx += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.windows[idx]

def pad_collate(batch: List[Dict[str, Any]], pad_id: int, ctx_len: int, 
                save_offsets: bool = False, tokenizer=None, add_bos_eos: bool = True) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    T = ctx_len
    input_ids = torch.full((bsz, T), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bsz, T), dtype=torch.long)
    global_row_index = torch.empty((bsz,), dtype=torch.long)
    window_index = torch.empty((bsz,), dtype=torch.long)
    token_start = torch.empty((bsz,), dtype=torch.long)
    token_end = torch.empty((bsz,), dtype=torch.long)
    doc_ids = []

    char_offsets = [] if save_offsets else None

    for i, item in enumerate(batch):
        orig_seq = list(item["ids"])
        seq = orig_seq.copy()
        am = list(item["att"])
        L = len(seq)
        if L < T:
            pad_len = T - L
            seq = [pad_id] * pad_len + seq
            am  = [0] * pad_len + am
            if save_offsets and tokenizer is not None:
                orig_text = item.get("text", "")
                if orig_text:
                    try:
                        if hasattr(tokenizer, "backing_tokenizer"):
                            tok_obj = tokenizer.backing_tokenizer
                        else:
                            tok_obj = tokenizer
                        enc_with_offsets = tok_obj(
                            orig_text, 
                            return_offsets_mapping=True, 
                            add_special_tokens=add_bos_eos
                        )
                        if isinstance(enc_with_offsets, dict):
                            offsets_list = enc_with_offsets.get("offset_mapping", [])
                        else:
                            offsets_list = []
                    except Exception:
                        offsets_list = []
                    
                    if not offsets_list:
                        offsets_list = [(0, 0)] * len(orig_seq)
                else:
                    offsets_list = [(0, 0)] * len(orig_seq)
                
                pad_offsets = [(0, 0)] * pad_len
                offsets_list = pad_offsets + offsets_list[:T]
                if len(offsets_list) < T:
                    offsets_list.extend([(0, 0)] * (T - len(offsets_list)))
                char_offsets.append(offsets_list[:T])
        else:
            seq = seq[:T]
            am  = am[:T]
            if save_offsets and tokenizer is not None:
                orig_text = item.get("text", "")
                if orig_text:
                    try:
                        if hasattr(tokenizer, "backing_tokenizer"):
                            tok_obj = tokenizer.backing_tokenizer
                        else:
                            tok_obj = tokenizer
                        enc_with_offsets = tok_obj(
                            orig_text,
                            return_offsets_mapping=True,
                            add_special_tokens=add_bos_eos
                        )
                        if isinstance(enc_with_offsets, dict):
                            offsets_list = enc_with_offsets.get("offset_mapping", [])[:T]
                        else:
                            offsets_list = []
                    except Exception:
                        offsets_list = []
                    
                    if not offsets_list or len(offsets_list) < T:
                        offsets_list = offsets_list[:T] if offsets_list else []
                        offsets_list.extend([(0, 0)] * (T - len(offsets_list)))
                else:
                    offsets_list = [(0, 0)] * T
                char_offsets.append(offsets_list[:T])
        input_ids[i] = torch.tensor(seq, dtype=torch.long)
        attn_mask[i] = torch.tensor(am,  dtype=torch.long)
        global_row_index[i] = item["global_row_index"]
        window_index[i] = item["window_index_within_note"]
        token_start[i] = item["token_start"]
        token_end[i] = item["token_end"]
        doc_ids.append(item.get("doc_id", item["global_row_index"]))

    result = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "global_row_index": global_row_index,
        "window_index": window_index,
        "token_start": token_start,
        "token_end": token_end,
        "doc_id": doc_ids,
    }
    if save_offsets and char_offsets:
        result["char_offsets"] = char_offsets
    return result

def _find_decoder_layers_module(model):
    """
    Try common module paths to get the list of decoder blocks.
    Returns the module that is indexable (supports __getitem__). Matches original behavior.
    """
    candidates = [
        "model.layers",            # Gemma / MedGemma
        "model.decoder.layers",    # some decoders
        "transformer.h",           # GPT-style
        "transformer.layers",
        "layers",
    ]
    for path in candidates:
        mod = model
        ok = True
        for name in path.split("."):
            if not hasattr(mod, name):
                ok = False; break
            mod = getattr(mod, name)
        if ok:
            try:
                _ = mod[0]
                return mod
            except Exception:
                continue
    raise RuntimeError(f"Could not locate decoder layers (tried: {candidates}).")


@torch.no_grad()
def _get_layer_acts_hooked(model, input_ids, attention_mask, layer_index: int, normalize: bool):
    """
    Capture only the chosen layer via a forward hook (no output_hidden_states), like the original.
    Returns activations on the model's device, shape [B, T, H].
    """
    layers = _find_decoder_layers_module(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"layer_index {layer_index} out of range (have {len(layers)})")
    target = layers[layer_index]
    cached = {"acts": None}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        cached["acts"] = h.detach()

    handle = target.register_forward_hook(hook, with_kwargs=False)
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    acts = cached["acts"]
    if acts is None:
        raise RuntimeError("Activation hook did not fire; model/layer path may differ.")
    if normalize:
        acts = l2_normalize_per_token(acts)
    return acts

def extract_layer_hidden_states(
    model,
    batch: Dict[str, torch.Tensor],
    layer_index: int,
    dtype: torch.dtype,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    """Return [B, T, H] activations on CPU, without trimming or zeroing pad positions."""
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(dtype in (torch.bfloat16, torch.float16))):
            ids = batch["input_ids"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)

            acts = None
            try:
                acts = _get_layer_acts_hooked(model, ids, att, layer_index, normalize=normalize)
            except Exception:
                out = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                L = len(hs)
                idx = max(0, min(layer_index + 1, L - 1))
                acts = hs[idx]
                if normalize:
                    acts = l2_normalize_per_token(acts)

        return acts.detach().to("cpu")

def parse_args():
    p = argparse.ArgumentParser(description="Generate LLM hidden activations (Option A sharding).")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--text_column", type=str, required=True)

    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn_impl", type=str, default="auto",
                   choices=["auto", "sdpa", "flash2", "eager", "xformers"],
                   help="Attention backend preference. Model/config support may vary.")
    p.add_argument("--layer_index", type=int, required=True,
                   help="Decoder block index for hook; fallback uses hidden_states[idx=clamp(layer_index+1,...)]")
    p.add_argument("--ctx_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=None, help="Token step between windows; default = ctx_len (no overlap)")
    p.add_argument("--llm_micro_bsz", "--batch_texts", dest="llm_micro_bsz", type=int, default=4)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--val_percent", type=float, default=0.0)
    p.add_argument("--max_rows", type=int, default=-1)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--normalize", action="store_true",
                   help="Apply per-token mean-centered L2 norm (matches original l2_normalize_per_token).")
    p.add_argument("--device_map", type=str, default=None, 
                   help="HuggingFace device_map (e.g. 'auto'). If None, defaults to single-GPU (rank-based).")

    p.add_argument("--tokenizer_truncation_side", type=str, default="right", choices=["left", "right"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefix_rank_in_filenames", action="store_true",
                   help="Prefix saved batch files with rank to make names globally unique")
    p.add_argument("--resume", action="store_true",
               help="Skip batches whose output file already exists and is loadable")
    p.add_argument("--save_char_offsets", action="store_true",
                   help="Save per-token character offsets for NER span alignment")
    p.add_argument("--save_doc_id", action="store_true",
                   help="Save document ID per window (uses global_row_index if available)")
    p.add_argument("--doc_id_column", type=str, default="",
                   help="CSV column containing unique document IDs. If empty, uses row index.")
    p.add_argument("--save_section", action="store_true",
                   help="Save section headers/labels per window (requires CSV column 'section')")
    p.add_argument("--gpus_per_rank", type=int, default=None,
                   help="Number of GPUs to allocate per rank. Required when using --device_map auto "
                        "with multiple ranks to prevent GPU contention. E.g., for a 70B model on 8 GPUs "
                        "with 2 ranks, use --gpus_per_rank 4. If not specified with device_map=auto and "
                        "world_size > 1, will auto-compute based on available GPUs.")
    return p.parse_args()

def safe_torch_save(obj, final_path: Path):
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())   # durability on network filesystems
    os.replace(tmp, final_path)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    
    gpus_per_rank = args.gpus_per_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if args.device_map == "auto" and world_size > 1:
        orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if orig_visible:
            n_gpus = len([g for g in orig_visible.split(",") if g.strip()])
        else:
            n_gpus = torch.cuda.device_count()
        
        if gpus_per_rank is None:
            gpus_per_rank = n_gpus // world_size
            if gpus_per_rank == 0:
                raise RuntimeError(
                    f"Cannot use device_map='auto' with {world_size} ranks on {n_gpus} GPUs. "
                    f"Each rank needs at least 1 GPU. Reduce --nproc-per-node to <= {n_gpus}."
                )
            print(f"[Info] Auto-computed gpus_per_rank={gpus_per_rank} "
                  f"({n_gpus} GPUs / {world_size} ranks)")
    
    rank, world, local_rank = ddp_setup_if_needed(gpus_per_rank=gpus_per_rank)
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    try:
        if args.attn_impl == "flash2":
            print(f"[Info][rank {rank}] Requesting attn_implementation=flash_attention_2")
        elif args.attn_impl == "sdpa":
            print(f"[Info][rank {rank}] Requesting attn_implementation=sdpa")
        elif args.attn_impl == "xformers":
            print(f"[Info][rank {rank}] Requesting attn_implementation=xformers")
        elif args.attn_impl == "eager":
            try:
                from torch.backends.cuda import sdp_kernel
                sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
                print(f"[Info][rank {rank}] Forcing eager (math) SDPA kernels")
            except Exception as e:
                print(f"[Warn][rank {rank}] Could not set sdp_kernel eager mode: {e}")
        else:
            print(f"[Info][rank {rank}] Using default attention backend (auto)")
    except Exception as e:
        print(f"[Warn][rank {rank}] Could not set attention backend preference: {e}")

    if rank == 0:
        print(f"[Info] Loading CSV from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.text_column not in df.columns:
        raise KeyError(f"text_column '{args.text_column}' not found in CSV columns: {list(df.columns)[:20]}")

    if args.max_rows is not None and args.max_rows > 0:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    N = len(df)
    if rank == 0:
        print(f"[Info] Total rows: {N}")

    gen = torch.Generator()
    gen.manual_seed(args.shuffle_seed if args.shuffle else 0)
    perm = torch.randperm(N, generator=gen) if args.shuffle else torch.arange(N)

    val_p = max(0.0, min(0.99, float(args.val_percent)))
    n_val = int(round(N * val_p))
    n_train = N - n_val
    train_idx_global = perm[:n_train]
    val_idx_global   = perm[n_train:] if n_val > 0 else torch.empty(0, dtype=torch.long)

    train_idx_local = shard_indices(n_train, rank, world, train_idx_global)
    val_idx_local   = shard_indices(n_val,   rank, world, val_idx_global) if n_val > 0 else torch.empty(0, dtype=torch.long)

    if rank == 0:
        print(f"[Info] Sharding with Option A across world={world}. Train {n_train} / Val {n_val}")

    train_np = None
    val_np = None
    try:
        import numpy as np
        train_np = train_idx_local.cpu().numpy()
        val_np = val_idx_local.cpu().numpy() if val_idx_local.numel() > 0 else np.empty((0,), dtype=int)
    except Exception as e:
        if rank == 0:
            print(f"[Warn] Could not convert indices to numpy: {e}")

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
    
    local_model_path = get_local_model_path(args.model_id)
    if rank == 0:
        print(f"[Info] Loading tokenizer: {local_model_path} (from {args.model_id})")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    if rank == 0:
        print(f"[Info] Loading model: {local_model_path} (from {args.model_id}) on device {device}")
    extra_model_kwargs = {}
    if args.attn_impl in ("sdpa", "flash2", "xformers"):
        extra_model_kwargs["attn_implementation"] = {"sdpa":"sdpa","flash2":"flash_attention_2","xformers":"xformers"}[args.attn_impl]
    
    if args.device_map:
        use_device_map = args.device_map
    elif device.type == "cuda":
        use_device_map = { "": device.index }
    else:
        use_device_map = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=use_device_map,
            trust_remote_code=True,
            **extra_model_kwargs,
        ).eval()
    except TypeError:
        print(f"[Warn][rank {rank}] attn_implementation not supported by transformers/model; loading without it.")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=use_device_map,
            trust_remote_code=True,
        ).eval()

    out_root = Path(args.output_dir)
    shard_dir = out_root / f"shard_{rank:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "rank": rank, "world_size": world, "local_rank": local_rank,
        "model_id": args.model_id, "dtype": args.dtype, "layer_index": args.layer_index,
        "ctx_len": args.ctx_len, "stride": args.stride if args.stride is not None else args.ctx_len,
        "llm_micro_bsz": args.llm_micro_bsz, "val_percent": val_p, "shuffle": bool(args.shuffle),
        "shuffle_seed": args.shuffle_seed, "normalize": bool(args.normalize),
        "normalization_method": "l2_per_token" if args.normalize else "none",
        "tokenizer_truncation_side": args.tokenizer_truncation_side,
        "total_rows": N, "n_train_global": n_train, "n_val_global": n_val,
        "n_train_local": int(train_idx_local.numel()), "n_val_local": int(val_idx_local.numel()),
        "csv_path": args.csv_path, "text_column": args.text_column,
        "attn_impl": args.attn_impl,
        "device_map": args.device_map,
        "gpus_per_rank": gpus_per_rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "save_char_offsets": bool(args.save_char_offsets),
        "save_doc_id": bool(args.save_doc_id),
        "doc_id_column": args.doc_id_column if args.doc_id_column else None,
        "save_section": bool(args.save_section),
        "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else args.model_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(shard_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    try:
        if train_np is not None:
            import numpy as _np
            _np.save(shard_dir / "indices_train.npy", train_np)
            _np.save(shard_dir / "indices_val.npy", val_np if val_np is not None else _np.empty((0,), dtype=int))
    except Exception as e:
        print(f"[Warn][rank {rank}] Could not save indices npy files: {e}")

    def run_split(name: str, idx_local: torch.Tensor):
        if idx_local.numel() == 0:
            if rank == 0: print(f"[Info] split={name}: no local samples on rank {rank}")
            return

        src_rows = idx_local.tolist()
        texts = df.iloc[src_rows][args.text_column].astype(str).tolist()
        
        if args.doc_id_column and args.doc_id_column in df.columns:
            doc_ids = df.iloc[src_rows][args.doc_id_column].tolist()
        else:
            doc_ids = src_rows

        ds = WindowedTextDataset(
            texts=texts,
            src_rows=src_rows,
            doc_ids=doc_ids,
            tokenizer=tokenizer,
            ctx_len=args.ctx_len,
            stride=args.stride if args.stride is not None else args.ctx_len,
            tokenizer_truncation_side=args.tokenizer_truncation_side,
            add_bos_eos=True,
        )
        dl = DataLoader(
            ds, batch_size=args.llm_micro_bsz, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=lambda batch: pad_collate(
                batch, pad_id, ctx_len=args.ctx_len,
                save_offsets=args.save_char_offsets,
                tokenizer=tokenizer if args.save_char_offsets else None,
                add_bos_eos=True
            ),
        )

        save_dir = shard_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        bidx = 0

        for batch in dl:
            fname = (f"batch_{bidx:06d}.pt"
                    if not args.prefix_rank_in_filenames
                    else f"rank{rank}_batch_{bidx:06d}.pt")
            out_path = save_dir / fname

            if args.resume and out_path.exists():
                try:
                    data = torch.load(out_path, map_location="cpu")
                    ok = ("hidden_states" in data and "attention_mask" in data and "input_ids" in data)
                    if ok:
                        hs = data["hidden_states"]
                        if torch.is_tensor(hs):
                            ok = (hs.ndim == 3 and hs.shape[1] == args.ctx_len)
                        elif isinstance(hs, list):
                            ok = all(torch.is_tensor(t) and t.ndim == 2 for t in hs)
                    if ok:
                        if rank == 0 and (bidx % 500 == 0):
                            print(f"[resume] split={name} rank={rank} skipping {out_path.name}")
                        bidx += 1
                        del data
                        continue
                except Exception:
                    pass
                try: out_path.unlink()
                except FileNotFoundError: pass

            acts_bth = extract_layer_hidden_states(
                model, batch, layer_index=args.layer_index, dtype=torch_dtype, device=device,
                normalize=bool(args.normalize)
            )

            payload = {
                "hidden_states": acts_bth,
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "global_row_index": batch["global_row_index"],
                "window_index": batch["window_index"],
                "token_start": batch["token_start"],
                "token_end": batch["token_end"],
            }
            
            if args.save_char_offsets and "char_offsets" in batch:
                payload["char_offsets"] = batch["char_offsets"]
            
            if args.save_doc_id:
                payload["doc_id"] = batch["doc_id"]
            
            if args.save_section:
                section_values = []
                for global_idx in batch["global_row_index"].tolist():
                    if hasattr(df, "iloc") and "section" in df.columns:
                        section_val = str(df.iloc[global_idx].get("section", ""))
                    else:
                        section_val = ""
                    section_values.append(section_val)
                payload["section"] = section_values
            
            safe_torch_save(payload, out_path)
            bidx += 1

            del acts_bth, batch, payload
            if device.type == "cuda":
                torch.cuda.synchronize()

        if rank == 0:
            print(f"[Info] split={name}: rank {rank} wrote {bidx} batch files to {save_dir}")

    try:
        run_split("train", train_idx_local)
        if n_val > 0:
            run_split("val", val_idx_local)
    finally:
        if dist.is_initialized():
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        print("[Done] Activation generation complete.")

if __name__ == "__main__":
    main()
