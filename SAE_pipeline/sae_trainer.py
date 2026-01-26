#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import os
os.environ.setdefault("TORCH_FSDP_USE_ORIG_PARAMS","1")

sae_train_allinone_v3.py
- Option A: detect & mask NaN tokens in the train/val loops (no dataset sanitization).
- Masked aux loss computed outside the model using the same token mask.
- Preserves your original flags/optimizer/EMA/FSDP semantics and whole-file batching.
"""

import os, re, json, math, time, argparse, random, datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from contextlib import nullcontext, contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Optional Adafactor & bitsandbytes
try:
    from transformers.optimization import Adafactor
except Exception:
    Adafactor = None
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

CKPT_EVERY = 50  # or make it a CLI flag
# ==================== Utils ====================
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def barrier():
    if dist.is_available() and dist.is_initialized(): dist.barrier()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def str2dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bfloat16","bf16"): return torch.bfloat16
    if s in ("float16","fp16"):  return torch.float16
    return torch.float32

# ==================== Config (preserved flags/defaults) ====================
@dataclass
class TrainConfig:
    activations_dir: str = "activations"
    save_dir: str = "sae_multinode/run05-newdata-deadlatents"
    d_model: int = 5120
    n_latents: int = 262144
    k: int = 128
    k_aux: int = 512
    dead_threshold_steps: int = 200
    relu_after_topk: bool = False
    lr: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 6.25e-10
    wd: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ema_enable: bool = False
    aux_coeff: float = 1.0/32.0
    train_steps: int = 2000
    micro_bsz: int = 8
    log_every: int = 20
    val_every: int = 1
    val_micro_bsz: int = 8
    optimizer: str = "adafactor"
    sae_fsdp: bool = False
    sae_fsdp_devices: str = "0,1,2,3"
    dtype: str = "bfloat16"

# ==================== FSDP ====================
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig, FullOptimStateDictConfig, CPUOffload

@contextmanager
def fsdp_ctx(m):
    if isinstance(m, FSDP):
        # Materialize original-shaped params for math ops like einsum
        with FSDP.summon_full_params(m, writeback=False, recurse=True):
            yield
    else:
        yield

# def _parse_devices(devs: str):
#     return [int(x) for x in devs.split(",") if x.strip() != ""]

# def init_sae_distributed(cfg):
#     import datetime
#     dev_ids = [int(x) for x in cfg.sae_fsdp_devices.split(",") if x.strip()]

#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     rank       = int(os.environ.get("RANK", 0))
#     world_size = int(os.environ.get("WORLD_SIZE", 1))

#     # map local_rank -> actual GPU id
#     device_id = dev_ids[local_rank] if cfg.sae_fsdp else dev_ids[0]
#     torch.cuda.set_device(device_id)
#     sae_device = torch.device(f"cuda:{device_id}")

#     if cfg.sae_fsdp and not dist.is_initialized():
#         dist.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             timeout=datetime.timedelta(hours=4),
#             # no device_id arg here
#         )

#     # sanity
#     assert torch.cuda.current_device() == sae_device.index

#     return dict(world=(len(dev_ids) if cfg.sae_fsdp else 1),
#                 rank=rank, local_rank=local_rank,
#                 sae_device=sae_device, sae_dev_ids=dev_ids)

def _parse_devices(devs: str):
    return [int(x) for x in devs.split(",") if x.strip()]

def init_sae_distributed(cfg):
    import datetime
    import os
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world      = int(os.environ.get("WORLD_SIZE", 1))

    dev_ids = _parse_devices(cfg.sae_fsdp_devices) or [0]
    device_id = dev_ids[local_rank] if cfg.sae_fsdp else dev_ids[0]

    # Bind CUDA device BEFORE init_process_group
    torch.cuda.set_device(device_id)
    sae_device = torch.device(f"cuda:{device_id}")

    if cfg.sae_fsdp and not dist.is_initialized():
        try:
            # Newer PyTorch: accepts torch.device
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(hours=4),
                device_id=sae_device,   # <-- torch.device, NOT int
            )
        except TypeError:
            # Older PyTorch: no device_id kwarg
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(hours=4),
            )

    # Sanity check
    assert torch.cuda.current_device() == sae_device.index

    return dict(world=(len(dev_ids) if cfg.sae_fsdp else 1),
                rank=rank, local_rank=local_rank,
                sae_device=sae_device, sae_dev_ids=dev_ids)

# def init_sae_distributed(cfg: TrainConfig) -> Dict[str, Any]:
#     # Preserve your devices string semantics
#     if not cfg.sae_fsdp:
#         dev = f"cuda:{cfg.sae_fsdp_devices.split(',')[0]}"
#         torch.cuda.set_device(torch.device(dev))
#         return dict(world=1, rank=0, local_rank=0, sae_device=torch.device(dev), sae_dev_ids=[int(cfg.sae_fsdp_devices.split(',')[0])])
#     dev_ids = _parse_devices(cfg.sae_fsdp_devices)
#     world = len(dev_ids)
#     rank = int(os.environ.get("RANK","0"))
#     local_rank = int(os.environ.get("LOCAL_RANK","0"))
#     assert local_rank < world, f"LOCAL_RANK {local_rank} >= world {world} (devices {dev_ids})"
#     sae_device = torch.device(f"cuda:{dev_ids[local_rank]}")
#     torch.cuda.set_device(sae_device)
#     if not dist.is_initialized():
#         dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4), device_id=sae_device.index)
#     return dict(world=world, rank=rank, local_rank=local_rank, sae_device=sae_device, sae_dev_ids=dev_ids)

# ==================== SAE ====================
class TopK(nn.Module):
    def __init__(self, k: int, relu_after: bool = True):
        super().__init__()
        self.k = k
        self.relu_after = relu_after
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.k <= 0 or self.k >= z.shape[-1]:
            return torch.relu(z) if self.relu_after else z
        vals, idx = torch.topk(z, k=self.k, dim=-1)
        if self.relu_after: vals = torch.relu(vals)
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, vals)
        return out

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int,
                 relu_after_topk: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_model = d_model; self.n_latents = n_latents; self.k = k
        self.topk = TopK(k, relu_after=relu_after_topk)
        self.W_enc = nn.Parameter(torch.empty(n_latents, d_model, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(n_latents, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_model, n_latents, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        nn.init.xavier_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())

        # Post-hoc feature calibration state (persisted with checkpoints)
        self.register_buffer("feature_scale", torch.ones(n_latents, dtype=torch.float32))
        self.register_buffer("feature_hits", torch.zeros(n_latents, dtype=torch.long))

    # ---- RESTORED API ----
    def encode_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Linear encoder pre-activations: [B,T,N]"""
        return torch.einsum("btd,nd->btn", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z, preact) where z is top-k gated activations."""
        pre = self.encode_preact(x)
        pre = torch.clamp(pre, -1e4, 1e4)  # gentle guard for bf16 extremes
        z = self.topk(pre)
        return z, pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Linear decoder: [B,T,D]"""
        return torch.einsum("btn,dn->btd", z, self.W_dec) + self.b_dec

    @torch.no_grad()
    def renorm_decoder_columns(self, eps: float = 1e-12, preserve_product: bool = True):
        """
        L2-normalize decoder columns; if preserve_product, scale encoder rows inversely
        so that W_dec[:,j]/c and W_enc[j,:]*c keeps x_hat ~ invariant.
        """
        col_norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(eps)  # [1,N]
        self.W_dec.div_(col_norms)
        if preserve_product:
            self.W_enc.mul_(col_norms.squeeze(0).unsqueeze(1))  # scale each encoder row j by c_j

    @torch.no_grad()
    def calibrate_feature_scales(
        self,
        dataloader,
        device: torch.device,
        *,
        steps: Optional[int] = None,
        quantile: float = 0.999,
        eps: float = 1e-6,
    ) -> None:
        """
        Estimate per-feature activation scales from a stream of activation batches.

        Args:
            dataloader: yields dicts matching ActivationDatasetFiles ({"activations", "masks"}).
            device: device on which calibration should run.
            steps: optional cap on number of batches; default = iterate full loader.
            quantile: high quantile used as robust scale (falling back to max if needed).
            eps: lower bound applied to the resulting scales.
        """

        self.eval()

        max_stats = torch.zeros(self.n_latents, device=device, dtype=torch.float32)
        hit_counts = torch.zeros_like(self.feature_hits, dtype=torch.float32, device=device)

        if steps is None:
            total_steps = len(dataloader) if hasattr(dataloader, "__len__") else None
            steps = total_steps if total_steps is not None else float("inf")
        else:
            total_steps = steps

        for step_idx, batch in enumerate(dataloader):
            if step_idx >= steps:
                break

            acts = batch["activations"].to(device=device, dtype=self.W_enc.dtype, non_blocking=True)
            mask = batch["masks"].to(device=device, dtype=torch.bool, non_blocking=True)

            # Reuse Option-A sanitisation: drop NaN tokens before measuring activations.
            nan_tokens = torch.isnan(acts).any(dim=-1)
            if nan_tokens.any():
                mask = mask & ~nan_tokens
                acts = torch.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)

            if mask.sum() == 0:
                continue

            # NOTE: Activations from gen_LLMActivations are ALREADY normalized
            # (mean-centered + L2 normalized per token) when --normalize flag was used.
            # We should NOT normalize again here, as that would double-normalize.
            # The SAE was trained on these already-normalized activations.
            
            pre = self.encode_preact(acts)
            z = F.relu(pre)

            z_flat = z.view(-1, self.n_latents)
            mask_flat = mask.view(-1)
            if mask_flat.any():
                valid_z = z_flat[mask_flat]
                if valid_z.numel() == 0:
                    continue

                try:
                    batch_q = torch.quantile(valid_z, quantile, dim=0)
                except RuntimeError:
                    batch_q = valid_z.max(dim=0).values
                max_stats = torch.maximum(max_stats, batch_q.float())

                hit_counts += (valid_z > 0).float().sum(dim=0)

        max_stats = max_stats.clamp_min(eps)
        self.feature_scale.copy_(max_stats.cpu())
        self.feature_hits.copy_(hit_counts.cpu().to(dtype=self.feature_hits.dtype))

    @torch.no_grad()
    def decoder_for_viz(self) -> torch.Tensor:
        """Scaled decoder weights for downstream visualisation."""
        scale = self.feature_scale.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        return self.W_dec * scale.unsqueeze(0)

    @torch.no_grad()
    def activations_for_viz(self, z: torch.Tensor) -> torch.Tensor:
        """Return latent activations mapped into [0, 1] using calibrated feature scales."""
        scale = self.feature_scale.to(device=z.device, dtype=z.dtype)
        return (z / scale.view(1, 1, -1)).clamp(0, 1)

    def forward(self, x: torch.Tensor, k_aux: int = 0, mask: torch.Tensor = None):
        # Main path
        z, pre = self.encode(x)                 # uses self.W_enc (FSDP-managed)
        x_hat  = self.decode(z)                 # uses self.W_dec (FSDP-managed)

        aux_loss = None
        if self.training and k_aux and k_aux > self.k:
            # Residual (detach so aux only trains encoder)
            residual = (x - x_hat).detach()

            # Project residual into latent space
            pre_res = self.encode_preact(residual)  # uses self.W_enc

            # Select top-k_aux latents for residual
            k_aux_eff = min(k_aux, pre_res.shape[-1])
            with torch.no_grad():
                _, idx_aux = torch.topk(pre_res, k=k_aux_eff, dim=-1)
                aux_sel = torch.zeros_like(pre_res).scatter_(-1, idx_aux, 1.0)

            z_aux = F.relu(pre_res) * aux_sel

            # Decode with decoder weights **detached** (no grad to decoder)
            residual_hat = torch.einsum("btn,dn->btd",
                                        z_aux,
                                        self.W_dec.detach()) + self.b_dec.unsqueeze(0).unsqueeze(0)

            # Masked aux loss (if mask provided); else unmasked
            if mask is not None:
                aux_num = ((residual_hat - residual).float().pow(2) * mask.unsqueeze(-1)).sum()
                aux_den = mask.sum().clamp_min(1).to(residual_hat.dtype)
                aux_loss = aux_num / aux_den
            else:
                aux_loss = F.mse_loss(residual_hat.float(), residual.float())

        # Return like before (note: now we return aux_loss here)
        return x_hat, z, pre, aux_loss

# ==================== Losses ====================
def mse_masked(x_hat: torch.Tensor, x: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    diff2 = (x_hat - x) ** 2
    masked = diff2 * mask_bt.unsqueeze(-1)
    denom = mask_bt.sum().clamp_min(1).to(x.dtype)
    return masked.sum() / denom

def nmse_masked(x_hat: torch.Tensor, x: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    num = ((x_hat - x) ** 2 * mask_bt.unsqueeze(-1)).sum()
    den = ((x ** 2) * mask_bt.unsqueeze(-1)).sum().clamp_min(1e-12)
    return num / den

# ==================== EMA ====================
class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = module
        self.shadow = {n: p.detach().clone() for n,p in module.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, module: nn.Module = None):
        m = module or self.module
        for n,p in m.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, module: nn.Module):
        for n,p in module.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

# ==================== Dataset ====================
class ActivationDatasetFiles(Dataset):
    def __init__(self, activations_dir: str, split: str = "train", preload: bool = False, 
                 rank: int = 0, world_size: int = 1, max_files: Optional[int] = None):
        self.root = activations_dir; self.split = split; self.preload = preload
        files: List[str] = []
        flat = os.path.join(self.root, split)
        if os.path.isdir(flat):
            for f in os.listdir(flat):
                if f.endswith(".pt") and "batch_" in f:
                    files.append(os.path.join(flat, f))
        for entry in sorted(os.listdir(self.root)):
            if entry.startswith("shard_"):
                d = os.path.join(self.root, entry, split)
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith(".pt") and "batch_" in f:
                            files.append(os.path.join(d, f))
        if not files: raise FileNotFoundError(f"No batch files in {self.root} split={split}")
        files.sort()

        if max_files is not None and max_files > 0:
             print(f"[dataset] split={split} Limiting to {max_files} files (was {len(files)})")
             # Shuffle before truncation to avoid bias if files are ordered by time/patient/source
             rng = random.Random(42) 
             rng.shuffle(files)
             files = files[:max_files]
             files.sort() # Re-sort after selection for deterministic sharding across ranks
        
        # Optional: shard files deterministically across ranks (useful if preloading)
        if world_size > 1 and self.preload:
            files = files[rank::world_size]
            
        self.batch_files = files
        print(f"[dataset] split={split} rank={rank}/{world_size} files={len(self.batch_files)}")
        self.model_config: Dict[str, Any] = {}
        try:
            first = torch.load(self.batch_files[0], map_location="cpu")
            if isinstance(first, dict):
                if "metadata" in first and isinstance(first["metadata"], dict):
                    self.model_config = first["metadata"].get("model_config", {})
                elif "model_config" in first and isinstance(first["model_config"], dict):
                    self.model_config = first["model_config"]
        except Exception as e:
            print("[dataset] warn: cannot read model_config:", e)
        
        self.data = [None] * len(self.batch_files)
        if self.preload:
            print(f"[dataset] Preloading {len(self.batch_files)} files for split={split} into RAM (threaded)...")
            t0 = time.time()
            
            def _load_index(i):
                return i, self._load_file(i)

            with ThreadPoolExecutor(max_workers=16) as executor:
                results = executor.map(_load_index, range(len(self.batch_files)))
                for i, content in results:
                    self.data[i] = content
                    
            print(f"[dataset] Preloaded {len(self.data)} files in {time.time()-t0:.2f}s")

    def _load_file(self, idx: int) -> Dict[str, torch.Tensor]:
        d = torch.load(self.batch_files[idx], map_location="cpu")
        acts = d.get("activations", d.get("hidden_states", None))
        m    = d.get("masks", d.get("attention_mask", None))
        if acts is None or m is None:
            raise KeyError(f"Missing activations/masks in {self.batch_files[idx]}")
        if m.dtype != torch.bool: m = m.bool()
        if isinstance(acts, list):
            T = max(t.shape[0] for t in acts); D = acts[0].shape[-1]; B = len(acts)
            acts_pad = torch.zeros((B, T, D), dtype=acts[0].dtype)
            masks_pad = torch.zeros((B, T), dtype=torch.bool)
            for i,a in enumerate(acts):
                L = a.shape[0]; acts_pad[i, -L:, :] = a; masks_pad[i, -L:] = True
            acts, m = acts_pad, masks_pad
        # NOTE: No sanitization here (Option A); keep raw so we can detect NaNs later.
        return {"activations": acts, "masks": m}

    def __len__(self): return len(self.batch_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload: return self.data[idx]
        return self._load_file(idx)

# ==================== Manifests & checkpoints ====================
def write_manifest_once(save_dir: str, split: str, files: List[str], rank: int):
    man = os.path.join(save_dir, f"manifest_{split}.txt")
    if rank == 0 and not os.path.exists(man):
        os.makedirs(save_dir, exist_ok=True)
        with open(man, "w") as f:
            for p in files: f.write(p + "\n")
    barrier()

def load_manifest(save_dir: str, split: str) -> List[str]:
    man = os.path.join(save_dir, f"manifest_{split}.txt")
    with open(man) as f:
        return [ln.strip() for ln in f if ln.strip()]

def save_latest(save_dir: str, sae, opt, global_step: int, files_seen: int, world_rank: int, write_model: bool = True):
    latest_model = os.path.join(save_dir, "latest.pt")
    files_json   = os.path.join(save_dir, f"latest_files_rank{world_rank:02d}.json")
    
    is_fsdp = isinstance(sae, FSDP)
    model_sd = None
    optim_sd = None

    if write_model:
        if is_fsdp:
            ctx = FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT,
                                       state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                       optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True))
        else:
            ctx = nullcontext()

        with ctx:
            if is_fsdp:
                model_sd = sae.state_dict()
                try:
                    optim_sd = FSDP.optim_state_dict(sae, opt)
                except Exception:
                    optim_sd = opt.state_dict()
            elif world_rank == 0:
                base = sae.module if hasattr(sae, "module") else sae
                model_sd = base.state_dict()
                optim_sd = opt.state_dict()

        if world_rank == 0:
            sd = {"model": model_sd, "step": int(global_step)}
            if optim_sd is not None:
                sd["optim"] = optim_sd
            torch.save(sd, latest_model)

    # always write the lightweight per-rank breadcrumb
    with open(files_json, "w") as f:
        json.dump({"files_seen": int(files_seen), "step": int(global_step)}, f)

def load_latest(save_dir: str, sae, opt, world_rank: int) -> Tuple[int,int]:
    latest_model = os.path.join(save_dir, "latest.pt")
    files_json   = os.path.join(save_dir, f"latest_files_rank{world_rank:02d}.json")
    global_step = 0; files_seen = 0
    if os.path.exists(latest_model):
        ckpt = torch.load(latest_model, map_location="cpu")
        with (FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT) if isinstance(sae, FSDP) else nullcontext()):
            (sae.module if hasattr(sae, "module") else sae).load_state_dict(ckpt["model"], strict=False)
        osd = ckpt.get("optim", None)
        if osd is not None:
            if isinstance(sae, FSDP):
                FSDP.load_optim_state_dict(opt, osd)
            else:
                opt.load_state_dict(osd)
        global_step = int(ckpt.get("step", 0))
    if os.path.exists(files_json):
        with open(files_json) as f: j = json.load(f)
        files_seen = int(j.get("files_seen", 0))
        global_step = max(global_step, int(j.get("step", global_step)))
    return global_step, files_seen

def save_best_model(save_dir: str, sae, val_nmse: float, step: int, world_rank: int, tag: str = "best.pt", *, write_meta: bool = True):
    # For FSDP, ALL ranks must enter the state_dict context and call state_dict().
    # Rank 0 will get the full dict (rank0_only=True), others get empty.
    is_fsdp = isinstance(sae, FSDP)

    if is_fsdp:
        ctx = FSDP.state_dict_type(sae, StateDictType.FULL_STATE_DICT,
                                   state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
    else:
        ctx = nullcontext()

    with ctx:
        if is_fsdp:
            # Collective call
            model_sd = sae.state_dict()
        else:
            # Non-FSDP: only rank 0 acts
            if world_rank == 0:
                base = sae.module if hasattr(sae, "module") else sae
                model_sd = base.state_dict()
            else:
                model_sd = None

    if world_rank == 0:
        sd = {"model": model_sd, "step": int(step), "val_nmse": float(val_nmse)}
        torch.save(sd, os.path.join(save_dir, tag))
        if write_meta:
            with open(os.path.join(save_dir, "best_meta.json"), "w") as f:
                json.dump({"step": int(step), "val_nmse": float(val_nmse), "path": tag}, f)

# ==================== Optimizer ====================
def make_optimizer(cfg: TrainConfig, params):
    name = cfg.optimizer.lower()
    if name == "adam8bit":
        if bnb is not None:
            return bnb.optim.Adam8bit(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.wd)
        else:
            print("[warn] bitsandbytes unavailable; falling back to Adam.")
            return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.wd)
    elif name == "adafactor":
        if Adafactor is None:
            print("[warn] Adafactor not available; using Adam.")
            return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.wd)
        return Adafactor(params, lr=cfg.lr, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=cfg.wd)
    else:
        return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.wd)

# ==================== Training ====================
def _allreduce_or_bool(mask: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()): return mask
    x = mask.to(dtype=torch.int32); dist.all_reduce(x, op=dist.ReduceOp.SUM); return (x > 0)

# def write_rank_progress(save_dir: str, files_seen: int, step: int, rank: int):
#     path = os.path.join(save_dir, f"latest_files_Progress_rank{rank:02d}.json")
#     tmp  = path + ".tmp"
#     with open(tmp, "w") as f:
#         json.dump({"files_seen": int(files_seen), "step": int(step)}, f)
#         f.flush(); os.fsync(f.fileno())
#     os.replace(tmp, path)

def train(cfg: TrainConfig,
          windows_per_file_hint: int = 8,
          allow_last_small_batch: bool = False,
          resume: bool = True,
          num_workers_train: int = 8,
          num_workers_val: int = 4,
          prefetch_train: int = 4,
          prefetch_val: int = 2,
          pin_memory: bool = True,
          persistent_workers: bool = True,
          renorm_every: int = 0):  # 0 = off; otherwise steps cadence
    assert cfg.micro_bsz % windows_per_file_hint == 0, "--micro_bsz must be a multiple of --windows_per_file_hint"
    set_seed(42); os.makedirs(cfg.save_dir, exist_ok=True)

    env = init_sae_distributed(cfg)
    

    rank, world = env["rank"], env["world"]

    sae_device = env["sae_device"]; sae_dtype = str2dtype(cfg.dtype)
    ema_enabled = (cfg.ema_enable and not cfg.sae_fsdp)

    # Data
    if rank == 0: print("[allinone_v3] Loading datasets...")
    
    # Calculate max files if preloading train
    train_max_files = None
    if getattr(cfg, "preload_train", False):
        files_per_step_est = max(1, cfg.micro_bsz // windows_per_file_hint)
        # 1.1x buffer to ensure we don't run out due to rounding or small files
        train_max_files = int(cfg.train_steps * files_per_step_est * world * 1.1)
        est_size_gb = (train_max_files * 8.0) / 1024.0 # Approx 8MB per file
        if rank == 0:
            print(f"[allinone_v3] --preload_train enabled.")
            print(f"              Limiting train set to ~{train_max_files} files (sufficient for {cfg.train_steps} steps)")
            print(f"              Est. RAM usage: {est_size_gb:.1f} GB. Ensure you have enough RAM!")
            
    train_dataset = ActivationDatasetFiles(cfg.activations_dir, "train", 
                                           preload=getattr(cfg, "preload_train", False),
                                           rank=rank, world_size=world,
                                           max_files=train_max_files)
    # Manually shard validation set so we only preload 1/Nth of the data per rank
    val_dataset   = ActivationDatasetFiles(cfg.activations_dir, "val", preload=True, rank=rank, world_size=world, max_files=None)
    print(f"[rank{rank}] local_rank={env['local_rank']} world={world} device={sae_device} n_train_files={len(train_dataset)} n_val_files={len(val_dataset)}")

    # create a per-rank heartbeat so you can ls for it
    # hb = os.path.join(cfg.save_dir, f"rank_heartbeat_{rank:02d}.txt")
    # with open(hb, "w") as f:
    #     f.write(f"hello from rank {rank}\n")
    d_model = train_dataset.model_config.get("hidden_size", cfg.d_model or 0)
    if d_model == 0:
        # Only rank 0 needs to load the first file to infer d_model, but safe if others do it too (won't preload)
        probe = torch.load(train_dataset.batch_files[0], map_location="cpu")
        acts = probe.get("activations", probe.get("hidden_states"))
        d_model = (acts[0].shape[-1] if isinstance(acts, list) else acts.shape[-1])

    if rank == 0:
        print(f"[allinone_v3] d_model={d_model} n_latents={cfg.n_latents} k={cfg.k} k_aux={cfg.k_aux} dtype={cfg.dtype}")
        with open(os.path.join(cfg.save_dir, "cfg.json"), "w") as f: json.dump(asdict(cfg), f, indent=2)

    if not getattr(cfg, "preload_train", False):
        write_manifest_once(cfg.save_dir, "train", train_dataset.batch_files, rank)
        # write_manifest_once(cfg.save_dir, "val",   val_dataset.batch_files, rank) # Skip manifest for sharded val
        train_dataset.batch_files = load_manifest(cfg.save_dir, "train")
        # val_dataset.batch_files   = load_manifest(cfg.save_dir, "val") # Skip loading manifest for sharded val
    else:
        if rank == 0: print("[allinone_v3] Skipping manifest for preloaded train set (using per-rank shards)")

    # If we preloaded train, we manually sharded it in __init__, so we use a local sampler (None)
    # Otherwise, we use DistributedSampler to shard it now.
    use_dist_sampler = (dist.is_available() and dist.is_initialized()) and not getattr(cfg, "preload_train", False)
    train_sampler = DistributedSampler(train_dataset, shuffle=False) if use_dist_sampler else None
    
    # For validation:
    # - We manually sharded the files in ActivationDatasetFiles(..., rank=rank, world_size=world)
    # - So each rank has a DISJOINT set of files.
    # - We use a standard SequentialSampler (default for DataLoader(sampler=None))
    # - This is perfectly fine for FSDP validation because:
    #   1. We gather metrics at the end using dist.all_reduce(..., op=SUM)
    #   2. Every sample is seen exactly once across the world (no overlap, no gaps)
    val_sampler   = None  

    def collate_file(batch): return batch[0]
    
    def collate_files_concat(file_batch):
    # file_batch: list of dicts, each is one .pt file -> {"activations":[8,T,D], "masks":[8,T]}
        acts = [b["activations"] for b in file_batch]
        masks = [b["masks"]       for b in file_batch]
        return {
            "activations": torch.cat(acts,  dim=0),   # [F*8, T, D]
            "masks":       torch.cat(masks, dim=0),   # [F*8, T]
        }
    # Validation: choose how many files to concatenate to reach desired rows
    assert cfg.val_micro_bsz % windows_per_file_hint == 0, \
        "--val_micro_bsz must be a multiple of windows_per_file_hint (usually 8)"
    val_files_per_step = max(1, cfg.val_micro_bsz // windows_per_file_hint)
    
    # If sampler is None (e.g. preload_train=True), we might want shuffle=True?
    # But original code used DistributedSampler(shuffle=False), so we stick to shuffle=False to minimize changes.
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, shuffle=False,
                              num_workers=num_workers_train, prefetch_factor=prefetch_train,
                              persistent_workers=persistent_workers, pin_memory=pin_memory,
                              collate_fn=collate_file, drop_last=False)
    val_workers = 0 if getattr(val_dataset, "preload", False) else num_workers_val
    val_loader   = DataLoader(val_dataset, batch_size=val_files_per_step, sampler=val_sampler, shuffle=False,
                              num_workers=val_workers, 
                              prefetch_factor=(prefetch_val if val_workers > 0 else None),
                              persistent_workers=(persistent_workers if val_workers > 0 else False), 
                              pin_memory=pin_memory,
                              collate_fn=collate_files_concat, drop_last=False)

    # Model
    sae = SparseAutoencoder(d_model=d_model, n_latents=cfg.n_latents, k=cfg.k,
                            relu_after_topk=cfg.relu_after_topk, dtype=sae_dtype).to(sae_device)

    if cfg.sae_fsdp:
        mp = MixedPrecision(param_dtype=sae_dtype, reduce_dtype=sae_dtype, buffer_dtype=sae_dtype)
        sae = FSDP(sae, device_id=sae_device, mixed_precision=mp,
                   sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
    if rank == 0:
        base = sae.module if isinstance(sae, FSDP) else sae
        print(f"[allinone_v3][model] params: {count_parameters(base):,} FSDP={cfg.sae_fsdp}")

    opt = make_optimizer(cfg, sae.parameters())
    ema = EMA(sae.module if isinstance(sae, FSDP) else sae, decay=cfg.ema_decay) if ema_enabled else None

    # Resume
    global_step = 0; files_seen = 0
    if resume:
        global_step, files_seen = load_latest(cfg.save_dir, sae, opt, rank)
        if rank == 0: print(f"[allinone_v3][resume] step={global_step} files_seen={files_seen}")

    train_iter = iter(train_loader)
    skipped = 0
    while skipped < files_seen:
        try: next(train_iter); skipped += 1
        except StopIteration:
            train_iter = iter(train_loader)

    best_val_nmse = float("inf")
    windows_per_file = windows_per_file_hint
    files_per_step = cfg.micro_bsz // windows_per_file


    # ---- Accumulator for dead-latent window ----
    base_mod = sae.module if isinstance(sae, FSDP) else sae
    fired_accum = torch.zeros((base_mod.n_latents,), dtype=torch.bool, device=sae_device)

    while global_step < cfg.train_steps:
        xs, ms = [], []
        got = 0
        while got < files_per_step:
            try: fb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader); fb = next(train_iter)
            acts = fb["activations"]; mask = fb["masks"]; Bf = acts.shape[0]; files_seen += 1
            
            if Bf != windows_per_file:
                if (got == 0) and allow_last_small_batch:
                    windows_per_file = Bf; files_per_step = 1
                    if rank == 0: print(f"[allinone_v3] taking small file Bf={Bf}")
                else:
                    continue
            xs.append(acts); ms.append(mask); got += 1

        x = torch.cat(xs, dim=0).to(sae_device, dtype=sae_dtype, non_blocking=True)
        m = torch.cat(ms, dim=0).to(sae_device, dtype=torch.bool, non_blocking=True)

        # ---- OPTION A: DETECT + MASK NaN TOKENS, THEN SANITIZE ----
        nan_mask   = torch.isnan(x)                 # [B,T,D]
        nan_tokens = nan_mask.any(dim=-1) & m       # [B,T] only valid tokens
        if nan_tokens.any():
            m = m & ~nan_tokens                     # drop these tokens from loss
            # sanitize so forward math can't propagate NaNs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        vt = int(m.sum().item())
        did_update = vt > 0

        step_start_time = time.time()
        if did_update:
            sae.train(); opt.zero_grad(set_to_none=True)
            x_hat, z, pre, aux_loss = sae(x, k_aux=cfg.k_aux, mask=m)

            x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- Main reconstruction loss (masked) ----
            rec = mse_masked(x_hat.float(), x.float(), m)
            loss = rec + (cfg.aux_coeff * aux_loss if aux_loss is not None else 0.0)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            if ema_enabled and ema is not None: ema.update(base_mod)

            with torch.no_grad():
                fired_any = ((z > 0) & m.unsqueeze(-1)).any(dim=(0,1))
                fired_accum |= fired_any.to(sae_device)
        else:
            # No valid tokens on this rank → skip compute, but keep cadence
            rec = torch.tensor(0.0, device=sae_device)

        # optional debug: ensure global_step stays aligned across ranks
        if dist.is_available() and dist.is_initialized():
            gs_t = torch.tensor([global_step], device=sae_device, dtype=torch.int64)
            dist.all_reduce(gs_t, op=dist.ReduceOp.MAX)
            # all ranks should have identical global_step; if not, this assert will trip fast:
            assert gs_t.item() == global_step, f"desync: local step {global_step} vs max {gs_t.item()}"

        if renorm_every and (global_step % renorm_every == 0) and global_step > 0:
            with torch.no_grad():
                with fsdp_ctx(sae):
                    base_mod.renorm_decoder_columns()

        if (global_step % cfg.dead_threshold_steps == 0) and (global_step > 0):
            fired_global = _allreduce_or_bool(fired_accum)
            dead_mask = (~fired_global).to(torch.bool)
            n_reinit = update_dead_latents(sae, dead_mask, opt=opt, ema=(ema if ema_enabled else None))
            if rank == 0 and n_reinit > 0:
                print(json.dumps({"event":"update_dead_latents","count":int(n_reinit),"step":int(global_step)}))
            fired_accum.zero_()

        if rank == 0 and global_step == 0:
            print({
                "x_finite": torch.isfinite(x).all().item(),
                "x_nan": (~torch.isfinite(x)).sum().item(),
                "x_hat_finite": torch.isfinite(x_hat).all().item(),
                "pre_finite": torch.isfinite(pre).all().item(),
            })

        step_time = time.time() - step_start_time
        if rank == 0 and (global_step % cfg.log_every == 0):
            print(json.dumps({"step": int(global_step),
                              "loss": float(rec.detach().cpu().item()),
                              "nmse": float(nmse_masked(x_hat.detach().float(), x.float(), m).cpu().item()),
                              "aux": (float(aux_loss.detach().cpu().item()) if aux_loss is not None else None),
                              "valid_tokens": int(vt),
                              "files_seen": int(files_seen),
                              "batch_B": int(x.shape[0]),
                              "step_time_sec": round(step_time, 3)}))
        

        if (global_step % cfg.val_every == 0 and global_step > 0) or (global_step == cfg.train_steps - 1):
            # print(f"[rank{rank}] [val] start step={global_step}")
            val_start_time = time.time()
            sae.eval()
            vtotal_mse = 0.0; vtotal_nmse = 0.0; vtotal_tokens = 0
            with torch.no_grad():
                for vb in val_loader:
                    va = vb["activations"].to(sae_device, dtype=sae_dtype, non_blocking=True)
                    vm = vb["masks"].to(sae_device, dtype=torch.bool, non_blocking=True)

                    # Detect & mask NaN tokens, then sanitize
                    v_nan = torch.isnan(va)
                    v_nan_tokens = v_nan.any(dim=-1) & vm
                    if v_nan_tokens.any():
                        vm = vm & ~v_nan_tokens
                        va = torch.nan_to_num(va, nan=0.0, posinf=0.0, neginf=0.0)
                    if vm.sum() == 0: continue

                    vx_hat, vz, vpre, _ = sae(va, k_aux=0)   # no aux in eval
                    vx_hat = torch.nan_to_num(vx_hat, nan=0.0, posinf=0.0, neginf=0.0)

                    vtoks = int(vm.sum().item())
                    vtotal_mse  += float(mse_masked(vx_hat.float(), va.float(), vm).cpu()) * vtoks
                    vtotal_nmse += float(nmse_masked(vx_hat.float(), va.float(), vm).cpu()) * vtoks
                    vtotal_tokens += vtoks
            # vloss = vtotal_mse  / max(1, vtotal_tokens)
            # vnmse = vtotal_nmse / max(1, vtotal_tokens)

            # optional: reduce validation metric so rank0 reflects global view
            if dist.is_available() and dist.is_initialized():
                # Reduce totals (MSE sum, NMSE sum, token count)
                metrics_t = torch.tensor([vtotal_mse, vtotal_nmse, vtotal_tokens], device=sae_device, dtype=torch.float64)
                dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
                vtotal_mse_global = metrics_t[0].item()
                vtotal_nmse_global = metrics_t[1].item()
                vtotal_tokens_global = metrics_t[2].item()

                vloss = vtotal_mse_global / max(1, vtotal_tokens_global)
                vnmse = vtotal_nmse_global / max(1, vtotal_tokens_global)
            else:
                vloss = vtotal_mse  / max(1, vtotal_tokens)
                vnmse = vtotal_nmse / max(1, vtotal_tokens)

            val_time = time.time() - val_start_time
            # print(f"[rank{rank}] [val] end   step={global_step} time={val_time:.3f}s")

            if rank == 0: print(json.dumps({"val_loss": vloss, "val_nmse": vnmse, "step": int(global_step), "val_time_sec": round(val_time, 3)}))
            if global_step % 10000 == 0:
                save_best_model(cfg.save_dir, sae, vnmse, global_step, rank, tag=f"val_{global_step}.pt", write_meta=False)
            if vnmse < best_val_nmse:
                best_val_nmse = vnmse
                save_best_model(cfg.save_dir, sae, vnmse, global_step, rank, tag=f"best_{global_step}.pt")
            barrier()

        global_step += 1

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

@torch.no_grad()
def update_dead_latents(sae: nn.Module, dead_mask: torch.Tensor, opt=None, ema=None) -> int:
    base = sae.module if isinstance(sae, FSDP) else sae
    N = base.n_latents
    if dead_mask.numel() != N:
        raise ValueError(f"dead_mask has {dead_mask.numel()} entries, expected {N}")
    idx = torch.nonzero(dead_mask, as_tuple=False).view(-1)
    if idx.numel() == 0: return 0
    ctx = FSDP.summon_full_params(sae) if isinstance(sae, FSDP) else nullcontext()
    with ctx:
        W_enc = base.W_enc; W_dec = base.W_dec; b_enc = base.b_enc
        w_new = torch.empty((idx.numel(), W_enc.shape[1]), device=W_enc.device, dtype=W_enc.dtype)
        nn.init.xavier_uniform_(w_new)
        W_enc[idx, :] = w_new
        W_dec[:, idx] = w_new.t()
        b_enc[idx] = 0.0
        if opt is not None:
            for g in opt.param_groups:
                for p in g["params"]:
                    st = opt.state.get(p, None)
                    if not st: continue
                    for k in ("exp_avg","exp_avg_sq"):
                        buf = st.get(k, None)
                        if buf is None: continue
                        if p is W_enc or p is W_dec or p is b_enc:
                            buf.zero_()
        if ema is not None and hasattr(ema, "shadow"):
            for n,p in base.named_parameters():
                if n in ema.shadow:
                    ema.shadow[n].data.copy_(p.data)
    return int(idx.numel())

# ==================== CLI ====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activations_dir", type=str, default="activations")
    p.add_argument("--d_model", type=int, default=5376)
    p.add_argument("--n_latents", type=int, default=262144)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--k_aux", type=int, default=512)
    p.add_argument("--dead_threshold_steps", type=int, default=200)
    p.add_argument("--relu_after_topk", action="store_true")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=6.25e-10)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_enable", action="store_true")
    p.add_argument("--aux_coeff", type=float, default=1/32)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--micro_bsz", type=int, default=8)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--val_micro_bsz", type=int, default=16,
               help="validation rows per step (must be a multiple of windows_per_file_hint)")

    p.add_argument("--optimizer", type=str, default="adafactor", choices=["adam","adafactor","adam8bit"])
    p.add_argument("--sae_fsdp", action="store_true")
    p.add_argument("--sae_fsdp_devices", type=str, default="0,1,2,3")
    p.add_argument("--save_dir", type=str, default="sae_multinode/run05-newdata-deadlatents")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32","float16","bfloat16"])
    p.add_argument("--windows_per_file_hint", type=int, default=8)
    p.add_argument("--allow_last_small_batch", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--num_workers_train", type=int, default=8)
    p.add_argument("--num_workers_val", type=int, default=2)
    p.add_argument("--prefetch_train", type=int, default=512)
    p.add_argument("--prefetch_val", type=int, default=2)
    p.add_argument("--preload_train", action="store_true", help="Preload training data into RAM (limits dataset to train_steps)")
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--no-pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false")
    p.add_argument("--renorm_every", type=int, default=0, help="0=off; otherwise L2-normalize decoder columns every N steps")
    a = p.parse_args()

    cfg = TrainConfig(
        activations_dir=a.activations_dir,
        save_dir=a.save_dir,
        d_model=a.d_model,
        n_latents=a.n_latents,
        k=a.k,
        k_aux=a.k_aux,
        dead_threshold_steps=a.dead_threshold_steps,
        relu_after_topk=a.relu_after_topk,
        lr=a.lr, beta1=a.beta1, beta2=a.beta2, eps=a.eps, wd=a.wd,
        grad_clip=a.grad_clip,
        ema_decay=a.ema_decay, ema_enable=a.ema_enable,
        aux_coeff=a.aux_coeff,
        train_steps=a.train_steps,
        micro_bsz=a.micro_bsz,
        log_every=a.log_every, val_every=a.val_every, val_micro_bsz=a.val_micro_bsz,
        optimizer=a.optimizer,
        sae_fsdp=a.sae_fsdp, sae_fsdp_devices=a.sae_fsdp_devices,
        dtype=a.dtype,
    )
    # Patch in the non-config arg for convenience
    cfg.preload_train = a.preload_train
    return cfg, a

if __name__ == "__main__":
    cfg, a = parse_args()
    train(cfg,
          windows_per_file_hint=a.windows_per_file_hint,
          allow_last_small_batch=a.allow_last_small_batch,
          resume=a.resume,
          num_workers_train=a.num_workers_train,
          num_workers_val=a.num_workers_val,
          prefetch_train=a.prefetch_train,
          prefetch_val=a.prefetch_val,
          pin_memory=a.pin_memory,
          persistent_workers=a.persistent_workers,
          renorm_every=a.renorm_every)
