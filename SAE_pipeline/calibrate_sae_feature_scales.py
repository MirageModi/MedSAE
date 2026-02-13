#!/usr/bin/env python3
"""Post-hoc calibration of SAE feature scales for visualisation/analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any
import json

import torch

from sae_train_NaNMasking import SparseAutoencoder, ActivationDatasetFiles


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "sae_state_dict" in ckpt and isinstance(ckpt["sae_state_dict"], dict):
            return ckpt["sae_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unrecognised checkpoint format")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate per-feature scales for a trained SAE")
    ap.add_argument("ckpt", help="Path to SAE checkpoint (best.pt, latest.pt, etc.)")
    ap.add_argument("activations_dir", help="Root directory containing activation shards")
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="Activation split to sample (default: train)")
    ap.add_argument("--device", default="cuda", help="Device for calibration (default: cuda)")
    ap.add_argument("--output", default=None,
                    help="Optional output checkpoint path; defaults to <ckpt>_calibrated.pt")
    ap.add_argument("--steps", type=int, default=None,
                    help="Max number of batches to scan (default: all)")
    ap.add_argument("--quantile", type=float, default=0.999,
                    help="High quantile used as robust scale (default: 0.999)")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--prefetch", type=int, default=2, help="DataLoader prefetch factor")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Number of activation files per batch (default: 1)")
    ap.add_argument("--k", type=int, default=None, help="Override SAE Top-K if missing in ckpt")
    ap.add_argument("--preload", action="store_true", help="Preload activations into RAM using threads (faster)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of activation files to load (useful with --preload to limit memory). If set, files are shuffled before selection.")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    cfg = {}
    if isinstance(ckpt, dict):
        cfg = ckpt.get("config", {}) or {}

    if not cfg:
        cfg_path = ckpt_path.with_name("cfg.json")
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse config file {cfg_path}") from exc

    k = args.k or cfg.get("k")
    if k is None:
        raise ValueError("Could not infer SAE k; please pass --k explicitly")

    W_enc = state_dict["W_enc"]
    n_latents, d_model = W_enc.shape
    relu_after = cfg.get("relu_after_topk", True)

    sae = SparseAutoencoder(
        d_model=d_model,
        n_latents=n_latents,
        k=k,
        relu_after_topk=relu_after,
        dtype=W_enc.dtype,
    )
    sae.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device)
    sae.to(device)
    sae.eval()

    dataset = ActivationDatasetFiles(
        args.activations_dir, 
        split=args.split, 
        preload=args.preload,
        max_files=args.max_files
    )

    def _collate(batch):
        return batch[0]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    print(f"Loaded Activation files. Now calibrating feature scales...")
    sae.calibrate_feature_scales(
        loader,
        device=device,
        steps=args.steps,
        quantile=args.quantile,
    )
    print(f"Calibrated feature scales. Now updating checkpoint...")
    sae_state = sae.state_dict()
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt["model"] = sae_state
        elif "sae_state_dict" in ckpt and isinstance(ckpt["sae_state_dict"], dict):
            ckpt["sae_state_dict"] = sae_state
        else:
            ckpt = sae_state
    else:
        ckpt = sae_state

    out_path = (
        Path(args.output)
        if args.output is not None
        else ckpt_path.with_name(ckpt_path.stem + "_calibrated.pt")
    )
    torch.save(ckpt, out_path)
    print(f"Calibrated checkpoint written to {out_path}")


if __name__ == "__main__":
    main()


