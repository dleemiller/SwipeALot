"""Grid search attention distillation metrics over a validation subset.

This script loads a HuggingFace checkpoint (trust_remote_code=True), runs inference with
attention extraction, and evaluates how "peaky" and stable the resulting per-character
attention profiles are under different pooling/temperature settings.

Example:
    uv run attention-grid \\
      --checkpoint checkpoints/base_20251217_123323/checkpoint-6000 \\
      --num-samples 50000 \\
      --batch-size 16 \\
      --output-csv profiling_results/attention_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from swipealot.analysis.attention_capture import get_all_layer_attentions


def _sharpen_distribution(p: torch.Tensor, temperature: float, eps: float = 1e-12) -> torch.Tensor:
    if temperature == 1.0:
        return p
    # p is a distribution; do log-space sharpening for numerical stability.
    logp = torch.log(p.clamp_min(eps))
    return torch.softmax(logp / float(temperature), dim=-1)


def _topk_mass(p: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), p.shape[-1])
    return p.topk(k, dim=-1).values.sum(dim=-1)


def _total_variation(p: torch.Tensor) -> torch.Tensor:
    return torch.abs(p[..., 1:] - p[..., :-1]).sum(dim=-1)


def _peak_to_median(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    peak = p.max(dim=-1).values
    med = p.median(dim=-1).values
    return peak / (med + eps)


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * torch.log(p + eps)).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search attention distillation metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="HF checkpoint directory")
    parser.add_argument("--dataset-name", type=str, default="futo-org/swipe.futo.org")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--hf-home", type=str, default=".hf_cache")
    parser.add_argument("--last-k-layers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda; default auto")
    args = parser.parse_args()

    hf_home = Path(args.hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_home / "modules"))

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"--checkpoint must be a directory, got: {checkpoint}")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model.eval()

    config = model.config
    path_len = int(getattr(config, "max_path_len", 128))
    char_len = int(getattr(config, "max_char_len", 48))
    n_layers = int(getattr(config, "n_layers", 12))
    n_heads = int(getattr(config, "n_heads", 12))

    # Exclude special/non-content query tokens from supervision metrics.
    tok = processor.tokenizer
    exclude_ids = {
        int(tok.pad_token_id),
        int(tok.eos_token_id),
        int(tok.mask_token_id),
    }
    try:
        punc_id = tok.convert_tokens_to_ids("[PUNC]")
        if isinstance(punc_id, int) and punc_id >= 0:
            exclude_ids.add(int(punc_id))
    except Exception:
        pass

    last_k_values = sorted({int(k) for k in args.last_k_layers if int(k) > 0})
    temps = [float(t) for t in args.temperatures]

    # Accumulators per (k, T)
    keys = [(k, t) for k in last_k_values for t in temps]
    sums = {
        key: {
            "entropy": 0.0,
            "eff_positions": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "peak_to_median": 0.0,
            "tv": 0.0,
            "path_mass": 0.0,
            "count": 0,
        }
        for key in keys
    }

    ds = load_dataset(args.dataset_name, split=f"{args.split}[:{args.num_samples}]")

    def collate(examples: list[dict]) -> dict[str, torch.Tensor]:
        paths = [ex["data"] for ex in examples]
        texts = [ex["word"] for ex in examples]
        batch = processor(path_coords=paths, text=texts, return_tensors="pt")
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def batches():
        batch = []
        for ex in ds:
            batch.append(ex)
            if len(batch) >= args.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    pbar = tqdm(total=len(ds), desc="Batches", unit="samples")
    for raw_batch in batches():
        inputs = collate(raw_batch)
        _, attentions = get_all_layer_attentions(model, inputs)

        # attentions: tuple[L] of [B, H, S, S]
        if len(attentions) != n_layers:
            n_layers = len(attentions)

        seq_len = int(inputs["attention_mask"].shape[1])
        expected_seq = 1 + path_len + 1 + char_len
        if seq_len != expected_seq:
            raise RuntimeError(f"Unexpected seq_len={seq_len}, expected {expected_seq}")

        path_start = 1
        path_end = 1 + path_len
        char_start = 1 + path_len + 1
        char_end = char_start + char_len

        # Key mask (valid path positions) and query mask (valid text positions).
        attn_mask = inputs["attention_mask"]
        path_key_mask = attn_mask[:, path_start:path_end].to(torch.float32)  # [B, P]
        query_attn_mask = attn_mask[:, char_start:char_end].to(torch.bool)  # [B, C]

        input_ids = inputs["input_ids"]  # [B, C]
        query_keep = query_attn_mask.clone()
        for token_id in exclude_ids:
            query_keep = query_keep & (input_ids != int(token_id))

        # Build head-mean char->path attention per layer: [L, B, C, P]
        per_layer = []
        for a in attentions:
            # Promote to float32 for metrics even if model runs bf16.
            a = a.to(torch.float32)
            a_mean = a.mean(dim=1)  # [B, S, S]
            per_layer.append(a_mean[:, char_start:char_end, path_start:path_end])
        attn_layers = torch.stack(per_layer, dim=0)  # [L, B, C, P]

        # Precompute last-k means.
        for k in last_k_values:
            kk = min(k, attn_layers.shape[0])
            base = attn_layers[-kk:].mean(dim=0)  # [B, C, P]

            # Mask invalid path keys; compute pre-renorm path mass.
            base = base * path_key_mask.unsqueeze(1)
            path_mass = base.sum(dim=-1)  # [B, C]

            # Keep only valid content queries.
            base = base * query_keep.unsqueeze(-1).to(base.dtype)
            denom = base.sum(dim=-1, keepdim=True)
            valid = (denom.squeeze(-1) > 0) & query_keep

            p = torch.where(denom > 0, base / denom, torch.zeros_like(base))

            for t in temps:
                p_t = _sharpen_distribution(p, t)

                ent = _entropy(p_t)
                eff = torch.exp(ent)
                top5 = _topk_mass(p_t, 5)
                top10 = _topk_mass(p_t, 10)
                ratio = _peak_to_median(p_t)
                tv = _total_variation(p_t)

                # Mask to valid queries
                v = valid
                if v.any():
                    ent_v = ent[v]
                    eff_v = eff[v]
                    top5_v = top5[v]
                    top10_v = top10[v]
                    ratio_v = ratio[v]
                    tv_v = tv[v]
                    pm_v = path_mass[v]

                    key = (k, float(t))
                    sums[key]["entropy"] += float(ent_v.sum().item())
                    sums[key]["eff_positions"] += float(eff_v.sum().item())
                    sums[key]["top5"] += float(top5_v.sum().item())
                    sums[key]["top10"] += float(top10_v.sum().item())
                    sums[key]["peak_to_median"] += float(ratio_v.sum().item())
                    sums[key]["tv"] += float(tv_v.sum().item())
                    sums[key]["path_mass"] += float(pm_v.sum().item())
                    sums[key]["count"] += int(v.sum().item())

        pbar.update(len(raw_batch))

    pbar.close()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "dataset",
                "split",
                "num_samples",
                "batch_size",
                "n_layers",
                "n_heads",
                "path_len",
                "char_len",
                "last_k_layers",
                "temperature",
                "num_queries",
                "mean_path_mass",
                "mean_entropy",
                "mean_eff_positions",
                "mean_top5_mass",
                "mean_top10_mass",
                "mean_peak_to_median",
                "mean_total_variation",
            ],
        )
        writer.writeheader()

        for (k, t), stats in sums.items():
            n = stats["count"]
            if n <= 0:
                continue
            writer.writerow(
                {
                    "checkpoint": str(checkpoint),
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "num_samples": args.num_samples,
                    "batch_size": args.batch_size,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "path_len": path_len,
                    "char_len": char_len,
                    "last_k_layers": k,
                    "temperature": t,
                    "num_queries": n,
                    "mean_path_mass": stats["path_mass"] / n,
                    "mean_entropy": stats["entropy"] / n,
                    "mean_eff_positions": stats["eff_positions"] / n,
                    "mean_top5_mass": stats["top5"] / n,
                    "mean_top10_mass": stats["top10"] / n,
                    "mean_peak_to_median": stats["peak_to_median"] / n,
                    "mean_total_variation": stats["tv"] / n,
                }
            )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
