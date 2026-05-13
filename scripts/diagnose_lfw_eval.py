"""Diagnostic v2: sample BOTH genuine and impostor pairs separately,
verify the backbone has actually been updated by federated training,
and report whether the 100% TAR@FAR is real or a degenerate result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.dp_utils import load_eval_pairs
from federated_project.federation import create_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Round-2 LFW diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--eval-image-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=300,
        help="How many pairs of each class (genuine, impostor) to embed.",
    )
    return parser


def load_finetuned_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=device,
    )
    model.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=True)
    saved_W = ckpt["W_matrix"].to(device=device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model, ckpt


def load_raw_pretrained_model(pretrained_name: str, num_clients: int, device: torch.device):
    """Load the raw pre-trained model (no federated updates)."""
    model = create_model(num_clients=num_clients, pretrained=pretrained_name, device=device)
    model.eval()
    return model


@torch.no_grad()
def embed_one(model, path: Path, transform, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return model(tensor).squeeze(0).detach().cpu()


def describe(label: str, scores: np.ndarray) -> None:
    print(f"  {label} (n={len(scores)}):")
    print(f"    min={scores.min():.4f}  max={scores.max():.4f}")
    print(f"    mean={scores.mean():.4f}  std={scores.std():.4f}")
    print(f"    quantiles: 5%={np.quantile(scores,0.05):.4f}  "
          f"50%={np.quantile(scores,0.5):.4f}  95%={np.quantile(scores,0.95):.4f}")


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_finetuned_model(Path(args.checkpoint), device)

    print(f"Loading pairs: {args.pairs_file}")
    pairs = load_eval_pairs(args.pairs_file, args.eval_image_dir)
    genuine_pairs = [p for p in pairs if p.is_same_person]
    impostor_pairs = [p for p in pairs if not p.is_same_person]
    print(f"  total={len(pairs)}  genuine={len(genuine_pairs)}  impostor={len(impostor_pairs)}")
    print()

    transform = get_default_transforms(train=False)
    cache: dict[str, torch.Tensor] = {}

    def get_emb(m, path):
        key = f"{id(m)}::{path}"
        if key not in cache:
            cache[key] = embed_one(m, Path(path), transform, device)
        return cache[key]

    n = args.sample_per_class

    # ------------------------------------------------------------------
    # 1. Score distributions on the FINETUNED model
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"  Finetuned model: score distributions on {n} genuine + {n} impostor pairs")
    print("=" * 70)
    g_scores = []
    for p in genuine_pairs[:n]:
        ea = get_emb(model, p.image_a)
        eb = get_emb(model, p.image_b)
        g_scores.append(float(torch.dot(ea, eb).item()))
    i_scores = []
    for p in impostor_pairs[:n]:
        ea = get_emb(model, p.image_a)
        eb = get_emb(model, p.image_b)
        i_scores.append(float(torch.dot(ea, eb).item()))

    g_arr = np.array(g_scores)
    i_arr = np.array(i_scores)
    describe("Genuine ", g_arr)
    describe("Impostor", i_arr)

    overlap_lo = max(g_arr.min(), i_arr.min())
    overlap_hi = min(g_arr.max(), i_arr.max())
    if overlap_hi > overlap_lo:
        n_g_in = int(((g_arr >= overlap_lo) & (g_arr <= overlap_hi)).sum())
        n_i_in = int(((i_arr >= overlap_lo) & (i_arr <= overlap_hi)).sum())
        print(f"  Overlap [{overlap_lo:.4f}, {overlap_hi:.4f}]: "
              f"{n_g_in} genuine / {n_i_in} impostor in this region")
    else:
        print(f"  *** NO OVERLAP between genuine ({g_arr.min():.4f}+) "
              f"and impostor ({i_arr.max():.4f}-) ***")
        print(f"  Gap: {g_arr.min() - i_arr.max():.4f}")
        print(f"  This explains TAR=1.0 at any non-zero FAR.")

    # ------------------------------------------------------------------
    # 2. Compare against the RAW pre-trained backbone (no federated updates)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  Raw pre-trained baseline ({ckpt['pretrained']}): same {n}+{n} pairs")
    print("=" * 70)

    raw_model = load_raw_pretrained_model(
        pretrained_name=str(ckpt["pretrained"]),
        num_clients=int(ckpt["num_clients"]),
        device=device,
    )

    rg_scores = []
    for p in genuine_pairs[:n]:
        ea = get_emb(raw_model, p.image_a)
        eb = get_emb(raw_model, p.image_b)
        rg_scores.append(float(torch.dot(ea, eb).item()))
    ri_scores = []
    for p in impostor_pairs[:n]:
        ea = get_emb(raw_model, p.image_a)
        eb = get_emb(raw_model, p.image_b)
        ri_scores.append(float(torch.dot(ea, eb).item()))

    rg_arr = np.array(rg_scores)
    ri_arr = np.array(ri_scores)
    describe("Genuine  (pretrained)", rg_arr)
    describe("Impostor (pretrained)", ri_arr)

    # ------------------------------------------------------------------
    # 3. Backbone delta check: did training actually change the weights?
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Backbone delta: finetuned vs raw pretrained")
    print("=" * 70)
    raw_state = raw_model.feature_extractor.state_dict()
    fine_state = model.feature_extractor.state_dict()
    total_l2_diff = 0.0
    total_l2_norm = 0.0
    n_changed = 0
    n_total = 0
    for k in raw_state:
        if not raw_state[k].is_floating_point():
            continue
        d = (fine_state[k].to(raw_state[k].device) - raw_state[k]).float()
        l2_diff = float(d.norm().item())
        l2_norm = float(raw_state[k].float().norm().item())
        total_l2_diff += l2_diff ** 2
        total_l2_norm += l2_norm ** 2
        n_total += 1
        if l2_diff > 1e-6:
            n_changed += 1
    print(f"  {n_changed}/{n_total} float tensors differ from pretrained (>1e-6)")
    if total_l2_norm > 0:
        rel = (total_l2_diff ** 0.5) / (total_l2_norm ** 0.5)
        print(f"  Global relative L2 change: {rel:.6f}")
        if rel < 1e-5:
            print("  *** SUSPICIOUS: backbone is essentially unchanged. ***")
            print("      Federated updates did not propagate.")
        elif rel < 1e-3:
            print("  Backbone changed only slightly - mostly using pretrained features.")
        else:
            print("  Backbone has been substantively updated. ✓")

    # ------------------------------------------------------------------
    # 4. Side-by-side summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Summary comparison")
    print("=" * 70)
    print(f"  {'Model':<14} {'gen mean':>10} {'gen std':>10} "
          f"{'imp mean':>10} {'imp std':>10} {'gap':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'pretrained':<14} {rg_arr.mean():>10.4f} {rg_arr.std():>10.4f} "
          f"{ri_arr.mean():>10.4f} {ri_arr.std():>10.4f} "
          f"{(rg_arr.mean()-ri_arr.mean()):>10.4f}")
    print(f"  {'finetuned':<14} {g_arr.mean():>10.4f} {g_arr.std():>10.4f} "
          f"{i_arr.mean():>10.4f} {i_arr.std():>10.4f} "
          f"{(g_arr.mean()-i_arr.mean()):>10.4f}")


if __name__ == "__main__":
    main()
