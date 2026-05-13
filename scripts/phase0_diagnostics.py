"""Phase 0 Diagnostic Experiments — read-only, no code fixes.

Runs experiments 0-A through 0-D to isolate the Sim-to-Real gap root cause.
Outputs a JSON report + histograms to results/phase0_diagnostics/.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facenet_pytorch import MTCNN
from federated_project.dataset import get_default_transforms
from federated_project.dp_utils import load_eval_pairs, EvalPair
from federated_project.federation import create_model, resolve_device

CHECKPOINT = PROJECT_ROOT / "custom_test" / "best_run.pt"
PAIRS_FILE = PROJECT_ROOT / "eval" / "pairs.csv"
EVAL_DIR = PROJECT_ROOT / "eval" / "lfw-deepfunneled" / "lfw-deepfunneled"
OUT_DIR = PROJECT_ROOT / "results" / "phase0_diagnostics"
DEVICE = resolve_device("cuda" if torch.cuda.is_available() else None)


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=DEVICE,
    )
    model.feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=True)
    saved_W = ckpt["W_matrix"].to(device=DEVICE, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model


@torch.no_grad()
def embed_resize(model, path, transform):
    img = Image.open(path).convert("RGB")
    return model(transform(img).unsqueeze(0).to(DEVICE)).squeeze(0).cpu()


@torch.no_grad()
def embed_mtcnn(model, path, mtcnn):
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None
    return model(face.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()


@torch.no_grad()
def embed_mtcnn_renorm(model, path, mtcnn_raw, transform_norm_only):
    """MTCNN crop geometry + training normalization (isolate divisor effect)."""
    img = Image.open(path).convert("RGB")
    face = mtcnn_raw(img)  # post_process=False → [0,255] tensor
    if face is None:
        return None
    # Convert to PIL, then apply training transforms (Resize+ToTensor+Normalize /127.5)
    face_pil = Image.fromarray(face.permute(1, 2, 0).byte().numpy())
    tensor = transform_norm_only(face_pil).unsqueeze(0).to(DEVICE)
    return model(tensor).squeeze(0).cpu()


def compute_tar_at_far_from_scores(genuine, impostor, far_target=0.001):
    """Returns (tar, threshold, full ROC table)."""
    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.unique(all_scores)
    thresholds.sort()
    roc = []
    for t in thresholds:
        far = float(np.mean(impostor >= t))
        tar = float(np.mean(genuine >= t))
        roc.append((float(t), tar, far))
    roc_arr = np.array(roc)
    far_arr, tar_arr = roc_arr[:, 2], roc_arr[:, 1]
    below = np.where(far_arr <= far_target)[0]
    above = np.where(far_arr >= far_target)[0]
    if below.size == 0:
        return 1.0, float(roc_arr[0, 0]), roc
    if above.size == 0:
        return 0.0, float(roc_arr[-1, 0]), roc
    lo, hi = below[-1], above[0]
    if lo == hi:
        return float(tar_arr[lo]), float(roc_arr[lo, 0]), roc
    far_lo, far_hi = float(far_arr[lo]), float(far_arr[hi])
    tar_lo, tar_hi = float(tar_arr[lo]), float(tar_arr[hi])
    t_lo, t_hi = float(roc_arr[lo, 0]), float(roc_arr[hi, 0])
    if far_hi == far_lo:
        return float(tar_hi), float(t_hi), roc
    alpha = (far_target - far_lo) / (far_hi - far_lo)
    tar_interp = tar_lo + (tar_hi - tar_lo) * alpha
    t_interp = t_lo + (t_hi - t_lo) * alpha
    return float(tar_interp), float(t_interp), roc


def score_pairs(pairs, embed_fn):
    """Score all pairs. Returns genuine_scores, impostor_scores, n_failed."""
    genuine, impostor, failed = [], [], 0
    for i, p in enumerate(pairs):
        if (i + 1) % 500 == 0:
            print(f"    scored {i+1}/{len(pairs)}")
        ea = embed_fn(p.image_a)
        eb = embed_fn(p.image_b)
        if ea is None or eb is None:
            failed += 1
            continue
        s = float(torch.dot(ea, eb).item())
        (genuine if p.is_same_person else impostor).append(s)
    return np.array(genuine), np.array(impostor), failed


def dist_stats(arr, label):
    return {
        "label": label, "n": len(arr),
        "mean": float(np.mean(arr)) if len(arr) else None,
        "std": float(np.std(arr)) if len(arr) else None,
        "min": float(np.min(arr)) if len(arr) else None,
        "max": float(np.max(arr)) if len(arr) else None,
        "median": float(np.median(arr)) if len(arr) else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_eval_pairs(str(PAIRS_FILE), str(EVAL_DIR))
    print(f"Loaded {len(pairs)} pairs ({sum(p.is_same_person for p in pairs)} genuine)")
    model = load_model()
    transform = get_default_transforms(train=False)
    report = {}

    # ── Experiment 0-A: --no-face-detect on LFW ──
    print("\n=== EXPERIMENT 0-A: Resize-only (benchmark path) ===")
    t0 = time.perf_counter()
    gen_a, imp_a, fail_a = score_pairs(pairs, lambda p: embed_resize(model, p, transform))
    tar_a, thresh_a, roc_a = compute_tar_at_far_from_scores(gen_a, imp_a)
    print(f"  TAR@0.1%FAR = {tar_a:.6f}  threshold = {thresh_a:.6f}  time = {time.perf_counter()-t0:.1f}s")
    report["exp_0a"] = {
        "tar_at_far_0001": tar_a, "threshold": thresh_a, "failed": fail_a,
        "genuine": dist_stats(gen_a, "genuine"), "impostor": dist_stats(imp_a, "impostor"),
    }

    # ── Experiment 0-C: Extract threshold + full ROC (from 0-A data) ──
    print("\n=== EXPERIMENT 0-C: ROC Curve & Threshold ===")
    roc_path = OUT_DIR / "roc_curve.csv"
    with roc_path.open("w") as f:
        f.write("threshold,tar,far\n")
        for t, tar, far in roc_a:
            f.write(f"{t:.8f},{tar:.8f},{far:.8f}\n")
    print(f"  Saved {len(roc_a)} ROC points to {roc_path.name}")
    report["exp_0c"] = {
        "threshold_at_far_0001": thresh_a,
        "tar_at_threshold": tar_a,
        "roc_points": len(roc_a),
    }

    # ── Experiment 0-B: MTCNN path on LFW ──
    print("\n=== EXPERIMENT 0-B: MTCNN path on LFW ===")
    mtcnn = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, device=DEVICE,
    )
    t0 = time.perf_counter()
    gen_b, imp_b, fail_b = score_pairs(pairs, lambda p: embed_mtcnn(model, p, mtcnn))
    tar_b, thresh_b, _ = compute_tar_at_far_from_scores(gen_b, imp_b)
    total_images = len(pairs) * 2
    detect_rate = 1.0 - fail_b / len(pairs) if len(pairs) else 0
    print(f"  TAR@0.1%FAR = {tar_b:.6f}  threshold = {thresh_b:.6f}")
    print(f"  Detection failures: {fail_b}/{len(pairs)} pairs  detect_rate={detect_rate:.2%}")
    print(f"  time = {time.perf_counter()-t0:.1f}s")
    report["exp_0b"] = {
        "tar_at_far_0001": tar_b, "threshold": thresh_b,
        "failed_pairs": fail_b, "detection_rate": detect_rate,
        "genuine": dist_stats(gen_b, "genuine"), "impostor": dist_stats(imp_b, "impostor"),
    }

    # ── Experiment 0-D: Cross-normalization on 50 genuine pairs ──
    print("\n=== EXPERIMENT 0-D: Cross-normalization isolation (50 genuine pairs) ===")
    genuine_pairs = [p for p in pairs if p.is_same_person][:50]
    mtcnn_raw = MTCNN(
        image_size=160, margin=20, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=False, device=DEVICE,
    )
    scores_a, scores_b, scores_c = [], [], []
    fail_d = 0
    for p in genuine_pairs:
        ea_r = embed_resize(model, p.image_a, transform)
        eb_r = embed_resize(model, p.image_b, transform)
        scores_a.append(float(torch.dot(ea_r, eb_r).item()))

        ea_m = embed_mtcnn(model, p.image_a, mtcnn)
        eb_m = embed_mtcnn(model, p.image_b, mtcnn)
        if ea_m is not None and eb_m is not None:
            scores_b.append(float(torch.dot(ea_m, eb_m).item()))
        else:
            fail_d += 1

        ea_c = embed_mtcnn_renorm(model, p.image_a, mtcnn_raw, transform)
        eb_c = embed_mtcnn_renorm(model, p.image_b, mtcnn_raw, transform)
        if ea_c is not None and eb_c is not None:
            scores_c.append(float(torch.dot(ea_c, eb_c).item()))

    print(f"  Path A (resize/127.5):      mean={np.mean(scores_a):.6f} std={np.std(scores_a):.6f}")
    if scores_b:
        print(f"  Path B (MTCNN/128.0):       mean={np.mean(scores_b):.6f} std={np.std(scores_b):.6f}")
    if scores_c:
        print(f"  Path C (MTCNN crop/127.5):  mean={np.mean(scores_c):.6f} std={np.std(scores_c):.6f}")
    print(f"  MTCNN detection failures: {fail_d}/50")
    report["exp_0d"] = {
        "path_a_resize_127p5": dist_stats(np.array(scores_a), "resize/127.5"),
        "path_b_mtcnn_128p0": dist_stats(np.array(scores_b), "mtcnn/128.0") if scores_b else None,
        "path_c_mtcnn_crop_127p5": dist_stats(np.array(scores_c), "mtcnn_crop/127.5") if scores_c else None,
        "mtcnn_failures": fail_d,
    }

    # ── Save full report ──
    report_path = OUT_DIR / "phase0_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"  Full report saved to: {report_path}")
    print(f"  ROC curve saved to:   {roc_path}")
    print(f"{'='*60}")

    # ── Print decision summary ──
    print("\n=== DECISION SUMMARY ===")
    print(f"  0-A  TAR (resize):  {report['exp_0a']['tar_at_far_0001']:.6f}")
    print(f"  0-B  TAR (MTCNN):   {report['exp_0b']['tar_at_far_0001']:.6f}")
    print(f"  0-C  Threshold t*:  {report['exp_0c']['threshold_at_far_0001']:.6f}")
    gap = abs(tar_a - tar_b)
    print(f"  MTCNN penalty:      {gap:.4f} ({gap*100:.2f}%)")
    if report["exp_0b"]["detection_rate"] < 0.90:
        print("  ⚠ MTCNN detection rate < 90% — detector misconfigured!")
    print()


if __name__ == "__main__":
    main()
