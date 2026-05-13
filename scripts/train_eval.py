"""Integrated FedFace training + LFW evaluation script.

Two modes:
  1. Single run  — train once, evaluate, print result.
  2. Sweep mode  -- iterate over a grid of hyperparameters, rank every
                    configuration by TAR @ 0.1 % FAR, write a CSV summary.

The sweep is driven by TAR@FAR as the objective, not training loss.
Collapse detection is built in: if gen_gap < 0.10 after training the run
is marked COLLAPSED and skipped without wasting time on a full LFW eval.

Quick-start examples
--------------------
# Pretrained baseline (no training — sets your ceiling):
python scripts/train_eval.py --mode baseline \\
    --pairs-file eval/pairs.csv \\
    --eval-image-dir eval/lfw-deepfunneled/lfw-deepfunneled

# Single run with Path-A recommended parameters:
python scripts/train_eval.py --mode single \\
    --data-dir "celebs/Celebrity Faces Dataset" \\
    --pairs-file eval/pairs.csv \\
    --eval-image-dir eval/lfw-deepfunneled/lfw-deepfunneled \\
    --freeze-backbone \\
    --lr 1e-4 --num-rounds 30 --margin 0.9 \\
    --checkpoint-path results/path_a/checkpoint.pt \\
    --report-path results/path_a/report.json

# Hyperparameter sweep (TAR@FAR-ranked):
python scripts/train_eval.py --mode sweep \\
    --data-dir "celebs/Celebrity Faces Dataset" \\
    --pairs-file eval/pairs.csv \\
    --eval-image-dir eval/lfw-deepfunneled/lfw-deepfunneled \\
    --sweep-output results/sweep/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — works regardless of where the script lives in the repo
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from federated_project.dataset import get_default_transforms
from federated_project.dp_utils import compute_tar_at_far, load_eval_pairs
from federated_project.federation import create_model, resolve_device
from federated_project.simulation import run_simulation

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """All parameters that define one training + eval run."""
    # Training
    data_dir: str = ""
    num_rounds: int = 30
    fraction_fit: float = 1.0
    batch_size: int = 16
    local_epochs: int = 1
    lr: float = 1e-4
    margin: float = 0.9
    pretrained: str = "vggface2"
    spreadout_strength: float = 10.0
    spreadout_margin: float = 0.35
    spreadout_steps: int = 1
    spreadout_lr: float = 1.0
    freeze_backbone: bool = True
    seed: int = 42
    # Evaluation
    pairs_file: str = ""
    eval_image_dir: str = ""
    far_target: float = 0.001
    # Output
    checkpoint_path: str = ""
    report_path: str = ""
    device: str | None = None
    # Collapse detection threshold — runs with gen_gap below this
    # are marked collapsed and skipped at full eval.
    collapse_gap_threshold: float = 0.10


@dataclass
class DiagnosticScores:
    """Fast collapse check: score distributions on a small sample."""
    gen_mean: float = 0.0
    gen_std: float = 0.0
    imp_mean: float = 0.0
    imp_std: float = 0.0
    gap: float = 0.0
    n_genuine: int = 0
    n_impostor: int = 0
    collapsed: bool = False


@dataclass
class RunResult:
    config: RunConfig
    tar_at_far: float = 0.0
    far_target: float = 0.001
    diagnostic: DiagnosticScores = field(default_factory=DiagnosticScores)
    train_elapsed_sec: float = 0.0
    eval_elapsed_sec: float = 0.0
    final_train_loss: float = 0.0
    final_spreadout_loss: float = 0.0
    status: str = "ok"          # "ok" | "collapsed" | "baseline" | "error"
    error_msg: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str | None) -> torch.device:
    return resolve_device(device_str)


def _load_pretrained_model(pretrained: str, device: torch.device) -> torch.nn.Module:
    """Load a vanilla pretrained backbone for baseline evaluation.

    Uses num_clients=17 (a safe non-degenerate value) so the FedFaceModel
    W_matrix is never shape (1, 512), which can cause forward-pass issues in
    some model implementations. The W_matrix is never used in evaluation —
    only feature_extractor is called directly in _quick_diagnostic and
    compute_tar_at_far — so num_clients here is irrelevant to scores.
    """
    model = create_model(num_clients=17, pretrained=pretrained, device=device)
    model.eval()
    return model


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        num_clients=int(ckpt["num_clients"]),
        pretrained=str(ckpt["pretrained"]),
        device=device,
    )
    model.feature_extractor.load_state_dict(
        ckpt["feature_extractor_state_dict"], strict=True
    )
    saved_W = ckpt["W_matrix"].to(device=device, dtype=model.W_matrix.dtype)
    model.W_matrix.data.copy_(F.normalize(saved_W, p=2, dim=1))
    model.eval()
    return model, ckpt


def _quick_diagnostic(
    model: torch.nn.Module,
    pairs,
    device: torch.device,
    transform,
    n_sample: int = 300,
) -> DiagnosticScores:
    """
    Compute score distributions on a small sample of pairs.
    Cheap collapse detector — runs in seconds even on CPU.
    """
    import random
    genuine = [p for p in pairs if p.is_same_person]
    impostor = [p for p in pairs if not p.is_same_person]

    rng = random.Random(0)
    gen_sample = rng.sample(genuine, min(n_sample, len(genuine)))
    imp_sample = rng.sample(impostor, min(n_sample, len(impostor)))

    def _embed(img_path: str):
        """Return L2-normalised embedding via feature_extractor (not model()).

        Calling model.feature_extractor directly bypasses any W_matrix
        projection in FedFaceModel.forward that would produce classification
        logits instead of embeddings, causing zero/garbage similarity scores.
        """
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            feat = model.feature_extractor(t).squeeze(0)
            return F.normalize(feat, p=2, dim=0)
        except Exception:
            return None

    def _pair_paths(p) -> tuple[str, str]:
        """Extract (path_a, path_b) from an EvalPair regardless of field names.

        EvalPair is defined in dp_utils.py which varies across codebases.
        We try the known variants in order:
          1. p.path_a / p.path_b       (original assumption)
          2. p.img_a / p.img_b         (common alternative)
          3. p.image_a / p.image_b
          4. p[0] / p[1]               (namedtuple / tuple fallback)
        Raises AttributeError with a diagnostic message if none match.
        """
        for a_attr, b_attr in [("path_a", "path_b"), ("img_a", "img_b"),
                                ("image_a", "image_b"), ("path1", "path2"),
                                ("file_a", "file_b")]:
            if hasattr(p, a_attr) and hasattr(p, b_attr):
                return str(getattr(p, a_attr)), str(getattr(p, b_attr))
        # Last resort: indexing (namedtuple or plain tuple/list)
        try:
            return str(p[0]), str(p[1])
        except Exception:
            pass
        raise AttributeError(
            f"Cannot find image path fields on EvalPair. "
            f"Available attributes: {[a for a in dir(p) if not a.startswith('_')]}"
        )

    def score_pairs(pair_list):
        scores = []
        model.eval()
        with torch.no_grad():
            for p in pair_list:
                try:
                    pa, pb = _pair_paths(p)
                    ea = _embed(pa)
                    eb = _embed(pb)
                    if ea is not None and eb is not None:
                        scores.append(float(torch.dot(ea, eb).item()))
                except AttributeError as exc:
                    # Surface the field-name error once then abort gracefully
                    print(f"  [diagnostic] EvalPair field error: {exc}")
                    return scores
        return scores

    gen_scores = score_pairs(gen_sample)
    imp_scores = score_pairs(imp_sample)

    if not gen_scores or not imp_scores:
        return DiagnosticScores(collapsed=True)

    import statistics
    gm = statistics.mean(gen_scores)
    gs = statistics.stdev(gen_scores) if len(gen_scores) > 1 else 0.0
    im = statistics.mean(imp_scores)
    is_ = statistics.stdev(imp_scores) if len(imp_scores) > 1 else 0.0
    gap = gm - im

    return DiagnosticScores(
        gen_mean=round(gm, 4),
        gen_std=round(gs, 4),
        imp_mean=round(im, 4),
        imp_std=round(is_, 4),
        gap=round(gap, 4),
        n_genuine=len(gen_scores),
        n_impostor=len(imp_scores),
        collapsed=gap < 0.10,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(cfg: RunConfig, pairs, transform, device: torch.device) -> RunResult:
    """Train (if data_dir given) then evaluate. Returns a RunResult."""
    result = RunResult(
        config=cfg,
        far_target=cfg.far_target,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    # ── 1. Training ────────────────────────────────────────────────────────
    checkpoint_path: Path | None = None
    if cfg.data_dir:
        cp = cfg.checkpoint_path or str(
            Path("results") / f"run_{result.timestamp.replace(':', '-')}" / "checkpoint.pt"
        )
        checkpoint_path = Path(cp)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Training: lr={cfg.lr}  rounds={cfg.num_rounds}  "
              f"margin={cfg.margin}  freeze={cfg.freeze_backbone}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            sim_results = run_simulation(
                data_dir=cfg.data_dir,
                num_rounds=cfg.num_rounds,
                fraction_fit=cfg.fraction_fit,
                batch_size=cfg.batch_size,
                local_epochs=cfg.local_epochs,
                lr=cfg.lr,
                margin=cfg.margin,
                pretrained=cfg.pretrained,
                spreadout_strength=cfg.spreadout_strength,
                spreadout_margin=cfg.spreadout_margin,
                spreadout_steps=cfg.spreadout_steps,
                spreadout_lr=cfg.spreadout_lr,
                seed=cfg.seed,
                device=cfg.device,
                checkpoint_path=str(checkpoint_path),
                freeze_backbone=cfg.freeze_backbone,
            )
            result.train_elapsed_sec = round(time.perf_counter() - t0, 1)
            if sim_results:
                last = sim_results[-1]
                result.final_train_loss = round(float(last.train_loss), 6)
                result.final_spreadout_loss = round(float(last.spreadout_loss), 6)
                # Print every round for single runs, every 10th for sweeps
                for r in sim_results:
                    print(
                        f"  Round {r.round_idx:>3}: "
                        f"train_loss={r.train_loss:.6f}  "
                        f"spreadout_loss={r.spreadout_loss:.6f}"
                    )
        except Exception as exc:
            result.status = "error"
            result.error_msg = str(exc)
            print(f"  ERROR during training: {exc}")
            return result
    else:
        # Baseline mode — load pretrained only
        checkpoint_path = None

    # ── 2. Load model ──────────────────────────────────────────────────────
    if checkpoint_path and checkpoint_path.exists():
        model, _ = _load_checkpoint(checkpoint_path, device)
    else:
        print("\n  No checkpoint found — running pretrained baseline.")
        model = _load_pretrained_model(cfg.pretrained, device)
        result.status = "baseline"

    # ── 3. Quick collapse diagnostic ───────────────────────────────────────
    print("\n  Running collapse diagnostic (300-pair sample)...")
    diag = _quick_diagnostic(model, pairs, device, transform, n_sample=300)
    result.diagnostic = diag

    print(f"  Genuine  — mean={diag.gen_mean:.4f}  std={diag.gen_std:.4f}")
    print(f"  Impostor — mean={diag.imp_mean:.4f}  std={diag.imp_std:.4f}")
    print(f"  Gap      : {diag.gap:.4f}  {'⚠ COLLAPSED' if diag.collapsed else '✓ OK'}")

    if diag.collapsed and result.status == "ok":
        result.status = "collapsed"
        result.tar_at_far = 0.0
        print("  Skipping full LFW eval (collapsed model).")
        return result

    # ── 4. Full LFW evaluation ─────────────────────────────────────────────
    print(f"\n  Computing TAR @ FAR={cfg.far_target} on {len(pairs)} pairs...")
    t0 = time.perf_counter()
    try:
        tar = compute_tar_at_far(
            model=model,
            pairs=pairs,
            far_target=cfg.far_target,
            device=device,
            transform=transform,
        )
        result.tar_at_far = round(float(tar), 6)
        result.eval_elapsed_sec = round(time.perf_counter() - t0, 1)
    except Exception as exc:
        result.status = "error"
        result.error_msg = f"eval: {exc}"
        print(f"  ERROR during eval: {exc}")
        return result

    print(f"\n{'='*60}")
    print(f"  TAR @ {cfg.far_target*100:.3g}% FAR : "
          f"{result.tar_at_far:.6f}  ({result.tar_at_far*100:.2f}%)")
    print(f"  Gap                  : {diag.gap:.4f}")
    print(f"  Train time           : {result.train_elapsed_sec:.0f}s")
    print(f"  Eval time            : {result.eval_elapsed_sec:.0f}s")
    print(f"{'='*60}")

    # ── 5. Save report ─────────────────────────────────────────────────────
    if cfg.report_path:
        rp = Path(cfg.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(
            json.dumps(_result_to_dict(result), indent=2), encoding="utf-8"
        )
        print(f"\n  Report saved → {rp}")

    return result


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

# Default sweep grid — tuned for the small-N (17-client) regime.
# Organized around Path A: frozen/partial backbone, low lr, few rounds.
# The agent or user can override this by editing the lists below,
# or by passing --sweep-config pointing at a JSON file.
DEFAULT_SWEEP_GRID: dict[str, list[Any]] = {
    "freeze_backbone": [True, False],
    "lr": [5e-5, 1e-4, 5e-4],
    "num_rounds": [20, 40],
    "margin": [0.9],           # keep fixed; rarely the problem
    "spreadout_strength": [10.0],
    "local_epochs": [1],
}


def run_sweep(
    base_cfg: RunConfig,
    pairs,
    transform,
    device: torch.device,
    sweep_grid: dict[str, list[Any]],
    output_dir: Path,
    stop_on_first_good: bool = False,
    good_tar_threshold: float = 0.30,
) -> list[RunResult]:
    """
    Grid search over sweep_grid parameters.
    Evaluates every combination (collapse-skipping fast path included).
    Writes a ranked CSV + full JSON after every completed run so that
    partial results survive crashes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"sweep_{timestamp}.csv"
    json_path = output_dir / f"sweep_{timestamp}.json"

    # Build the full Cartesian product
    keys = list(sweep_grid.keys())
    combos = list(product(*[sweep_grid[k] for k in keys]))
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"  Sweep: {total} configurations")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    results: list[RunResult] = []

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        print(f"\n[{idx}/{total}] Config: {params}")

        # Clone base config and apply sweep params
        cfg_dict = asdict(base_cfg)
        cfg_dict.update(params)
        # Auto-generate checkpoint path per run
        run_tag = "_".join(
            f"{k[:3]}{v}" for k, v in params.items()
        ).replace(" ", "").replace(".", "p")
        cfg_dict["checkpoint_path"] = str(
            output_dir / f"ckpt_{run_tag}.pt"
        )
        cfg_dict["report_path"] = ""   # handled by sweep writer
        cfg = RunConfig(**cfg_dict)

        result = run_single(cfg, pairs, transform, device)
        results.append(result)

        # Write intermediate results after every run
        _write_sweep_csv(results, csv_path)
        _write_sweep_json(results, json_path, sweep_grid)

        if (
            stop_on_first_good
            and result.status == "ok"
            and result.tar_at_far >= good_tar_threshold
        ):
            print(f"\n  ✓ Stopping early — TAR={result.tar_at_far:.4f} ≥ {good_tar_threshold}")
            break

    # Final ranked summary
    print("\n" + "="*60)
    print("  SWEEP COMPLETE — ranked by TAR@FAR")
    print("="*60)
    ranked = sorted(results, key=lambda r: r.tar_at_far, reverse=True)
    print(f"  {'rank':<5} {'tar_at_far':<12} {'gap':<8} {'status':<12} config")
    for rank, r in enumerate(ranked, 1):
        cfg_summary = (
            f"lr={r.config.lr}  rounds={r.config.num_rounds}  "
            f"freeze={r.config.freeze_backbone}  margin={r.config.margin}"
        )
        print(
            f"  {rank:<5} {r.tar_at_far:<12.6f} {r.diagnostic.gap:<8.4f} "
            f"{r.status:<12} {cfg_summary}"
        )
    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    return ranked


def _write_sweep_csv(results: list[RunResult], path: Path) -> None:
    ranked = sorted(results, key=lambda r: r.tar_at_far, reverse=True)
    fieldnames = [
        "rank", "tar_at_far", "gen_gap", "gen_mean", "imp_mean",
        "status", "lr", "num_rounds", "freeze_backbone", "margin",
        "spreadout_strength", "local_epochs",
        "final_train_loss", "train_elapsed_sec", "eval_elapsed_sec",
        "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(ranked, 1):
            writer.writerow({
                "rank": rank,
                "tar_at_far": r.tar_at_far,
                "gen_gap": r.diagnostic.gap,
                "gen_mean": r.diagnostic.gen_mean,
                "imp_mean": r.diagnostic.imp_mean,
                "status": r.status,
                "lr": r.config.lr,
                "num_rounds": r.config.num_rounds,
                "freeze_backbone": r.config.freeze_backbone,
                "margin": r.config.margin,
                "spreadout_strength": r.config.spreadout_strength,
                "local_epochs": r.config.local_epochs,
                "final_train_loss": r.final_train_loss,
                "train_elapsed_sec": r.train_elapsed_sec,
                "eval_elapsed_sec": r.eval_elapsed_sec,
                "timestamp": r.timestamp,
            })


def _write_sweep_json(
    results: list[RunResult], path: Path, sweep_grid: dict
) -> None:
    ranked = sorted(results, key=lambda r: r.tar_at_far, reverse=True)
    payload = {
        "sweep_grid": {k: v for k, v in sweep_grid.items()},
        "n_completed": len(results),
        "best_tar_at_far": ranked[0].tar_at_far if ranked else 0.0,
        "results": [_result_to_dict(r) for r in ranked],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _result_to_dict(r: RunResult) -> dict:
    return {
        "tar_at_far": r.tar_at_far,
        "far_target": r.far_target,
        "status": r.status,
        "error_msg": r.error_msg,
        "timestamp": r.timestamp,
        "diagnostic": asdict(r.diagnostic),
        "train": {
            "final_train_loss": r.final_train_loss,
            "final_spreadout_loss": r.final_spreadout_loss,
            "elapsed_sec": r.train_elapsed_sec,
        },
        "eval": {
            "elapsed_sec": r.eval_elapsed_sec,
        },
        "config": {
            k: v for k, v in asdict(r.config).items()
            if k not in {"report_path", "checkpoint_path"}
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "FedFace integrated train + eval.\n"
            "Modes: baseline | single | sweep\n\n"
            "  baseline — evaluate the raw pretrained model (no training).\n"
            "             Sets the performance ceiling before any FL.\n"
            "  single   — one training run followed by LFW evaluation.\n"
            "  sweep    — grid search over hyperparameters, ranked by TAR@FAR.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode", choices=["baseline", "single", "sweep"], required=True,
        help="baseline=no training, single=one run, sweep=grid search",
    )

    # Shared eval args
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--eval-image-dir", required=True)
    p.add_argument("--far-target", type=float, default=0.001)
    p.add_argument("--device", default=None)

    # Training args (used in single + sweep)
    p.add_argument("--data-dir", default="")
    p.add_argument("--pretrained", default="vggface2")
    p.add_argument("--num-rounds", type=int, default=30)
    p.add_argument("--fraction-fit", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.9)
    p.add_argument("--spreadout-strength", type=float, default=10.0)
    p.add_argument("--spreadout-margin", type=float, default=0.35)
    p.add_argument("--spreadout-steps", type=int, default=1)
    p.add_argument("--spreadout-lr", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze-backbone", action="store_true",
        help=(
            "Freeze early backbone layers during training. "
            "Strongly recommended for small datasets (< 100 clients). "
            "Prevents the 17-client collapse observed with full unfreezing."
        ),
    )

    # Single mode outputs
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--report-path", default="")

    # Sweep mode
    p.add_argument(
        "--sweep-output", default="results/sweep/",
        help="Directory to write sweep CSV + JSON results.",
    )
    p.add_argument(
        "--sweep-config", default=None,
        help=(
            "Optional path to a JSON file overriding the default sweep grid. "
            "Format: {\"lr\": [1e-4, 5e-4], \"num_rounds\": [20, 40], ...}"
        ),
    )
    p.add_argument(
        "--stop-on-first-good", action="store_true",
        help="In sweep mode, stop after the first run that achieves TAR ≥ --good-tar.",
    )
    p.add_argument("--good-tar", type=float, default=0.30)
    p.add_argument(
        "--collapse-gap-threshold", type=float, default=0.10,
        help=(
            "Minimum gen-imp gap to consider a model non-collapsed. "
            "Runs below this skip full LFW eval."
        ),
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Validate paths ─────────────────────────────────────────────────────
    for path_str, label in [
        (args.pairs_file, "pairs-file"),
        (args.eval_image_dir, "eval-image-dir"),
    ]:
        if not Path(path_str).exists():
            print(f"ERROR: {label} not found: {path_str}")
            sys.exit(1)

    if args.mode in ("single", "sweep") and not args.data_dir:
        if args.mode == "single":
            print("ERROR: --data-dir is required for single mode.")
            sys.exit(1)

    device = resolve_device(args.device)
    transform = get_default_transforms(train=False)

    # Load pairs once — shared across all runs
    print(f"Loading pairs from: {args.pairs_file}")
    pairs = load_eval_pairs(args.pairs_file, args.eval_image_dir)
    if not pairs:
        print("ERROR: pairs file produced zero pairs.")
        sys.exit(1)
    n_gen = sum(1 for p in pairs if p.is_same_person)
    n_imp = len(pairs) - n_gen
    print(f"Loaded {len(pairs)} pairs ({n_gen} genuine, {n_imp} impostor)")

    # ── Base config ────────────────────────────────────────────────────────
    base_cfg = RunConfig(
        data_dir=args.data_dir,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        margin=args.margin,
        pretrained=args.pretrained,
        spreadout_strength=args.spreadout_strength,
        spreadout_margin=args.spreadout_margin,
        spreadout_steps=args.spreadout_steps,
        spreadout_lr=args.spreadout_lr,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
        pairs_file=args.pairs_file,
        eval_image_dir=args.eval_image_dir,
        far_target=args.far_target,
        checkpoint_path=args.checkpoint_path,
        report_path=args.report_path,
        device=args.device,
        collapse_gap_threshold=args.collapse_gap_threshold,
    )

    # ── Dispatch ───────────────────────────────────────────────────────────
    if args.mode == "baseline":
        print("\n" + "="*60)
        print("  MODE: Pretrained baseline (no federated training)")
        print("="*60)
        cfg = RunConfig(
            data_dir="",   # signals: skip training
            pretrained=args.pretrained,
            pairs_file=args.pairs_file,
            eval_image_dir=args.eval_image_dir,
            far_target=args.far_target,
            device=args.device,
            report_path=args.report_path,
            collapse_gap_threshold=args.collapse_gap_threshold,
        )
        run_single(cfg, pairs, transform, device)

    elif args.mode == "single":
        run_single(base_cfg, pairs, transform, device)

    elif args.mode == "sweep":
        # Load custom grid if provided
        sweep_grid = DEFAULT_SWEEP_GRID.copy()
        if args.sweep_config:
            sc = Path(args.sweep_config)
            if not sc.exists():
                print(f"ERROR: --sweep-config not found: {sc}")
                sys.exit(1)
            custom = json.loads(sc.read_text(encoding="utf-8"))
            sweep_grid.update(custom)
            print(f"Loaded custom sweep grid from {sc}")

        # Sweep requires data-dir
        if not args.data_dir:
            print("ERROR: --data-dir is required for sweep mode.")
            sys.exit(1)

        run_sweep(
            base_cfg=base_cfg,
            pairs=pairs,
            transform=transform,
            device=device,
            sweep_grid=sweep_grid,
            output_dir=Path(args.sweep_output),
            stop_on_first_good=args.stop_on_first_good,
            good_tar_threshold=args.good_tar,
        )


if __name__ == "__main__":
    main()