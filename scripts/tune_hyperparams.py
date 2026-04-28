"""Find the best simulation hyperparameters using Bayesian optimisation (Optuna).

Run forever (or until you Ctrl-C) and it will keep improving.
Every trial is saved to SQLite immediately — crash-safe, fully resumable.

Usage
-----
# Start a new search (runs until you Ctrl-C):
    python tune_hyperparams.py --data-dir /path/to/data

# Resume exactly where you left off (same study name, same db):
    python tune_hyperparams.py --data-dir /path/to/data --study-name my_study

# Run for a fixed number of trials then stop:
    python tune_hyperparams.py --data-dir /path/to/data --max-trials 200

Install dependency (once):
    pip install optuna
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import optuna
except ImportError:
    sys.exit("Optuna is not installed. Run:  pip install optuna")

from federated_project.simulation import run_simulation

optuna.logging.set_verbosity(optuna.logging.WARNING)  # we print our own logs


# ---------------------------------------------------------------------------
# Hyperparameter search space
# (edit bounds/choices to match what makes sense for your problem)
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        # Federated rounds
        "num_rounds": trial.suggest_int("num_rounds", 3, 20),

        # Fraction of clients selected per round
        "fraction_fit": trial.suggest_float("fraction_fit", 0.5, 1.0),

        # Mini-batch size
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),

        # Local training epochs per round
        "local_epochs": trial.suggest_int("local_epochs", 1, 5),

        # Learning rate — log scale is essential for LR
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),

        # Triplet margin
        "margin": trial.suggest_float("margin", 0.1, 1.5),

        # Pretrained backbone
        "pretrained": trial.suggest_categorical("pretrained", ["vggface2", "casia-webface"]),

        # Spreadout regularisation strength (0 = disabled)
        "spreadout_strength": trial.suggest_float("spreadout_strength", 0.0, 2.0),

        # Spreadout margin
        "spreadout_margin": trial.suggest_float("spreadout_margin", 0.1, 1.0),

        # How many spreadout steps per round
        "spreadout_steps": trial.suggest_int("spreadout_steps", 1, 10),

        # Spreadout learning rate
        "spreadout_lr": trial.suggest_float("spreadout_lr", 1e-4, 1.0, log=True),
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _make_objective(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        config = _suggest(trial)

        print(
            f"\n[Trial {trial.number}] "
            + ", ".join(f"{k}={v}" for k, v in config.items())
        )

        round_results = run_simulation(
            data_dir=args.data_dir,
            num_rounds=config["num_rounds"],
            fraction_fit=config["fraction_fit"],
            batch_size=config["batch_size"],
            local_epochs=config["local_epochs"],
            lr=config["lr"],
            margin=config["margin"],
            pretrained=config["pretrained"],
            spreadout_strength=config["spreadout_strength"],
            spreadout_margin=config["spreadout_margin"],
            spreadout_steps=config["spreadout_steps"],
            spreadout_lr=config["spreadout_lr"],
            seed=args.seed,
            device=args.device,
        )

        last = round_results[-1]
        train_loss = float(last.train_loss)
        spreadout_loss = float(last.spreadout_loss)

        if args.score_metric == "final_train_plus_spreadout":
            score = train_loss + spreadout_loss
        else:
            score = train_loss

        completed = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        prev_best = min((t.value for t in completed if t.number != trial.number), default=None)
        is_best = prev_best is None or score < prev_best

        print(
            f"    train_loss={train_loss:.6f}  "
            f"spreadout_loss={spreadout_loss:.6f}  "
            f"score={score:.6f}"
            + ("  ← NEW BEST!" if is_best else "")
        )

        # Store extra metrics so they appear in the CSV export
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("spreadout_loss", spreadout_loss)

        return score

    return objective


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search using Optuna."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-metric",
        choices=["final_train_loss", "final_train_plus_spreadout"],
        default="final_train_loss",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Stop after this many trials. Omit to run forever.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help=(
            "Name of the Optuna study. Reuse the same name to RESUME a previous run. "
            "Defaults to a timestamped name on first run."
        ),
    )
    parser.add_argument("--output-dir", default="results/tuning")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"sim_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_path}"

    print(f"Study name : {study_name}")
    print(f"Database   : {db_path}")
    print(f"Metric     : {args.score_metric}  (lower is better)")
    print(f"Max trials : {args.max_trials or 'unlimited — Ctrl-C to stop'}")
    print()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,          # resume automatically if db already exists
        sampler=optuna.samplers.TPESampler(seed=args.seed),  # Bayesian (Tree Parzen Estimator)
        pruner=optuna.pruners.NopPruner(),
    )

    already_done = len(study.trials)
    if already_done:
        print(f"Resuming study — {already_done} trial(s) already completed.")

    try:
        study.optimize(
            _make_objective(args),
            n_trials=args.max_trials,   # None = run forever
            catch=(Exception,),         # log failures, don't crash the whole search
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving results …")

    # ── Save best config ──────────────────────────────────────────────────────
    best = study.best_trial
    best_config = {
        "study_name": study_name,
        "score_metric": args.score_metric,
        "score": best.value,
        "params": best.params,
        "train_loss": best.user_attrs.get("train_loss"),
        "spreadout_loss": best.user_attrs.get("spreadout_loss"),
        "trial_number": best.number,
        "total_trials_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
    }

    best_path = output_dir / f"{study_name}_best.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    # ── Save all trials as CSV ────────────────────────────────────────────────
    csv_path = output_dir / f"{study_name}_all_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best score : {best.value:.6f}")
    print(f"Best params:")
    print(json.dumps(best.params, indent=2))
    print(f"\nBest config saved to : {best_path}")
    print(f"All trials saved to  : {csv_path}")
    print(f"Full Optuna database : {db_path}")
    print()
    print("To resume this search later, run with:")
    print(f"  --study-name {study_name}")


if __name__ == "__main__":
    main()