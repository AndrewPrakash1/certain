#!/usr/bin/env python3
"""Matched Dr. GRPO runs: outcome vs certificate. One thing changes: the reward."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.rewards import register_aligntune_rewards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["certificate", "outcome", "format"], required=True)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "exports" / "grpo" / "train.jsonl"))
    parser.add_argument(
        "--model",
        default=os.environ.get("CERTAIN_TRAIN_MODEL", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"),
    )
    parser.add_argument("--sft-adapter", default="")
    parser.add_argument("--backend", default=os.environ.get("CERTAIN_BACKEND", "unsloth"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reward_name = {
        "certificate": "certain_certificate",
        "outcome": "certain_outcome",
        "format": "certain_format",
    }[args.reward]
    output = ROOT / "outputs" / f"grpo_{args.reward}"
    model = args.sft_adapter or args.model
    config = {
        "type": "rl",
        "algorithm": "grpo",
        "loss_type": "dr_grpo",
        "model_name": model,
        "dataset_name": args.dataset,
        "backend": args.backend,
        "output_dir": str(output),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.lr,
        "seed": args.seed,
        "reward_functions": [reward_name],
        "note": "Both runs must keep max_completion_length, steps, group size, and seed matched.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    registered = register_aligntune_rewards()
    config["registered"] = registered
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    from aligntune.core.backend_factory import create_rl_trainer

    trainer = create_rl_trainer(
        model_name=model,
        dataset_name=args.dataset,
        algorithm="grpo",
        backend=args.backend,
        loss_type="dr_grpo",
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        seed=args.seed,
        reward_functions=[reward_name],
        output_dir=str(output),
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
