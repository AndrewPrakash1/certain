#!/usr/bin/env python3
"""SFT cold start on verified certificates. Both GRPO runs branch from here."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "exports" / "sft" / "train.jsonl"))
    parser.add_argument(
        "--model",
        default=os.environ.get("CERTAIN_TRAIN_MODEL", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"),
    )
    parser.add_argument("--backend", default=os.environ.get("CERTAIN_BACKEND", "unsloth"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "sft"))
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = sum(1 for line in Path(args.dataset).open() if line.strip()) if Path(args.dataset).exists() else 0
    config = {
        "type": "sft",
        "model_name": args.model,
        "dataset_name": args.dataset,
        "backend": args.backend,
        "output_dir": args.output,
        "num_epochs": args.epochs,
        "max_samples": args.max_samples,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "n_rows": n,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "config.json").write_text(json.dumps(config, indent=2))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    from aligntune.core.backend_factory import create_sft_trainer

    trainer = create_sft_trainer(
        model_name=args.model,
        dataset_name=args.dataset,
        backend=args.backend,
        num_epochs=args.epochs,
        batch_size=4,
        learning_rate=2e-4,
        max_samples=args.max_samples,
        output_dir=args.output,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
