#!/usr/bin/env python3
"""Build the CERTAIN dataset: finance generator + math + optional GSM8K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.curate import export_datasets, gate, maybe_run_curatorkit
from certain.finance_gen import generate_finance
from certain.gsm8k import ingest_gsm8k
from certain.math_gen import generate_math


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--heldout", type=int, default=200)
    parser.add_argument("--math-train", type=int, default=400)
    parser.add_argument("--math-heldout", type=int, default=80)
    parser.add_argument("--gsm8k-limit", type=int, default=300)
    parser.add_argument("--no-gsm8k", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    problems = []
    problems += generate_finance(args.train, seed=7, split="train")
    problems += generate_finance(args.heldout, seed=10007, split="heldout", start=10_000)
    problems += generate_math(args.math_train, seed=11, split="train", start=20_000)
    problems += generate_math(args.math_heldout, seed=11011, split="heldout", start=30_000)

    gsm_rejected = []
    if not args.no_gsm8k:
        kept_g, rejected_g = ingest_gsm8k(
            args.out / "cache",
            split="train",
            limit=args.gsm8k_limit,
            download=True,
        )
        for p in kept_g:
            p.split = "train"
        problems += kept_g
        gsm_rejected = rejected_g
        print(f"gsm8k kept={len(kept_g)} rejected={len(rejected_g)}")

    kept, rejected = gate(problems)
    rejected.extend(gsm_rejected)
    manifest = export_datasets(kept, args.out, rejected)
    maybe_run_curatorkit(args.out / "raw" / "all.jsonl", args.out)
    print(json.dumps({k: manifest[k] for k in ("n_kept", "n_train", "n_heldout", "n_rejected", "kinds")}, indent=2))
    print(f"manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
