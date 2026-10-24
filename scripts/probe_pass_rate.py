#!/usr/bin/env python3
"""Measure the base model's certificate pass rate on 100 held-out problems.

If pass rate < 5%, SFT cold start is required before the two GRPO runs.
Without SGLang this writes a skip report (expected on a Mac).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.metrics import aggregate, score_completion
from certain.prompts import SYSTEM_PROMPT


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate(url: str, problem: str, model: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem if problem.startswith("Problem:") else f"Problem:\n{problem}"},
            ],
            "max_tokens": 256,
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    return payload["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "exports" / "heldout.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"))
    parser.add_argument("--model", default=os.environ.get("CERTAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "probe.json")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        args.out.write_text(json.dumps({"error": f"missing {args.data}", "pass_rate": None}))
        print(f"missing {args.data}; generate data first")
        return 1

    rows = load_jsonl(args.data)[: args.n]
    try:
        scores = []
        completions = []
        for row in rows:
            text = generate(args.url, row["problem"], args.model)
            completions.append({"id": row["id"], "completion": text})
            scores.append(score_completion(text, row["gold"], row["id"]))
        report = aggregate(scores).to_dict()
        report["pass_rate"] = report["faithful_accuracy"]
        report["need_sft"] = report["pass_rate"] < 0.05
        report["completions"] = completions
    except Exception as exc:
        report = {
            "error": str(exc),
            "pass_rate": None,
            "need_sft": True,
            "n": len(rows),
            "note": "SGLang unreachable; defaulting to SFT cold start.",
        }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "completions" and k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
