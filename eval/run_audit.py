#!/usr/bin/env python3
"""Held-out eval through SGLang + AuditKIT, plus CERTAIN-only metrics.

AuditKIT exact-match cannot see a certificate. Faithful accuracy and
unfaithful-correct are computed here from the same predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certain.metrics import MetricReport, aggregate, score_completion
from certain.prompts import SYSTEM_PROMPT


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sglang_complete(url: str, model: str, problem: str, max_tokens: int = 256) -> tuple[str, float]:
    t0 = time.perf_counter()
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem if problem.startswith("Problem:") else f"Problem:\n{problem}"},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode())
    text = payload["choices"][0]["message"]["content"]
    return text, time.perf_counter() - t0


def auditkit_compare(rows: list[dict], predictions: dict[str, list[str]]) -> dict | None:
    try:
        import auditkit as ak
    except Exception as exc:
        return {"skipped": True, "reason": str(exc)}

    samples = [ak.Sample(input=r["problem"], target=str(r["gold"]), id=r.get("id")) for r in rows]
    models = {}
    for name, texts in predictions.items():
        captured = list(texts)

        def _fn(prompts, _captured=captured):
            return _captured

        models[name] = _fn
    try:
        result = ak.compare_models(samples, models=models)
        headline = getattr(result, "headline", None) or str(result)
        return {"headline": headline, "raw": str(result)[:4000]}
    except Exception as exc:
        # older AuditKIT: evaluate each model
        headlines = {}
        for name, texts in predictions.items():
            try:
                res = ak.evaluate(samples, model=lambda prompts, t=texts: t)
                headlines[name] = getattr(res, "headline", str(res))
            except Exception as inner:
                headlines[name] = str(inner)
        return {"headline": headlines, "error": str(exc)}


def constructed_baselines(rows: list[dict]) -> dict[str, list[str]]:
    """Oracle gold vs unfaithful-correct vs a weak base. Used when no adapters yet."""
    gold, outcome, base = [], [], []
    for i, row in enumerate(rows):
        gold.append(row.get("completion") or row.get("output"))
        outcome.append(f"I added the numbers in my head.\n\\boxed{{{row['gold']}}}")
        wrong = "0" if str(row["gold"]) != "0" else "1"
        if i % 5 == 0:
            base.append(f"\\boxed{{{row['gold']}}}")
        else:
            base.append(f"not sure\n\\boxed{{{wrong}}}")
    return {"oracle_gold": gold, "constructed_outcome": outcome, "constructed_base": base}


def score_model(rows: list[dict], texts: list[str]) -> MetricReport:
    return aggregate(score_completion(text, row["gold"], row.get("id", str(i))) for i, (row, text) in enumerate(zip(rows, texts)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "exports" / "heldout.jsonl")
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", ""))
    parser.add_argument("--models", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_results.json")
    parser.add_argument("--constructed", action="store_true", help="score oracle/unfaithful/base traces")
    args = parser.parse_args()

    rows = load_jsonl(args.data)[: args.limit]
    predictions: dict[str, list[str]] = {}
    latencies: dict[str, list[float]] = {}

    if args.constructed or not args.url:
        predictions.update(constructed_baselines(rows))

    if args.url and args.models:
        for spec in args.models.split(","):
            name = spec.strip()
            if not name:
                continue
            texts, times = [], []
            for row in rows:
                text, dt = sglang_complete(args.url, name, row["problem"])
                texts.append(text)
                times.append(dt)
            predictions[name] = texts
            latencies[name] = times

    metrics = {name: score_model(rows, texts).to_dict() for name, texts in predictions.items()}
    # drop per-sample blobs from the nested copy used for AuditKIT? keep them; they are the scoreboard
    audit = auditkit_compare(rows, predictions) if predictions else None
    payload = {
        "n": len(rows),
        "metrics": {
            name: {k: v for k, v in report.items() if k != "samples"} | {"n": report["n"]}
            for name, report in metrics.items()
        },
        "details": metrics,
        "latency_s": {k: (sum(v) / len(v) if v else None) for k, v in latencies.items()},
        "throughput_qps": {k: (len(v) / sum(v) if v and sum(v) else None) for k, v in latencies.items()},
        "auditkit": audit,
        "sglang_url": args.url or None,
        "mode": "sglang" if args.url and args.models else "constructed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    # compact copy for the demo scoreboard
    (ROOT / "demo" / "scoreboard.json").write_text(
        json.dumps({"n": payload["n"], "metrics": payload["metrics"], "mode": payload["mode"]}, indent=2)
    )
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
