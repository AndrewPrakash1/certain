"""CuratorKIT-shaped gate: hygiene, executable check, trainer-ready export."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from certain.dsl import interpret_text
from certain.finance_gen import Problem
from certain.prompts import SYSTEM_PROMPT

PII_PATTERNS = ("ssn", "social security", "passport", "credit card")


def hygiene_ok(problem: Problem) -> tuple[bool, str | None]:
    blob = (problem.problem + " " + problem.certificate).lower()
    for pat in PII_PATTERNS:
        if pat in blob:
            return False, f"pii:{pat}"
    if any(ch in problem.problem for ch in ("\x00",)):
        return False, "binary"
    return True, None


def executable_ok(problem: Problem) -> tuple[bool, str | None]:
    result = interpret_text(problem.completion())
    if not result.computes:
        return False, result.parse_error or "does_not_compute"
    from certain.dsl import parse_number, values_close

    gold = parse_number(problem.gold)
    if gold is None or result.value is None or not values_close(result.value, gold):
        return False, "gold_mismatch"
    return True, None


def gate(problems: Iterable[Problem]) -> tuple[list[Problem], list[dict[str, Any]]]:
    kept: list[Problem] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for problem in problems:
        digest = hashlib.sha256(problem.problem.encode()).hexdigest()
        if digest in seen:
            rejected.append({"id": problem.id, "reason": "duplicate"})
            continue
        ok, reason = hygiene_ok(problem)
        if not ok:
            rejected.append({"id": problem.id, "reason": reason})
            continue
        ok, reason = executable_ok(problem)
        if not ok:
            rejected.append({"id": problem.id, "reason": reason})
            continue
        seen.add(digest)
        kept.append(problem)
    return kept, rejected


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_datasets(kept: list[Problem], out_dir: Path, rejected: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [p for p in kept if p.split == "train"]
    held = [p for p in kept if p.split != "train"]

    sft_rows = [
        {
            "instruction": p.prompt(),
            "input": "",
            "output": p.completion(),
            "system": SYSTEM_PROMPT,
            "id": p.id,
            "domain": p.domain,
            "kind": p.kind,
        }
        for p in train
    ]
    grpo_rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p.prompt()},
            ],
            "answer": p.gold,
            "ground_truth": p.gold,
            "gold": p.gold,
            "id": p.id,
            "domain": p.domain,
            "kind": p.kind,
        }
        for p in train
    ]
    held_rows = [p.to_record() for p in held]
    raw_rows = [p.to_record() for p in kept]

    files = {
        "raw/all.jsonl": raw_rows,
        "exports/sft/train.jsonl": sft_rows,
        "exports/grpo/train.jsonl": grpo_rows,
        "exports/heldout.jsonl": held_rows,
        "rejected.jsonl": rejected,
    }
    checksums: dict[str, str] = {}
    for rel, rows in files.items():
        path = out_dir / rel
        _write_jsonl(path, rows)
        checksums[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "name": "CERTAIN finance+math certificates",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_kept": len(kept),
        "n_train": len(train),
        "n_heldout": len(held),
        "n_rejected": len(rejected),
        "domains": sorted({p.domain for p in kept}),
        "kinds": sorted({p.kind for p in kept}),
        "gate": "hygiene + executable interpreter + gold match",
        "checksums": checksums,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tooling": {
            "curatorkit": _try_curatorkit_version(),
            "gate": "certain.curate.gate",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    card = (
        "# CERTAIN dataset card\n\n"
        f"- kept: {len(kept)}\n"
        f"- train: {len(train)}\n"
        f"- held-out: {len(held)}\n"
        f"- rejected: {len(rejected)}\n"
        "- label source: generator certificate executed by `certain.dsl.interpret`\n"
        "- a sample is kept only when the interpreter value equals the stored gold\n"
    )
    (out_dir / "dataset_card.md").write_text(card)
    (out_dir / "checksums.txt").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in checksums.items()) + "\n"
    )
    (out_dir / "lexsi_provenance.json").write_text(
        json.dumps(
            {
                "pipeline": ["ingest", "hygiene", "executable_gate", "export"],
                "executable_gate": "certain.dsl.interpret_text",
                "manifest": "manifest.json",
            },
            indent=2,
        )
    )
    return manifest


def _try_curatorkit_version() -> str | None:
    try:
        import curatorkit

        return getattr(curatorkit, "__version__", "installed")
    except Exception:
        return None


def maybe_run_curatorkit(source_jsonl: Path, output_dir: Path) -> dict[str, Any] | None:
    """Run CuratorKIT when the library is installed. Never required for export."""
    try:
        from curatorkit import Curator, CuratorConfig
    except Exception:
        return None
    cfg = CuratorConfig(
        dataset={"path": str(source_jsonl)},
        clean=True,
        dedup="minhash",
        export_formats=["alpaca", "grpo"],
        output_dir=str(output_dir / "curatorkit"),
    )
    result = Curator(cfg).run()
    summary = {"curatorkit": True}
    if hasattr(result, "print_summary"):
        result.print_summary()
    return summary
