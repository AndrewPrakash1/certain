"""Load GSM8K and keep rows whose official scratchpad re-executes."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from certain.dsl import parse_number
from certain.finance_gen import Problem
from certain.math_gen import certificate_from_gsm8k_solution
from certain.prompts import user_prompt

GSM8K_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}

_ANS_RE = re.compile(r"####\s*(.+)\s*$", re.MULTILINE)


def _gold_from_solution(solution: str) -> str | None:
    match = _ANS_RE.search(solution)
    if not match:
        return None
    token = match.group(1).strip().replace(",", "")
    return token if parse_number(token) is not None else None


def download_gsm8k(dest: Path, split: str = "train") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"gsm8k_{split}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = GSM8K_URLS[split]
    with urllib.request.urlopen(url, timeout=60) as resp, path.open("wb") as fh:
        fh.write(resp.read())
    return path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_gsm8k(rows: list[dict], *, split: str, limit: int | None = None) -> tuple[list[Problem], list[dict]]:
    kept: list[Problem] = []
    rejected: list[dict] = []
    for i, row in enumerate(rows):
        question = row.get("question") or row.get("problem") or ""
        solution = row.get("answer") or row.get("solution") or ""
        gold = _gold_from_solution(solution)
        if not question or gold is None:
            rejected.append({"reason": "missing_gold", "question": question[:160]})
            continue
        wrapped = certificate_from_gsm8k_solution(question, solution, gold)
        if wrapped is None:
            rejected.append({"reason": "certificate_mismatch", "question": question[:160], "gold": gold})
            continue
        # unwrap certificate body
        from certain.dsl import extract_blocks

        blocks = extract_blocks(wrapped)
        problem = Problem(
            id=f"gsm8k-{split}-{i:04d}",
            domain="math",
            kind="gsm8k",
            problem=question.strip(),
            gold=gold,
            answer_name=blocks.answer_token or "s1",
            certificate=(blocks.certificate or "") + "\n",
            split=split,
            journal=False,
        )
        kept.append(problem)
        if limit is not None and len(kept) >= limit:
            break
    return kept, rejected


def ingest_gsm8k(
    cache_dir: Path,
    *,
    split: str = "train",
    limit: int | None = None,
    download: bool = True,
) -> tuple[list[Problem], list[dict]]:
    path = cache_dir / f"gsm8k_{split}.jsonl"
    if download and not path.exists():
        try:
            path = download_gsm8k(cache_dir, split)
        except Exception as exc:
            return [], [{"reason": "download_failed", "error": str(exc)}]
    if not path.exists():
        return [], [{"reason": "missing_file", "path": str(path)}]
    return filter_gsm8k(load_jsonl(path), split=split, limit=limit)
