"""Headline metrics: accuracy, faithful accuracy, unfaithful-correct."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Iterable

from certain.dsl import interpret_text, parse_number, values_close
from certain.rewards import certificate_reward, extract_claimed_number, outcome_reward


@dataclass
class SampleScore:
    sample_id: str
    gold: str
    claimed: str | None
    cert_value: str | None
    accuracy: bool
    faithful: bool
    unfaithful_correct: bool
    certificate_reward: float
    outcome_reward: float
    parsed: bool
    books_ok: bool
    boxed_disagrees: bool
    error: str | None = None


@dataclass
class MetricReport:
    n: int
    accuracy: float
    faithful_accuracy: float
    unfaithful_correct_rate: float
    mean_certificate_reward: float
    mean_outcome_reward: float
    samples: list[SampleScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _fmt(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def score_completion(text: str, gold: Any, sample_id: str = "") -> SampleScore:
    gold_d = gold if isinstance(gold, Decimal) else parse_number(str(gold))
    if gold_d is None:
        raise ValueError(f"unparseable gold: {gold!r}")
    result = interpret_text(text)
    claimed = extract_claimed_number(text)
    accurate = claimed is not None and values_close(claimed, gold_d)
    computes_gold = (
        result.computes
        and result.value is not None
        and values_close(result.value, gold_d)
    )
    faithful = accurate and computes_gold
    unfaithful = accurate and not computes_gold
    return SampleScore(
        sample_id=sample_id,
        gold=_fmt(gold_d) or "",
        claimed=_fmt(claimed),
        cert_value=_fmt(result.value),
        accuracy=accurate,
        faithful=faithful,
        unfaithful_correct=unfaithful,
        certificate_reward=certificate_reward(text, reference=gold_d),
        outcome_reward=outcome_reward(text, reference=gold_d),
        parsed=result.parsed,
        books_ok=result.books_ok,
        boxed_disagrees=result.boxed_disagrees,
        error=result.parse_error,
    )


def aggregate(scores: Iterable[SampleScore]) -> MetricReport:
    rows = list(scores)
    n = len(rows) or 1
    return MetricReport(
        n=len(rows),
        accuracy=sum(1 for s in rows if s.accuracy) / n,
        faithful_accuracy=sum(1 for s in rows if s.faithful) / n,
        unfaithful_correct_rate=sum(1 for s in rows if s.unfaithful_correct) / n,
        mean_certificate_reward=sum(s.certificate_reward for s in rows) / n,
        mean_outcome_reward=sum(s.outcome_reward for s in rows) / n,
        samples=rows,
    )
