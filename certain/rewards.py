"""Executable rewards. No learned reward model.

Certificate GRPO is paid by the interpreter. Outcome GRPO is paid only
when a boxed / extracted number matches gold. The two runs share a
trainer, a prompt, and a token budget; only this file changes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence

from certain.dsl import interpret_text, parse_number, values_close

PARSE_WEIGHT = 0.2
DEFINED_WEIGHT = 0.2
BOOKS_WEIGHT = 0.2
MATCH_WEIGHT = 0.4


def _as_gold(reference: Any) -> Decimal | None:
    if reference is None:
        return None
    if isinstance(reference, Decimal):
        return reference
    if isinstance(reference, (int, float)):
        return Decimal(str(reference))
    if isinstance(reference, dict):
        for key in ("gold", "answer", "ground_truth", "target", "reference"):
            if key in reference:
                return _as_gold(reference[key])
        return None
    return parse_number(str(reference))


def _gold_from_kwargs(kwargs: dict[str, Any]) -> Decimal | None:
    for key in ("gold", "answer", "ground_truth", "target", "reference"):
        if key in kwargs:
            value = kwargs[key]
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            gold = _as_gold(value)
            if gold is not None:
                return gold
    return None


def certificate_reward(text: str, reference: Any = None, **kwargs: Any) -> float:
    """Shaped reward in [0, 1]. Illegal certificates score 0."""
    gold = _as_gold(reference) if reference is not None else _gold_from_kwargs(kwargs)
    result = interpret_text(text or "")
    if not result.parsed:
        return 0.0
    tenths = 2
    if result.no_undefined:
        tenths += 2
    if result.books_ok:
        tenths += 2
    if (
        gold is not None
        and result.value is not None
        and result.no_undefined
        and result.books_ok
        and not result.boxed_disagrees
        and values_close(result.value, gold)
    ):
        tenths += 4
    return tenths / 10.0


def outcome_reward(text: str, reference: Any = None, **kwargs: Any) -> float:
    """Binary reward on the extracted number. The certificate is ignored."""
    gold = _as_gold(reference) if reference is not None else _gold_from_kwargs(kwargs)
    if gold is None:
        return 0.0
    extracted = extract_claimed_number(text or "")
    return 1.0 if extracted is not None and values_close(extracted, gold) else 0.0


def extract_claimed_number(text: str) -> Decimal | None:
    result = interpret_text(text)
    if result.boxed_value is not None:
        return result.boxed_value
    if result.answer_name and result.value is not None:
        return result.value
    if result.answer_name is None:
        maybe = None
        extracted = result
        if extracted.value is not None:
            maybe = extracted.value
        if maybe is not None:
            return maybe
    # last standalone number in the completion, used only by outcome reward
    matches = list(__import__("re").finditer(r"-?\$?\d+(?:\.\d+)?", text or ""))
    if not matches:
        return None
    return parse_number(matches[-1].group(0).replace("$", ""))


def batch_rewards(
    fn,
    completions: Sequence[str],
    **kwargs: Any,
) -> list[float]:
    """TRL / AlignTune-style batched callable."""
    golds = None
    for key in ("gold", "answer", "ground_truth", "target"):
        if key in kwargs:
            golds = kwargs[key]
            break
    if golds is None:
        golds = [None] * len(completions)
    elif not isinstance(golds, (list, tuple)):
        golds = [golds] * len(completions)
    out: list[float] = []
    for text, gold in zip(completions, golds):
        completion = text
        if isinstance(text, list):
            # chat-style: [{'role': 'assistant', 'content': '...'}]
            completion = text[-1]["content"] if text and isinstance(text[0], dict) else str(text)
        out.append(float(fn(completion, reference=gold)))
    return out


def certificate_reward_batch(completions: Sequence[str], **kwargs: Any) -> list[float]:
    return batch_rewards(certificate_reward, completions, **kwargs)


def outcome_reward_batch(completions: Sequence[str], **kwargs: Any) -> list[float]:
    return batch_rewards(outcome_reward, completions, **kwargs)


def format_only_reward(text: str, reference: Any = None, **kwargs: Any) -> float:
    """Control: pay for a certificate block, never for truth. Buffer-time extra."""
    result = interpret_text(text or "")
    return PARSE_WEIGHT if result.parsed else 0.0


def register_aligntune_rewards() -> list[str]:
    """Register CERTAIN rewards with AlignTune if the library is installed."""
    registered: list[str] = []
    try:
        from aligntune.rewards.core import RewardConfig, RewardFunction, RewardType
        from aligntune.rewards.registry import RewardRegistry
    except Exception:
        return registered

    class _CertReward(RewardFunction):
        def compute(self, text: str, reference=None, **kwargs) -> float:
            return certificate_reward(text, reference=reference, **kwargs)

    class _OutcomeReward(RewardFunction):
        def compute(self, text: str, reference=None, **kwargs) -> float:
            return outcome_reward(text, reference=reference, **kwargs)

    class _FormatReward(RewardFunction):
        def compute(self, text: str, reference=None, **kwargs) -> float:
            return format_only_reward(text, reference=reference, **kwargs)

    mapping = {
        "certain_certificate": (_CertReward, RewardType.HELPFULNESS),
        "certain_outcome": (_OutcomeReward, RewardType.HELPFULNESS),
        "certain_format": (_FormatReward, RewardType.LENGTH),
    }
    for name, (cls, rtype) in mapping.items():
        try:
            RewardRegistry.register_custom_reward(
                name, cls, RewardConfig(reward_type=rtype, weight=1.0)
            )
            registered.append(name)
        except Exception:
            continue
    return registered


def group_advantages(rewards: Iterable[float]) -> list[float]:
    """GRPO-style group-relative advantages. Used by the CPU kill-test."""
    values = [float(r) for r in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = var**0.5
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]
