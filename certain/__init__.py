"""CERTAIN: Certificate-Enforced Reasoning.

A closed certificate language plus the rewards, data, and metrics
used to train a local model that is paid only when the proof runs.
"""

from certain.dsl import (
    CertificateResult,
    LineResult,
    extract_blocks,
    interpret,
    interpret_text,
)
from certain.metrics import score_completion
from certain.rewards import certificate_reward, outcome_reward

__all__ = [
    "CertificateResult",
    "LineResult",
    "certificate_reward",
    "extract_blocks",
    "interpret",
    "interpret_text",
    "outcome_reward",
    "score_completion",
]

__version__ = "0.1.0"
