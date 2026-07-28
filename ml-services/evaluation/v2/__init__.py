"""Evaluator v2 contracts.

The v2 runtime is intentionally not registered yet. These models define the
boundary that later signal extraction and decision phases must satisfy.
"""

from .schemas import CallDecision, SignalBundle
from .validation import validate_decision_references

__all__ = [
    "CallDecision",
    "SignalBundle",
    "validate_decision_references",
]
