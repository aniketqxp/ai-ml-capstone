"""Evaluator v2 contracts.

The v2 runtime is intentionally not registered yet. These models define the
boundary that later signal extraction and decision phases must satisfy.
"""

from .acoustic_dynamics import (
    DEFAULT_CONFIG,
    DYNAMICS_VERSION,
    DynamicsConfig,
    derive_acoustic_dynamics,
)
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    ResolvedDomainPlan,
    resolve_domain_plan,
)
from .schemas import CallDecision, SignalBundle
from .validation import validate_decision_references

__all__ = [
    "CallDecision",
    "DEFAULT_CONFIG",
    "DYNAMICS_VERSION",
    "DomainProfile",
    "DynamicsConfig",
    "ProfileSelection",
    "ResolvedDomainPlan",
    "SignalBundle",
    "derive_acoustic_dynamics",
    "resolve_domain_plan",
    "validate_decision_references",
]
