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
from .decisions import (
    DECISION_POLICY_ID,
    DECISION_POLICY_VERSION,
    build_call_decision,
    signal_bundle_sha256,
)
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    ResolvedDomainPlan,
    resolve_domain_plan,
)
from .evaluation_data import (
    EVALUATION_DATASET_ID,
    EVALUATION_DATASET_VERSION,
    CallAnnotation,
    CounterfactualPair,
    EvaluationDataset,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)
from .findings import (
    FINDINGS_VERSION,
    FindingDerivation,
    RequirementAssessment,
    RequirementAssessmentBatch,
    RequirementVerdict,
    derive_findings,
)
from .presentation import (
    OUTPUT_INVENTORY,
    PRESENTATION_VERSION,
    CallEvaluationView,
    PresentationValidationReport,
    project_call_evaluation,
)
from .runtime import (
    RUNTIME_VERSION,
    EvaluationV2Run,
    ShadowRunStatus,
    run_shadow_evaluation,
    safe_run_shadow_evaluation,
)
from .schemas import CallDecision, SignalBundle
from .semantic_assessment import (
    SEMANTIC_ASSESSOR_VERSION,
    assess_requirements,
)
from .signal_validation import (
    SIGNAL_VALIDATION_VERSION,
    AblationVariant,
    SignalValidationReport,
)
from .supervisor import (
    SUPERVISOR_VERSION,
    SupervisorContext,
    SupervisorDraft,
    SupervisorLimits,
    SupervisorResult,
    build_supervisor_context,
    build_supervisor_prompt,
    resolve_supervisor_response,
)
from .validation import validate_decision_references

__all__ = [
    "DECISION_POLICY_ID",
    "DECISION_POLICY_VERSION",
    "DEFAULT_CONFIG",
    "DYNAMICS_VERSION",
    "EVALUATION_DATASET_ID",
    "EVALUATION_DATASET_VERSION",
    "FINDINGS_VERSION",
    "OUTPUT_INVENTORY",
    "PRESENTATION_VERSION",
    "RUNTIME_VERSION",
    "SEMANTIC_ASSESSOR_VERSION",
    "SIGNAL_VALIDATION_VERSION",
    "SUPERVISOR_VERSION",
    "AblationVariant",
    "CallAnnotation",
    "CallDecision",
    "CallEvaluationView",
    "CounterfactualPair",
    "DomainProfile",
    "DynamicsConfig",
    "EvaluationDataset",
    "EvaluationV2Run",
    "FindingDerivation",
    "PresentationValidationReport",
    "ProfileSelection",
    "RequirementAssessment",
    "RequirementAssessmentBatch",
    "RequirementVerdict",
    "ResolvedDomainPlan",
    "ShadowRunStatus",
    "SignalBundle",
    "SignalValidationReport",
    "SupervisorContext",
    "SupervisorDraft",
    "SupervisorLimits",
    "SupervisorResult",
    "assess_requirements",
    "build_call_decision",
    "build_supervisor_context",
    "build_supervisor_prompt",
    "derive_acoustic_dynamics",
    "derive_findings",
    "load_evaluation_dataset",
    "project_call_evaluation",
    "resolve_domain_plan",
    "resolve_supervisor_response",
    "run_shadow_evaluation",
    "safe_run_shadow_evaluation",
    "signal_bundle_sha256",
    "validate_decision_references",
    "validate_evaluation_dataset",
]
