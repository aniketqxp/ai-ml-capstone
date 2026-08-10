from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .acoustic_dynamics import derive_acoustic_dynamics
from .adapters import build_signal_bundle
from .decisions import DECISION_POLICY_VERSION, build_call_decision
from .domain_profiles import (
    DomainProfile,
    ProfileSelection,
    load_general_service_profile,
    resolve_domain_plan,
)
from .findings import (
    RequirementAssessmentBatch,
    derive_findings,
)
from .presentation import CallEvaluationView, project_call_evaluation
from .schemas import (
    CallDecision,
    ContractModel,
    DecisionStatus,
    SourceProvenance,
)
from .semantic_assessment import assess_requirements
from .validation import validate_decision_references

RUNTIME_VERSION = "0.1.0"
HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / "profiles" / "banking_v1.json"
SELECTIONS_PATH = HERE / "profiles" / "banking_call_selections.json"


class ShadowRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    UNSUPPORTED_DOMAIN = "unsupported_domain"
    PROFILE_SELECTION_UNAVAILABLE = "profile_selection_unavailable"
    FAILED = "failed"


class LegacyAttentionProxy(ContractModel):
    evaluator_version: str = Field(min_length=1)
    attention_required: bool
    basis: Literal["failed_compliance_or_escalation"]
    failed_compliance_count: int = Field(ge=0)
    escalation_tier: str = Field(min_length=1)


class DecisionComparison(ContractModel):
    comparable: bool
    reason: str = Field(min_length=1)
    attention_agreement: bool | None = None

    @model_validator(mode="after")
    def validate_agreement(self):
        if self.comparable != (self.attention_agreement is not None):
            raise ValueError(
                "attention agreement is present only when runs are comparable"
            )
        return self


class EvaluationV2Run(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    runtime_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    mode: Literal["shadow"] = "shadow"
    status: ShadowRunStatus
    evaluator_version: str = Field(min_length=1)
    profile_id: str | None = None
    decision_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    decision: CallDecision | None = None
    presentation: CallEvaluationView | None = None
    legacy_proxy: LegacyAttentionProxy | None = None
    comparison: DecisionComparison | None = None
    limitations: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_run(self):
        succeeded = self.status == ShadowRunStatus.SUCCEEDED
        if succeeded != (
            self.decision is not None and self.presentation is not None
        ):
            raise ValueError(
                "successful shadow runs require decision and presentation"
            )
        if succeeded != (self.decision_sha256 is not None):
            raise ValueError(
                "successful shadow runs require a decision hash"
            )
        if self.status == ShadowRunStatus.FAILED:
            if not self.failure_reason:
                raise ValueError("failed shadow runs require a reason")
        elif self.failure_reason is not None:
            raise ValueError(
                "only failed shadow runs may include a failure reason"
            )
        if self.comparison is not None and self.legacy_proxy is None:
            raise ValueError("comparison requires a legacy proxy")
        return self


def _source(method: str) -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.runtime",
        producer_version=RUNTIME_VERSION,
        method=method,
    )


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(call_id: str, marker: str) -> str:
    digest = hashlib.sha256(
        f"{call_id}:{RUNTIME_VERSION}:{marker}".encode()
    ).hexdigest()[:16]
    return f"{call_id}:v2:{digest}"


def _profile() -> DomainProfile:
    return DomainProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )


def _general_selection(call_id: str) -> ProfileSelection:
    return ProfileSelection(
        call_id=call_id,
        intent_ids=["service.general_request"],
        selection_method="general_service_fallback",
    )


def _known_selections() -> dict[str, ProfileSelection]:
    payload = json.loads(SELECTIONS_PATH.read_text(encoding="utf-8"))
    return {
        item["call_id"]: ProfileSelection.model_validate(item)
        for item in payload["calls"]
    }


def _selection(
    call_id: str,
    explicit: ProfileSelection | dict[str, Any] | None,
) -> ProfileSelection | None:
    if explicit is not None:
        selection = (
            explicit
            if isinstance(explicit, ProfileSelection)
            else ProfileSelection.model_validate(explicit)
        )
        if selection.call_id != call_id:
            raise ValueError("profile selection call_id does not match")
        return selection
    return _known_selections().get(call_id)


def legacy_attention_proxy(
    evaluation: dict[str, Any] | None,
) -> LegacyAttentionProxy | None:
    if not evaluation:
        return None
    compliance = evaluation.get("compliance") or {}
    failed = sum(
        isinstance(item, dict) and item.get("passed") is False
        for item in compliance.values()
    )
    escalation = str(
        (evaluation.get("escalation") or {}).get("risk_level")
        or "none"
    ).lower()
    return LegacyAttentionProxy(
        evaluator_version=str(
            (evaluation.get("_evaluator") or {}).get("version")
            or evaluation.get("rubric_version")
            or "v1"
        ),
        attention_required=failed > 0 or escalation in {
            "review",
            "escalate",
        },
        basis="failed_compliance_or_escalation",
        failed_compliance_count=failed,
        escalation_tier=escalation,
    )


def _comparison(
    decision: CallDecision,
    proxy: LegacyAttentionProxy | None,
) -> DecisionComparison | None:
    if proxy is None:
        return None
    if decision.decision_status != DecisionStatus.COMPLETE:
        return DecisionComparison(
            comparable=False,
            reason=(
                "v2 requirement coverage is incomplete; attention values "
                "must not be treated as equivalent."
            ),
        )
    return DecisionComparison(
        comparable=True,
        reason=(
            "Both evaluators produced a complete attention value under "
            "their named policies."
        ),
        attention_agreement=(
            proxy.attention_required == decision.attention_required
        ),
    )


def _unavailable(
    *,
    call_id: str,
    status: ShadowRunStatus,
    limitation: str,
    legacy_proxy: LegacyAttentionProxy | None,
) -> EvaluationV2Run:
    return EvaluationV2Run(
        runtime_version=RUNTIME_VERSION,
        run_id=_run_id(call_id, status.value),
        call_id=call_id,
        status=status,
        evaluator_version=f"v2-policy-{DECISION_POLICY_VERSION}",
        legacy_proxy=legacy_proxy,
        limitations=[limitation],
        provenance=_source("feature_flagged_shadow_unavailable"),
    )


def run_shadow_evaluation(
    *,
    transcript: dict[str, Any],
    sentiment: dict[str, Any] | None = None,
    legacy_evaluation: dict[str, Any] | None = None,
    profile_selection: ProfileSelection | dict[str, Any] | None = None,
    assessments: RequirementAssessmentBatch | dict[str, Any] | None = None,
    run_semantic_assessor: bool = True,
    transcript_source: str | None = None,
    sentiment_source: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvaluationV2Run:
    report = progress or (lambda _stage: None)
    report("evaluating_v2_prepare")
    call_id = str(transcript.get("call_id") or "").strip()
    if not call_id:
        raise ValueError("transcript call_id is required")
    domain = str(transcript.get("domain") or "").strip().lower()
    proxy = legacy_attention_proxy(legacy_evaluation)
    selection = _selection(call_id, profile_selection)
    uses_general_profile = domain != "banking" or selection is None
    if uses_general_profile:
        profile = load_general_service_profile()
        selection = _general_selection(call_id)
    else:
        profile = _profile()
    plan = resolve_domain_plan(profile, selection)
    report("evaluating_v2_signals")
    bundle = build_signal_bundle(
        transcript,
        sentiment,
        transcript_source=transcript_source,
        sentiment_source=sentiment_source,
    )
    if sentiment:
        bundle = derive_acoustic_dynamics(bundle)
    assessment_batch = None
    if assessments is not None:
        assessment_batch = (
            assessments
            if isinstance(assessments, RequirementAssessmentBatch)
            else RequirementAssessmentBatch.model_validate(assessments)
        )
    elif run_semantic_assessor:
        report("evaluating_v2_requirements")
        assessment_batch = assess_requirements(bundle, plan)
    report("evaluating_v2_findings")
    derivation = derive_findings(bundle, plan, assessment_batch)
    report("evaluating_v2_decision")
    decision = build_call_decision(bundle, derivation)
    validate_decision_references(decision, bundle)
    report("evaluating_v2_presentation")
    presentation = project_call_evaluation(decision, bundle=bundle)
    limitations = [
        "research_domain_profile",
        "bounded_supervisor_not_run",
    ]
    if uses_general_profile:
        limitations.append("general_service_profile_not_domain_compliance")
    if assessment_batch is None:
        limitations.append("semantic_requirement_assessments_missing")
    if not sentiment:
        limitations.append("acoustic_input_unavailable")
    marker = presentation.decision_sha256
    return EvaluationV2Run(
        runtime_version=RUNTIME_VERSION,
        run_id=_run_id(call_id, marker),
        call_id=call_id,
        status=ShadowRunStatus.SUCCEEDED,
        evaluator_version=decision.evaluator_version,
        profile_id=plan.profile_id,
        decision_sha256=presentation.decision_sha256,
        decision=decision,
        presentation=presentation,
        legacy_proxy=proxy,
        comparison=_comparison(decision, proxy),
        limitations=limitations,
        provenance=_source("feature_flagged_shadow_evaluation"),
    )


def safe_run_shadow_evaluation(**kwargs) -> EvaluationV2Run:
    try:
        return run_shadow_evaluation(**kwargs)
    # Shadow execution must record any failure without failing the v1 job.
    except Exception as exc:  # noqa: BLE001
        transcript = kwargs.get("transcript") or {}
        call_id = str(transcript.get("call_id") or "unknown-call")
        proxy = legacy_attention_proxy(kwargs.get("legacy_evaluation"))
        reason = f"{type(exc).__name__}: {exc}"
        return EvaluationV2Run(
            runtime_version=RUNTIME_VERSION,
            run_id=_run_id(call_id, _sha256(reason)),
            call_id=call_id,
            status=ShadowRunStatus.FAILED,
            evaluator_version=f"v2-policy-{DECISION_POLICY_VERSION}",
            legacy_proxy=proxy,
            limitations=["shadow_runtime_failure"],
            failure_reason=reason[:500],
            provenance=_source("feature_flagged_shadow_failure"),
        )
