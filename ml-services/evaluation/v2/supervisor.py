"""Bounded Supervisor context, response guard, and deterministic fallback."""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import (
    ActionExecution,
    ActionType,
    CallDecision,
    ContractModel,
    DecisionStatus,
    Finding,
    FindingCategory,
    FindingSeverity,
    Modality,
    ReliabilityStatus,
    SourceProvenance,
)

SUPERVISOR_VERSION = "0.1.0"
SUPERVISOR_SYSTEM_PROMPT = """\
You are the bounded context selector for a call evaluation.

The deterministic decision, attention value, controlling findings, and
permitted action are immutable. Do not restate, reinterpret, score, override,
or recommend changes to them.

Select only useful supporting finding, positive finding, evidence, and
uncertainty identifiers from the supplied context. The optional context_note
may describe cited call facts only. It must not state a verdict, action,
recommendation, or instruction.

Return one JSON object with exactly these keys:
supporting_finding_ids, positive_finding_ids, evidence_ids,
uncertainty_codes, context_note.
"""

_FORBIDDEN_NOTE_LANGUAGE = re.compile(
    r"\b(?:"
    r"attention|decision|action|review|"
    r"escalate|escalated|escalation|"
    r"override|ignore|approve|approval|reject|rejected|"
    r"pass|passed|fail|failed|clear|cleared|"
    r"recommend|recommended|should|must"
    r")\b",
    re.IGNORECASE,
)


class SupervisorFallbackReason(str, Enum):
    MISSING_RESPONSE = "missing_response"
    INVALID_JSON = "invalid_json"
    INVALID_CONTRACT = "invalid_contract"
    UNKNOWN_REFERENCE = "unknown_reference"
    FORBIDDEN_DECISION_LANGUAGE = "forbidden_decision_language"


class SupervisorLimits(ContractModel):
    max_triggered_findings: int = Field(default=8, ge=1, le=20)
    max_positive_findings: int = Field(default=4, ge=0, le=12)
    max_evidence_per_finding: int = Field(default=3, ge=1, le=6)
    max_contradictions_per_finding: int = Field(
        default=2,
        ge=0,
        le=4,
    )
    max_uncertainties: int = Field(default=4, ge=0, le=12)
    max_text_chars: int = Field(default=320, ge=40, le=1000)
    max_context_chars: int = Field(
        default=16000,
        ge=1000,
        le=50000,
    )


class SupervisorActionLock(ContractModel):
    action_type: ActionType
    execution: ActionExecution
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    finding_ids: list[str] = Field(default_factory=list)
    automation_allowed: bool
    requires_human_approval: bool


class SupervisorDecisionLock(ContractModel):
    call_id: str = Field(min_length=1)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1)
    decision_status: DecisionStatus
    attention_required: bool
    controlling_finding_ids: list[str] = Field(default_factory=list)
    permitted_action: SupervisorActionLock


class SupervisorEvidenceContext(ContractModel):
    evidence_id: str = Field(min_length=1)
    modality: Modality
    segment_ids: list[str] = Field(default_factory=list)
    speaker: str | None = None
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    quote: str | None = None
    observation: str | None = None
    role: Literal["evidence", "contradiction"]


class SupervisorFindingContext(ContractModel):
    finding_id: str = Field(min_length=1)
    finding_type: str = Field(min_length=1)
    category: FindingCategory
    severity: FindingSeverity
    title: str = Field(min_length=1)
    summary: str | None = None
    business_definition: str | None = None
    reliability_status: ReliabilityStatus
    reliability_reasons: list[str] = Field(default_factory=list)
    controlling: bool
    evidence: list[SupervisorEvidenceContext] = Field(default_factory=list)
    contradictions: list[SupervisorEvidenceContext] = Field(
        default_factory=list
    )


class SupervisorUncertaintyContext(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    modality: Modality | None = None


class SupervisorOmissions(ContractModel):
    triggered_findings: int = Field(default=0, ge=0)
    controlling_findings: int = Field(default=0, ge=0)
    positive_findings: int = Field(default=0, ge=0)
    uncertainties: int = Field(default=0, ge=0)
    evidence_items: int = Field(default=0, ge=0)
    contradiction_items: int = Field(default=0, ge=0)


class SupervisorContext(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    supervisor_version: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    decision_lock: SupervisorDecisionLock
    triggered_findings: list[SupervisorFindingContext] = Field(
        default_factory=list
    )
    positive_findings: list[SupervisorFindingContext] = Field(
        default_factory=list
    )
    uncertainties: list[SupervisorUncertaintyContext] = Field(
        default_factory=list
    )
    permitted_actions: list[SupervisorActionLock] = Field(
        min_length=1,
        max_length=1,
    )
    omissions: SupervisorOmissions
    context_char_limit: int = Field(ge=1000)
    context_char_count: int = Field(ge=0)
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_lock(self):
        if self.call_id != self.decision_lock.call_id:
            raise ValueError("Supervisor context call id mismatch")
        if (
            self.permitted_actions[0]
            != self.decision_lock.permitted_action
        ):
            raise ValueError(
                "permitted action must equal the deterministic lock"
            )
        if self.context_char_count > self.context_char_limit:
            raise ValueError("Supervisor context exceeds its character limit")
        finding_ids = {
            item.finding_id for item in self.triggered_findings
        }
        known_controlling = finding_ids.intersection(
            self.decision_lock.controlling_finding_ids
        )
        if len(known_controlling) + self.omissions.controlling_findings != len(
            self.decision_lock.controlling_finding_ids
        ):
            raise ValueError(
                "controlling findings must be included or counted as omitted"
            )
        return self


class SupervisorPrompt(ContractModel):
    system_prompt: str = Field(min_length=1)
    context: SupervisorContext


class SupervisorDraft(ContractModel):
    supporting_finding_ids: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    positive_finding_ids: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
    )
    uncertainty_codes: list[str] = Field(
        default_factory=list,
        max_length=4,
    )
    context_note: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def validate_unique_references(self):
        for values, label in (
            (self.supporting_finding_ids, "supporting finding"),
            (self.positive_finding_ids, "positive finding"),
            (self.evidence_ids, "evidence"),
            (self.uncertainty_codes, "uncertainty"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} reference")
        return self


class SupervisorResult(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    supervisor_version: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    decision_lock: SupervisorDecisionLock
    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    supporting_finding_ids: list[str] = Field(default_factory=list)
    positive_finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty_codes: list[str] = Field(default_factory=list)
    context_note: str | None = None
    fallback_used: bool
    fallback_reason: SupervisorFallbackReason | None = None
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_fallback(self):
        if self.call_id != self.decision_lock.call_id:
            raise ValueError("Supervisor result call id mismatch")
        if self.fallback_used != (self.fallback_reason is not None):
            raise ValueError(
                "fallback usage and reason must be set together"
            )
        return self


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.supervisor_guard",
        producer_version=SUPERVISOR_VERSION,
        method="bounded_context_strict_response_and_fallback",
    )


def decision_sha256(decision: CallDecision) -> str:
    payload = json.dumps(
        decision.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _action_lock(decision: CallDecision) -> SupervisorActionLock:
    action = decision.recommended_action
    return SupervisorActionLock(
        action_type=action.action_type,
        execution=action.execution,
        label=action.label,
        reason=action.reason,
        finding_ids=action.finding_ids,
        automation_allowed=action.automation_allowed,
        requires_human_approval=action.requires_human_approval,
    )


def _decision_lock(decision: CallDecision) -> SupervisorDecisionLock:
    return SupervisorDecisionLock(
        call_id=decision.call_id,
        decision_sha256=decision_sha256(decision),
        evaluator_version=decision.evaluator_version,
        decision_status=decision.decision_status,
        attention_required=decision.attention_required,
        controlling_finding_ids=(
            decision.decision_trace.controlling_finding_ids
        ),
        permitted_action=_action_lock(decision),
    )


def _evidence_context(
    evidence,
    *,
    role: Literal["evidence", "contradiction"],
    text_limit: int,
) -> SupervisorEvidenceContext:
    return SupervisorEvidenceContext(
        evidence_id=evidence.evidence_id,
        modality=evidence.modality,
        segment_ids=evidence.segment_ids[:4],
        speaker=(
            evidence.speaker.value if evidence.speaker else None
        ),
        start_seconds=evidence.start_seconds,
        end_seconds=evidence.end_seconds,
        quote=_truncate(evidence.quote, text_limit),
        observation=_truncate(evidence.observation, text_limit),
        role=role,
    )


def _finding_context(
    finding: Finding,
    *,
    controlling: bool,
    limits: SupervisorLimits,
    compact: bool = False,
) -> SupervisorFindingContext:
    text_limit = 100 if compact else limits.max_text_chars
    evidence_limit = 1 if compact else limits.max_evidence_per_finding
    contradiction_limit = (
        0 if compact else limits.max_contradictions_per_finding
    )
    return SupervisorFindingContext(
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        category=finding.category,
        severity=finding.severity,
        title=_truncate(finding.title, text_limit) or finding.title,
        summary=(
            None
            if compact
            else _truncate(finding.summary, text_limit)
        ),
        business_definition=(
            None
            if compact
            else _truncate(finding.business_definition, text_limit)
        ),
        reliability_status=finding.reliability.status,
        reliability_reasons=(
            []
            if compact
            else [
                _truncate(reason, text_limit) or reason
                for reason in finding.reliability.reasons[:3]
            ]
        ),
        controlling=controlling,
        evidence=[
            _evidence_context(
                item,
                role="evidence",
                text_limit=text_limit,
            )
            for item in finding.evidence[:evidence_limit]
        ],
        contradictions=[
            _evidence_context(
                item,
                role="contradiction",
                text_limit=text_limit,
            )
            for item in finding.counter_evidence[
                :contradiction_limit
            ]
        ],
    )


def _serialized_length(context: SupervisorContext) -> int:
    return len(
        json.dumps(
            context.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _set_context_length(context: SupervisorContext) -> None:
    for _ in range(4):
        measured = _serialized_length(context)
        if measured == context.context_char_count:
            return
        object.__setattr__(context, "context_char_count", measured)


def _omitted_evidence(
    findings: list[Finding],
    selected: list[SupervisorFindingContext],
) -> tuple[int, int]:
    by_id = {item.finding_id: item for item in selected}
    evidence = 0
    contradictions = 0
    for finding in findings:
        included = by_id.get(finding.finding_id)
        evidence += len(finding.evidence) - (
            len(included.evidence) if included else 0
        )
        contradictions += len(finding.counter_evidence) - (
            len(included.contradictions) if included else 0
        )
    return evidence, contradictions


def _context(
    *,
    decision: CallDecision,
    triggered: list[SupervisorFindingContext],
    positive: list[SupervisorFindingContext],
    uncertainties: list[SupervisorUncertaintyContext],
    omissions: SupervisorOmissions,
    limits: SupervisorLimits,
) -> SupervisorContext:
    lock = _decision_lock(decision)
    context = SupervisorContext(
        supervisor_version=SUPERVISOR_VERSION,
        call_id=decision.call_id,
        decision_lock=lock,
        triggered_findings=triggered,
        positive_findings=positive,
        uncertainties=uncertainties,
        permitted_actions=[lock.permitted_action],
        omissions=omissions,
        context_char_limit=limits.max_context_chars,
        context_char_count=0,
        provenance=_source(),
    )
    _set_context_length(context)
    return context


def build_supervisor_context(
    decision: CallDecision,
    limits: SupervisorLimits | None = None,
) -> SupervisorContext:
    """Select bounded decision context without raw transcript or signals."""
    limits = limits or SupervisorLimits()
    controlling_ids = set(
        decision.decision_trace.controlling_finding_ids
    )
    by_id = {
        finding.finding_id: finding
        for finding in decision.triggered_findings
    }
    controlling = [
        by_id[finding_id]
        for finding_id in (
            decision.decision_trace.controlling_finding_ids
        )
    ]
    other_triggered = sorted(
        (
            finding
            for finding in decision.triggered_findings
            if finding.finding_id not in controlling_ids
        ),
        key=lambda item: (
            item.display_priority,
            item.finding_type,
        ),
    )
    remaining_slots = max(
        0,
        limits.max_triggered_findings - len(controlling),
    )
    selected_negative = controlling + other_triggered[:remaining_slots]
    selected_positive = sorted(
        decision.positive_findings,
        key=lambda item: (
            item.display_priority,
            item.finding_type,
        ),
    )[: limits.max_positive_findings]
    selected_uncertainties = decision.uncertainties[
        : limits.max_uncertainties
    ]

    negative_context = [
        _finding_context(
            finding,
            controlling=finding.finding_id in controlling_ids,
            limits=limits,
        )
        for finding in selected_negative
    ]
    positive_context = [
        _finding_context(
            finding,
            controlling=False,
            limits=limits,
        )
        for finding in selected_positive
    ]
    uncertainty_context = [
        SupervisorUncertaintyContext(
            code=item.code,
            message=_truncate(
                item.message,
                limits.max_text_chars,
            )
            or item.message,
            modality=item.modality,
        )
        for item in selected_uncertainties
    ]
    omitted_evidence, omitted_contradictions = _omitted_evidence(
        decision.triggered_findings + decision.positive_findings,
        negative_context + positive_context,
    )
    omissions = SupervisorOmissions(
        triggered_findings=(
            len(decision.triggered_findings)
            - len(negative_context)
        ),
        positive_findings=(
            len(decision.positive_findings)
            - len(positive_context)
        ),
        uncertainties=(
            len(decision.uncertainties)
            - len(uncertainty_context)
        ),
        evidence_items=omitted_evidence,
        contradiction_items=omitted_contradictions,
    )
    context = _context(
        decision=decision,
        triggered=negative_context,
        positive=positive_context,
        uncertainties=uncertainty_context,
        omissions=omissions,
        limits=limits,
    )
    if context.context_char_count <= limits.max_context_chars:
        return context

    compact_negative = [
        _finding_context(
            finding,
            controlling=True,
            limits=limits,
            compact=True,
        )
        for finding in controlling
    ]
    while compact_negative:
        compact_omissions = SupervisorOmissions(
            triggered_findings=(
                len(decision.triggered_findings)
                - len(compact_negative)
            ),
            controlling_findings=(
                len(controlling) - len(compact_negative)
            ),
            positive_findings=len(decision.positive_findings),
            uncertainties=len(decision.uncertainties),
            evidence_items=sum(
                len(finding.evidence)
                for finding in (
                    decision.triggered_findings
                    + decision.positive_findings
                )
            )
            - sum(
                len(item.evidence) for item in compact_negative
            ),
            contradiction_items=sum(
                len(finding.counter_evidence)
                for finding in (
                    decision.triggered_findings
                    + decision.positive_findings
                )
            ),
        )
        compact_context = _context(
            decision=decision,
            triggered=compact_negative,
            positive=[],
            uncertainties=[],
            omissions=compact_omissions,
            limits=limits,
        )
        if (
            compact_context.context_char_count
            <= limits.max_context_chars
        ):
            return compact_context
        compact_negative.pop()

    empty_omissions = SupervisorOmissions(
        triggered_findings=len(decision.triggered_findings),
        controlling_findings=len(controlling),
        positive_findings=len(decision.positive_findings),
        uncertainties=len(decision.uncertainties),
        evidence_items=sum(
            len(finding.evidence)
            for finding in (
                decision.triggered_findings
                + decision.positive_findings
            )
        ),
        contradiction_items=sum(
            len(finding.counter_evidence)
            for finding in (
                decision.triggered_findings
                + decision.positive_findings
            )
        ),
    )
    minimal = _context(
        decision=decision,
        triggered=[],
        positive=[],
        uncertainties=[],
        omissions=empty_omissions,
        limits=limits,
    )
    if minimal.context_char_count > limits.max_context_chars:
        raise ValueError(
            "decision lock alone exceeds Supervisor context limit"
        )
    return minimal


def build_supervisor_prompt(
    decision: CallDecision,
    limits: SupervisorLimits | None = None,
) -> SupervisorPrompt:
    return SupervisorPrompt(
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context=build_supervisor_context(decision, limits),
    )


def _deterministic_copy(
    context: SupervisorContext,
) -> tuple[str, str]:
    lock = context.decision_lock
    if lock.attention_required:
        titles = [
            finding.title
            for finding in context.triggered_findings
            if finding.finding_id
            in lock.controlling_finding_ids
        ]
        headline = "Call requires attention"
        summary = (
            "Controlling findings: " + "; ".join(titles) + "."
            if titles
            else (
                "One or more deterministic findings require "
                "attention; detailed context was omitted by the "
                "size limit."
            )
        )
        return headline, summary
    if lock.decision_status == DecisionStatus.COMPLETE:
        return (
            "No attention finding identified",
            "No qualifying negative finding requires attention.",
        )
    return (
        "Evaluation evidence is incomplete",
        (
            "No attention finding is asserted because one or more "
            "applicable requirements remain unassessed."
        ),
    )


def _available_references(
    context: SupervisorContext,
) -> tuple[set[str], set[str], set[str], set[str]]:
    controlling = set(
        context.decision_lock.controlling_finding_ids
    )
    supporting = {
        finding.finding_id
        for finding in context.triggered_findings
        if finding.finding_id not in controlling
    }
    positive = {
        finding.finding_id
        for finding in context.positive_findings
    }
    evidence = {
        item.evidence_id
        for finding in (
            context.triggered_findings
            + context.positive_findings
        )
        for item in finding.evidence
    }
    uncertainties = {
        item.code for item in context.uncertainties
    }
    return supporting, positive, evidence, uncertainties


def _validate_draft_references(
    draft: SupervisorDraft,
    context: SupervisorContext,
) -> None:
    supporting, positive, evidence, uncertainties = (
        _available_references(context)
    )
    if not set(draft.supporting_finding_ids).issubset(supporting):
        raise LookupError("unknown supporting finding reference")
    if not set(draft.positive_finding_ids).issubset(positive):
        raise LookupError("unknown positive finding reference")
    if not set(draft.evidence_ids).issubset(evidence):
        raise LookupError("unknown evidence reference")
    if not set(draft.uncertainty_codes).issubset(uncertainties):
        raise LookupError("unknown uncertainty reference")


def _fallback(
    context: SupervisorContext,
    reason: SupervisorFallbackReason,
) -> SupervisorResult:
    headline, summary = _deterministic_copy(context)
    controlling = set(
        context.decision_lock.controlling_finding_ids
    )
    evidence_ids = [
        evidence.evidence_id
        for finding in context.triggered_findings
        if finding.finding_id in controlling
        for evidence in finding.evidence[:1]
    ][:8]
    return SupervisorResult(
        supervisor_version=SUPERVISOR_VERSION,
        call_id=context.call_id,
        decision_lock=context.decision_lock,
        headline=headline,
        summary=summary,
        positive_finding_ids=[
            item.finding_id
            for item in context.positive_findings[:2]
        ],
        evidence_ids=evidence_ids,
        uncertainty_codes=[
            item.code for item in context.uncertainties[:2]
        ],
        fallback_used=True,
        fallback_reason=reason,
        provenance=_source(),
    )


def resolve_supervisor_response(
    context: SupervisorContext,
    raw_response: str | dict[str, Any] | None,
) -> SupervisorResult:
    """Accept bounded context selection or return deterministic fallback."""
    if raw_response is None:
        return _fallback(
            context,
            SupervisorFallbackReason.MISSING_RESPONSE,
        )
    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return _fallback(
                context,
                SupervisorFallbackReason.INVALID_JSON,
            )
    elif isinstance(raw_response, dict):
        payload = raw_response
    else:
        return _fallback(
            context,
            SupervisorFallbackReason.INVALID_CONTRACT,
        )

    try:
        draft = SupervisorDraft.model_validate(payload)
    except (TypeError, ValueError):
        return _fallback(
            context,
            SupervisorFallbackReason.INVALID_CONTRACT,
        )
    try:
        _validate_draft_references(draft, context)
    except LookupError:
        return _fallback(
            context,
            SupervisorFallbackReason.UNKNOWN_REFERENCE,
        )
    if (
        draft.context_note
        and _FORBIDDEN_NOTE_LANGUAGE.search(draft.context_note)
    ):
        return _fallback(
            context,
            SupervisorFallbackReason.FORBIDDEN_DECISION_LANGUAGE,
        )

    headline, summary = _deterministic_copy(context)
    return SupervisorResult(
        supervisor_version=SUPERVISOR_VERSION,
        call_id=context.call_id,
        decision_lock=context.decision_lock,
        headline=headline,
        summary=summary,
        supporting_finding_ids=draft.supporting_finding_ids,
        positive_finding_ids=draft.positive_finding_ids,
        evidence_ids=draft.evidence_ids,
        uncertainty_codes=draft.uncertainty_codes,
        context_note=draft.context_note,
        fallback_used=False,
        provenance=_source(),
    )
