from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .schemas import (
    ActionExecution,
    ActionType,
    CallDecision,
    ContractModel,
    DecisionStatus,
    EvidenceRef,
    Finding,
    FindingCategory,
    Modality,
    ReliabilityStatus,
    SourceProvenance,
    Speaker,
    Visibility,
)
from .supervisor import SupervisorResult, decision_sha256

PRESENTATION_VERSION = "0.1.0"
MAX_PRIMARY_REASONS = 2
MAX_POSITIVE_HIGHLIGHTS = 2
MAX_EVIDENCE_ITEMS = 6


class PageState(str, Enum):
    NEEDS_ATTENTION = "needs_attention"
    NO_ATTENTION_FINDING = "no_attention_finding"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"


class Representation(str, Enum):
    STATUS_TEXT = "status_text"
    REASON_LIST = "reason_list"
    ACTION = "action"
    CHECKLIST = "checklist"
    EVIDENCE_TIMELINE = "evidence_timeline"
    NOTICE = "notice"
    DISCLOSURE = "disclosure"
    NOT_RENDERED = "not_rendered"


class EvidenceKind(str, Enum):
    TRANSCRIPT = "transcript"
    AUDIO_SUPPORT = "audio_support"
    BUSINESS_CONTEXT = "business_context"


class EvidencePurpose(str, Enum):
    REASON = "reason"
    POSITIVE = "positive"


class InventoryItem(ContractModel):
    output_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    source_fields: list[str] = Field(min_length=1)
    visibility: Visibility
    representation: Representation
    visible_when: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class PresentedFinding(ContractModel):
    finding_id: str = Field(min_length=1)
    category: FindingCategory
    category_label: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    counter_evidence_count: int = Field(ge=0)
    reliability_note: str | None = None


class PresentedEvidence(ContractModel):
    evidence_id: str = Field(min_length=1)
    finding_ids: list[str] = Field(min_length=1)
    purposes: list[EvidencePurpose] = Field(min_length=1)
    kind: EvidenceKind
    speaker: Speaker | None = None
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    text: str | None = None
    seekable: bool
    supporting_only: bool

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.seekable != (self.start_seconds is not None):
            raise ValueError("seekable must match timestamp availability")
        if self.kind == EvidenceKind.AUDIO_SUPPORT and not self.supporting_only:
            raise ValueError("audio evidence must remain supporting-only")
        if len(self.finding_ids) != len(set(self.finding_ids)):
            raise ValueError("duplicate finding reference")
        if len(self.purposes) != len(set(self.purposes)):
            raise ValueError("duplicate evidence purpose")
        return self


class PresentedAction(ContractModel):
    action_type: ActionType
    execution: ActionExecution
    label: str = Field(min_length=1)
    execution_label: str = Field(min_length=1)
    basis_finding_ids: list[str] = Field(min_length=1)
    automation_allowed: bool
    requires_human_approval: bool


class CompletenessNotice(ContractModel):
    status: DecisionStatus
    title: Literal["Evaluation incomplete"] = "Evaluation incomplete"
    message: str = Field(min_length=1)
    affected_requirements: list[str] = Field(default_factory=list)


class PresentationDetails(ContractModel):
    additional_findings: list[PresentedFinding] = Field(
        default_factory=list
    )
    additional_positive_findings: list[PresentedFinding] = Field(
        default_factory=list
    )
    evidence: list[PresentedEvidence] = Field(default_factory=list)
    uncertainty_messages: list[str] = Field(default_factory=list)
    supervisor_context_note: str | None = None
    evaluator_version: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    decision_policy_id: str = Field(min_length=1)
    decision_policy_version: str = Field(min_length=1)
    supervisor_fallback_used: bool | None = None


class CallEvaluationView(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    presentation_version: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: PageState
    evaluation_status: DecisionStatus
    attention_required: bool
    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    primary_reasons: list[PresentedFinding] = Field(
        default_factory=list,
        max_length=MAX_PRIMARY_REASONS,
    )
    additional_reason_count: int = Field(ge=0)
    positive_highlights: list[PresentedFinding] = Field(
        default_factory=list,
        max_length=MAX_POSITIVE_HIGHLIGHTS,
    )
    additional_positive_count: int = Field(ge=0)
    recommended_action: PresentedAction | None = None
    evidence: list[PresentedEvidence] = Field(
        default_factory=list,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    completeness_notice: CompletenessNotice | None = None
    details: PresentationDetails
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_view(self):
        if self.attention_required != (
            self.state == PageState.NEEDS_ATTENTION
        ):
            raise ValueError("page state must preserve attention")
        if not self.attention_required:
            expected_state = (
                PageState.EVALUATION_INCOMPLETE
                if self.evaluation_status
                == DecisionStatus.INSUFFICIENT_EVIDENCE
                else PageState.NO_ATTENTION_FINDING
            )
            if self.state != expected_state:
                raise ValueError(
                    "no-attention page state must preserve completeness"
                )
        if self.attention_required != bool(self.primary_reasons):
            raise ValueError("attention requires a visible primary reason")
        if self.attention_required != (
            self.recommended_action is not None
        ):
            raise ValueError("attention and visible action must match")
        if (
            self.evaluation_status == DecisionStatus.COMPLETE
            and self.completeness_notice is not None
        ):
            raise ValueError(
                "complete evaluation cannot carry an incomplete notice"
            )
        if (
            self.evaluation_status != DecisionStatus.COMPLETE
            and self.completeness_notice is None
        ):
            raise ValueError("incomplete evaluation requires a notice")
        if self.state == PageState.EVALUATION_INCOMPLETE and (
            self.evaluation_status == DecisionStatus.COMPLETE
        ):
            raise ValueError(
                "incomplete page state requires incomplete evaluation"
            )

        finding_groups = (
            self.primary_reasons,
            self.positive_highlights,
            self.details.additional_findings,
            self.details.additional_positive_findings,
        )
        finding_ids = [
            item.finding_id
            for group in finding_groups
            for item in group
        ]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("findings must not be repeated across regions")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence must be deduplicated")
        known_evidence = set(evidence_ids)
        for finding in (
            self.primary_reasons + self.positive_highlights
        ):
            if not set(finding.evidence_ids).issubset(known_evidence):
                raise ValueError(
                    "visible findings require projected evidence"
                )
        detail_evidence = {
            item.evidence_id for item in self.details.evidence
        }
        if known_evidence.intersection(detail_evidence):
            raise ValueError(
                "primary and detail evidence must not be repeated"
            )
        for finding in (
            self.details.additional_findings
            + self.details.additional_positive_findings
        ):
            if not set(finding.evidence_ids).issubset(
                known_evidence.union(detail_evidence)
            ):
                raise ValueError(
                    "detail findings require projected detail evidence"
                )
        return self


class PresentationCheck(ContractModel):
    attention_is_literal: bool
    reasons_are_bounded: bool
    reasons_have_evidence: bool
    positives_are_bounded: bool
    action_is_unambiguous: bool
    completeness_is_explicit: bool
    content_is_deduplicated: bool
    no_score_fields: bool
    passed: bool


class ScenarioAnswerKey(ContractModel):
    attention_state: PageState
    reasons: list[str] = Field(default_factory=list)
    evidence_locations_seconds: list[float] = Field(default_factory=list)
    positive_handling: list[str] = Field(default_factory=list)
    action: str | None = None
    completeness: DecisionStatus


class PresentationScenarioResult(ContractModel):
    call_id: str = Field(min_length=1)
    answer_key: ScenarioAnswerKey
    check: PresentationCheck


class PresentationValidationReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    presentation_version: str = Field(min_length=1)
    scenario_source: str = Field(min_length=1)
    scenario_count: int = Field(ge=0)
    passed_scenario_count: int = Field(ge=0)
    structural_checks_passed: bool
    human_comprehension_status: Literal["not_run"] = "not_run"
    human_test_questions: list[str] = Field(min_length=1)
    inventory: list[InventoryItem] = Field(min_length=1)
    scenarios: list[PresentationScenarioResult]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.scenario_count != len(self.scenarios):
            raise ValueError("scenario count does not match rows")
        passed = sum(item.check.passed for item in self.scenarios)
        if self.passed_scenario_count != passed:
            raise ValueError("passed scenario count does not match rows")
        if self.structural_checks_passed != (
            passed == self.scenario_count
        ):
            raise ValueError("structural pass flag does not match rows")
        return self


OUTPUT_INVENTORY = (
    InventoryItem(
        output_id="decision.attention_state",
        source_fields=["attention_required", "decision_status"],
        visibility=Visibility.PRIMARY,
        representation=Representation.STATUS_TEXT,
        visible_when="always",
        rationale=(
            "Answers whether this call needs attention without implying a "
            "continuous risk score."
        ),
    ),
    InventoryItem(
        output_id="decision.controlling_findings",
        source_fields=["decision_trace.controlling_finding_ids"],
        visibility=Visibility.PRIMARY,
        representation=Representation.REASON_LIST,
        visible_when="attention is required",
        rationale=(
            "At most two evidence-backed reasons explain the decision."
        ),
    ),
    InventoryItem(
        output_id="decision.recommended_action",
        source_fields=["recommended_action"],
        visibility=Visibility.PRIMARY,
        representation=Representation.ACTION,
        visible_when="attention is required",
        rationale=(
            "One constrained next action is more useful than another score."
        ),
    ),
    InventoryItem(
        output_id="findings.positive_highlights",
        source_fields=["positive_findings"],
        visibility=Visibility.PRIMARY,
        representation=Representation.CHECKLIST,
        visible_when="usable positive findings exist",
        rationale=(
            "At most two grounded positives credit handling that went well."
        ),
    ),
    InventoryItem(
        output_id="findings.selected_evidence",
        source_fields=["finding.evidence"],
        visibility=Visibility.EVIDENCE,
        representation=Representation.EVIDENCE_TIMELINE,
        visible_when="a primary reason or positive is shown",
        rationale=(
            "Quotes and timestamps answer where the finding occurred."
        ),
    ),
    InventoryItem(
        output_id="decision.completeness",
        source_fields=["decision_status", "uncertainties"],
        visibility=Visibility.PRIMARY,
        representation=Representation.NOTICE,
        visible_when="the evaluation is partial or insufficient",
        rationale=(
            "An incomplete evaluation must never look like a cleared call."
        ),
    ),
    InventoryItem(
        output_id="findings.additional_context",
        source_fields=[
            "triggered_findings",
            "positive_findings",
            "counter_evidence",
        ],
        visibility=Visibility.DETAILS,
        representation=Representation.DISCLOSURE,
        visible_when="the user opens details",
        rationale=(
            "Non-controlling findings and contradictions remain available "
            "without competing with the decision."
        ),
    ),
    InventoryItem(
        output_id="evaluation.provenance",
        source_fields=[
            "evaluator_version",
            "domain_profile_id",
            "decision_trace.policy_id",
        ],
        visibility=Visibility.DETAILS,
        representation=Representation.DISCLOSURE,
        visible_when="the user opens evaluation details",
        rationale=(
            "Version and policy context support traceability, not scanning."
        ),
    ),
    InventoryItem(
        output_id="signals.raw_model_outputs",
        source_fields=[
            "sentiment probabilities",
            "pitch",
            "volume",
            "energy",
            "speech rate",
            "normalized acoustic values",
        ],
        visibility=Visibility.INTERNAL,
        representation=Representation.NOT_RENDERED,
        visible_when="never on the call evaluator page",
        rationale=(
            "Raw model telemetry is not a business conclusion and may only "
            "support a grounded finding."
        ),
    ),
    InventoryItem(
        output_id="decision.internal_logic",
        source_fields=[
            "decision_trace.qualifications",
            "detection_rule.thresholds",
            "reliability reasons",
            "hashes",
        ],
        visibility=Visibility.INTERNAL,
        representation=Representation.NOT_RENDERED,
        visible_when="never on the call evaluator page",
        rationale=(
            "Policy mechanics belong in diagnostics and exported evidence."
        ),
    ),
    InventoryItem(
        output_id="scores.composites",
        source_fields=[
            "overall score",
            "compliance percentage",
            "workflow percentage",
            "quality rating",
            "friction score",
        ],
        visibility=Visibility.INTERNAL,
        representation=Representation.NOT_RENDERED,
        visible_when="not produced by evaluator v2",
        rationale=(
            "These aggregates mix unlike claims and create false precision."
        ),
    ),
    InventoryItem(
        output_id="timeline.duplicate_lanes",
        source_fields=[
            "compliance markers",
            "workflow markers",
            "quality markers",
            "sentiment markers",
        ],
        visibility=Visibility.INTERNAL,
        representation=Representation.NOT_RENDERED,
        visible_when="not produced by evaluator v2",
        rationale=(
            "One selected evidence timeline replaces overlapping lanes."
        ),
    ),
)


_CATEGORY_LABELS = {
    FindingCategory.REQUIRED_CONTROL: "Required step",
    FindingCategory.PROCESS: "Call handling",
    FindingCategory.ESCALATION: "Customer risk",
    FindingCategory.OUTCOME: "Outcome",
    FindingCategory.AGENT_BEHAVIOR: "Agent handling",
    FindingCategory.DATA_QUALITY: "Evaluation quality",
}


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.presentation",
        producer_version=PRESENTATION_VERSION,
        method="bounded_decision_preserving_page_projection",
    )


def _state(decision: CallDecision) -> PageState:
    if decision.attention_required:
        return PageState.NEEDS_ATTENTION
    if (
        decision.decision_status
        == DecisionStatus.INSUFFICIENT_EVIDENCE
    ):
        return PageState.EVALUATION_INCOMPLETE
    return PageState.NO_ATTENTION_FINDING


def _copy(decision: CallDecision) -> tuple[str, str]:
    state = _state(decision)
    if state == PageState.NEEDS_ATTENTION:
        return (
            "Needs attention",
            "One or more evidence-backed findings require review.",
        )
    if state == PageState.NO_ATTENTION_FINDING:
        if decision.decision_status == DecisionStatus.PARTIAL:
            return (
                "No review finding identified",
                (
                    "Assessed requirements produced no qualifying negative "
                    "finding; one or more requirements remain uncertain."
                ),
            )
        return (
            "No review finding identified",
            "Applicable rules produced no qualifying negative finding.",
        )
    return (
        "Evaluation incomplete",
        (
            "No attention finding is asserted because applicable "
            "requirements remain unassessed."
        ),
    )


def _ordered(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda item: (
            item.display_priority,
            item.category.value,
            item.finding_type,
            item.finding_id,
        ),
    )


def _supervisor_order(
    findings: list[Finding],
    selected_ids: list[str],
) -> list[Finding]:
    by_id = {item.finding_id: item for item in findings}
    selected = [
        by_id[finding_id]
        for finding_id in selected_ids
        if finding_id in by_id
    ]
    selected_set = {item.finding_id for item in selected}
    return selected + [
        item
        for item in _ordered(findings)
        if item.finding_id not in selected_set
    ]


def _verify_supervisor(
    decision: CallDecision,
    supervisor: SupervisorResult | None,
) -> None:
    if supervisor is None:
        return
    lock = supervisor.decision_lock
    if (
        supervisor.call_id != decision.call_id
        or lock.call_id != decision.call_id
        or lock.decision_sha256 != decision_sha256(decision)
    ):
        raise ValueError(
            "supervisor result does not match the deterministic decision"
        )


def _evidence_kind(evidence: EvidenceRef) -> EvidenceKind:
    if evidence.modality in (Modality.TRANSCRIPT, Modality.TEXT):
        return EvidenceKind.TRANSCRIPT
    if evidence.modality in (Modality.ACOUSTIC, Modality.MULTIMODAL):
        return EvidenceKind.AUDIO_SUPPORT
    return EvidenceKind.BUSINESS_CONTEXT


def _selected_evidence(
    finding: Finding,
    supervisor_ids: set[str],
    limit: int,
) -> list[EvidenceRef]:
    ranked = sorted(
        finding.evidence,
        key=lambda item: (
            item.evidence_id not in supervisor_ids,
            _evidence_kind(item) == EvidenceKind.AUDIO_SUPPORT,
            item.start_seconds is None,
            item.start_seconds or 0.0,
            item.evidence_id,
        ),
    )
    selected: list[EvidenceRef] = []
    kinds: set[EvidenceKind] = set()
    for evidence in ranked:
        kind = _evidence_kind(evidence)
        if kind in kinds:
            continue
        selected.append(evidence)
        kinds.add(kind)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected_ids = {item.evidence_id for item in selected}
        remaining = [
            item
            for item in ranked
            if item.evidence_id not in selected_ids
        ]
        selected.extend(remaining[: limit - len(selected)])
    return selected


def _presented_finding(
    finding: Finding,
    evidence: list[EvidenceRef],
) -> PresentedFinding:
    reliability_note = None
    if finding.reliability.status == ReliabilityStatus.LIMITED:
        reliability_note = "Supporting evidence is limited."
    return PresentedFinding(
        finding_id=finding.finding_id,
        category=finding.category,
        category_label=_CATEGORY_LABELS[finding.category],
        title=finding.title,
        summary=finding.summary,
        evidence_ids=[item.evidence_id for item in evidence],
        counter_evidence_count=len(finding.counter_evidence),
        reliability_note=reliability_note,
    )


def _presented_evidence(
    evidence: EvidenceRef,
    finding: Finding,
    purpose: EvidencePurpose,
) -> PresentedEvidence:
    kind = _evidence_kind(evidence)
    return PresentedEvidence(
        evidence_id=evidence.evidence_id,
        finding_ids=[finding.finding_id],
        purposes=[purpose],
        kind=kind,
        speaker=evidence.speaker,
        start_seconds=evidence.start_seconds,
        end_seconds=evidence.end_seconds,
        text=evidence.quote or evidence.observation,
        seekable=evidence.start_seconds is not None,
        supporting_only=kind == EvidenceKind.AUDIO_SUPPORT,
    )


def _merge_evidence(
    items: list[PresentedEvidence],
    limit: int | None = None,
) -> list[PresentedEvidence]:
    merged: dict[str, PresentedEvidence] = {}
    for item in items:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item
            continue
        existing.finding_ids = sorted(
            set(existing.finding_ids + item.finding_ids)
        )
        existing.purposes = sorted(
            set(existing.purposes + item.purposes),
            key=lambda value: value.value,
        )
    ordered = sorted(
        merged.values(),
        key=lambda item: (
            item.start_seconds is None,
            item.start_seconds or 0.0,
            item.evidence_id,
        ),
    )
    return ordered if limit is None else ordered[:limit]


def _present_action(decision: CallDecision) -> PresentedAction | None:
    action = decision.recommended_action
    if action.action_type == ActionType.NONE:
        return None
    execution_label = (
        "Automatic"
        if action.execution == ActionExecution.AUTOMATIC
        else "Manager approval required"
    )
    return PresentedAction(
        action_type=action.action_type,
        execution=action.execution,
        label=action.label,
        execution_label=execution_label,
        basis_finding_ids=action.finding_ids,
        automation_allowed=action.automation_allowed,
        requires_human_approval=action.requires_human_approval,
    )


def _completeness_notice(
    decision: CallDecision,
) -> CompletenessNotice | None:
    if decision.decision_status == DecisionStatus.COMPLETE:
        return None
    message = (
        "Some applicable requirements could not be assessed. Visible "
        "findings remain valid, but this is not a complete call assessment."
        if decision.decision_status == DecisionStatus.PARTIAL
        else (
            "Applicable requirements could not be assessed, so this call "
            "has not been cleared."
        )
    )
    return CompletenessNotice(
        status=decision.decision_status,
        message=message,
        affected_requirements=[
            item.message
            for item in decision.uncertainties
            if item.code.startswith("requirement.")
        ],
    )


def project_call_evaluation(
    decision: CallDecision,
    supervisor: SupervisorResult | None = None,
) -> CallEvaluationView:
    """Project a decision into the minimal call-evaluator page contract."""
    _verify_supervisor(decision, supervisor)
    controlling_ids = set(
        decision.decision_trace.controlling_finding_ids
    )
    controlling = _ordered(
        [
            item
            for item in decision.triggered_findings
            if item.finding_id in controlling_ids
        ]
    )
    primary_source = controlling[:MAX_PRIMARY_REASONS]
    primary_ids = {item.finding_id for item in primary_source}

    visible_negative = [
        item
        for item in decision.triggered_findings
        if item.visibility != Visibility.INTERNAL
        and item.finding_id not in primary_ids
    ]
    visible_negative = _supervisor_order(
        visible_negative,
        supervisor.supporting_finding_ids if supervisor else [],
    )

    visible_positive = [
        item
        for item in decision.positive_findings
        if item.visibility != Visibility.INTERNAL
    ]
    visible_positive = _supervisor_order(
        visible_positive,
        supervisor.positive_finding_ids if supervisor else [],
    )
    positive_source = visible_positive[:MAX_POSITIVE_HIGHLIGHTS]
    positive_ids = {item.finding_id for item in positive_source}
    evidence_preferences = set(
        supervisor.evidence_ids if supervisor else []
    )

    primary: list[PresentedFinding] = []
    positives: list[PresentedFinding] = []
    evidence_items: list[PresentedEvidence] = []
    for finding in primary_source:
        selected = _selected_evidence(
            finding,
            evidence_preferences,
            limit=2,
        )
        primary.append(_presented_finding(finding, selected))
        evidence_items.extend(
            _presented_evidence(
                item,
                finding,
                EvidencePurpose.REASON,
            )
            for item in selected
        )
    for finding in positive_source:
        selected = _selected_evidence(
            finding,
            evidence_preferences,
            limit=1,
        )
        positives.append(_presented_finding(finding, selected))
        evidence_items.extend(
            _presented_evidence(
                item,
                finding,
                EvidencePurpose.POSITIVE,
            )
            for item in selected
        )

    additional_negative = [
        _presented_finding(
            finding,
            _selected_evidence(
                finding,
                evidence_preferences,
                limit=1,
            ),
        )
        for finding in visible_negative
    ]
    additional_positive = [
        _presented_finding(
            finding,
            _selected_evidence(
                finding,
                evidence_preferences,
                limit=1,
            ),
        )
        for finding in visible_positive
        if finding.finding_id not in positive_ids
    ]
    detail_evidence_items: list[PresentedEvidence] = []
    for finding, presented in zip(
        visible_negative,
        additional_negative,
        strict=True,
    ):
        selected_ids = set(presented.evidence_ids)
        detail_evidence_items.extend(
            _presented_evidence(
                item,
                finding,
                EvidencePurpose.REASON,
            )
            for item in finding.evidence
            if item.evidence_id in selected_ids
        )
    additional_positive_source = [
        item
        for item in visible_positive
        if item.finding_id not in positive_ids
    ]
    for finding, presented in zip(
        additional_positive_source,
        additional_positive,
        strict=True,
    ):
        selected_ids = set(presented.evidence_ids)
        detail_evidence_items.extend(
            _presented_evidence(
                item,
                finding,
                EvidencePurpose.POSITIVE,
            )
            for item in finding.evidence
            if item.evidence_id in selected_ids
        )
    primary_evidence = _merge_evidence(
        evidence_items,
        limit=MAX_EVIDENCE_ITEMS,
    )
    primary_evidence_ids = {
        item.evidence_id for item in primary_evidence
    }
    detail_evidence = [
        item
        for item in _merge_evidence(detail_evidence_items)
        if item.evidence_id not in primary_evidence_ids
    ]
    headline, summary = _copy(decision)
    return CallEvaluationView(
        presentation_version=PRESENTATION_VERSION,
        call_id=decision.call_id,
        decision_sha256=decision_sha256(decision),
        state=_state(decision),
        evaluation_status=decision.decision_status,
        attention_required=decision.attention_required,
        headline=headline,
        summary=summary,
        primary_reasons=primary,
        additional_reason_count=len(additional_negative),
        positive_highlights=positives,
        additional_positive_count=len(additional_positive),
        recommended_action=_present_action(decision),
        evidence=primary_evidence,
        completeness_notice=_completeness_notice(decision),
        details=PresentationDetails(
            additional_findings=additional_negative,
            additional_positive_findings=additional_positive,
            evidence=detail_evidence,
            uncertainty_messages=[
                item.message
                for item in decision.uncertainties
                if item.visibility != Visibility.INTERNAL
            ],
            supervisor_context_note=(
                supervisor.context_note if supervisor else None
            ),
            evaluator_version=decision.evaluator_version,
            domain_profile_id=decision.domain_profile_id,
            decision_policy_id=decision.decision_trace.policy_id,
            decision_policy_version=(
                decision.decision_trace.policy_version
            ),
            supervisor_fallback_used=(
                supervisor.fallback_used if supervisor else None
            ),
        ),
        provenance=_source(),
    )


def check_presentation(
    view: CallEvaluationView,
) -> PresentationCheck:
    payload = view.model_dump(mode="json")
    serialized_keys: list[str] = []

    def collect_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                serialized_keys.append(key)
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(payload)
    forbidden_fragments = ("score", "percentage", "rating", "confidence")
    no_score_fields = not any(
        fragment in key.lower()
        for key in serialized_keys
        for fragment in forbidden_fragments
    )
    visible_ids = [
        item.finding_id
        for item in (
            view.primary_reasons
            + view.positive_highlights
            + view.details.additional_findings
            + view.details.additional_positive_findings
        )
    ]
    evidence_ids = {item.evidence_id for item in view.evidence}
    all_evidence_ids = [
        item.evidence_id
        for item in view.evidence + view.details.evidence
    ]
    reasons_have_evidence = all(
        set(item.evidence_ids).issubset(evidence_ids)
        and bool(item.evidence_ids)
        for item in view.primary_reasons
    )
    action_is_unambiguous = (
        view.recommended_action is not None
    ) == view.attention_required
    completeness_is_explicit = (
        view.evaluation_status == DecisionStatus.COMPLETE
        and view.completeness_notice is None
    ) or (
        view.evaluation_status != DecisionStatus.COMPLETE
        and view.completeness_notice is not None
    )
    checks = {
        "attention_is_literal": (
            view.state.value
            in {
                "needs_attention",
                "no_attention_finding",
                "evaluation_incomplete",
            }
        ),
        "reasons_are_bounded": (
            len(view.primary_reasons) <= MAX_PRIMARY_REASONS
        ),
        "reasons_have_evidence": reasons_have_evidence,
        "positives_are_bounded": (
            len(view.positive_highlights)
            <= MAX_POSITIVE_HIGHLIGHTS
        ),
        "action_is_unambiguous": action_is_unambiguous,
        "completeness_is_explicit": completeness_is_explicit,
        "content_is_deduplicated": (
            len(visible_ids) == len(set(visible_ids))
            and len(all_evidence_ids) == len(set(all_evidence_ids))
        ),
        "no_score_fields": no_score_fields,
    }
    return PresentationCheck(
        **checks,
        passed=all(checks.values()),
    )


def scenario_result(
    view: CallEvaluationView,
) -> PresentationScenarioResult:
    return PresentationScenarioResult(
        call_id=view.call_id,
        answer_key=ScenarioAnswerKey(
            attention_state=view.state,
            reasons=[item.title for item in view.primary_reasons],
            evidence_locations_seconds=[
                item.start_seconds
                for item in view.evidence
                if item.start_seconds is not None
            ],
            positive_handling=[
                item.title for item in view.positive_highlights
            ],
            action=(
                view.recommended_action.label
                if view.recommended_action
                else None
            ),
            completeness=view.evaluation_status,
        ),
        check=check_presentation(view),
    )
