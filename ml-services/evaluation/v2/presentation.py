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
    SignalBundle,
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


class FindingOutcome(str, Enum):
    EFFECTIVE = "effective"
    INCORRECT = "incorrect"
    MISSED = "missed"
    OBSERVED_CONCERN = "observed_concern"


class ManagerAnswer(str, Enum):
    YES = "yes"
    PARTLY = "partly"
    NO = "no"
    UNCLEAR = "unclear"


class CheckStatus(str, Enum):
    DEMONSTRATED = "demonstrated"
    INCORRECT = "incorrect"
    NOT_DEMONSTRATED = "not_demonstrated"
    UNABLE_TO_DETERMINE = "unable_to_determine"


class AcousticStatus(str, Enum):
    AVAILABLE = "available"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


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
    outcome: FindingOutcome = FindingOutcome.OBSERVED_CONCERN


class ManagerQuestion(ContractModel):
    question_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    question: str = Field(min_length=1)
    answer: ManagerAnswer
    answer_label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class PresentedCheck(ContractModel):
    requirement_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: FindingCategory
    status: CheckStatus
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    promoted: bool = False


class AcousticObservation(ContractModel):
    observation_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    supporting_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_time_range(self):
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("acoustic observation has an invalid time range")
        return self


class AcousticContext(ContractModel):
    status: AcousticStatus
    coverage_label: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    observations: list[AcousticObservation] = Field(default_factory=list)


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
    delivery: Literal["email"] = "email"
    audience: Literal["manager", "customer"] = "manager"


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
    manager_questions: list[ManagerQuestion] = Field(
        default_factory=list,
        max_length=4,
    )
    checklist: list[PresentedCheck] = Field(default_factory=list)
    acoustic_context: AcousticContext | None = None
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


def _positive_order(findings: list[Finding]) -> list[Finding]:
    category_rank = {
        FindingCategory.OUTCOME: 0,
        FindingCategory.REQUIRED_CONTROL: 1,
        FindingCategory.PROCESS: 2,
        FindingCategory.AGENT_BEHAVIOR: 3,
        FindingCategory.ESCALATION: 4,
        FindingCategory.DATA_QUALITY: 5,
    }
    return sorted(
        findings,
        key=lambda item: (
            category_rank[item.category],
            item.display_priority,
            item.finding_id,
        ),
    )


def _supervisor_positive_order(
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
        for item in _positive_order(findings)
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


def _finding_outcome(finding: Finding) -> FindingOutcome:
    if finding.polarity.value == "positive":
        return FindingOutcome.EFFECTIVE
    verdict = next(
        (
            threshold.value
            for threshold in finding.detection_rule.thresholds
            if threshold.signal_name == "assessment.requirement_verdict"
        ),
        None,
    )
    if verdict == "incorrect":
        return FindingOutcome.INCORRECT
    if verdict == "missed":
        return FindingOutcome.MISSED
    if finding.category == FindingCategory.ESCALATION:
        return FindingOutcome.OBSERVED_CONCERN
    return FindingOutcome.INCORRECT


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
        outcome=_finding_outcome(finding),
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
        "Automatic manager notification"
        if action.execution == ActionExecution.AUTOMATIC
        else "Manager approval required before email delivery"
    )
    return PresentedAction(
        action_type=action.action_type,
        execution=action.execution,
        label=action.label,
        execution_label=execution_label,
        basis_finding_ids=action.finding_ids,
        automation_allowed=action.automation_allowed,
        requires_human_approval=action.requires_human_approval,
        audience=(
            "customer"
            if action.action_type == ActionType.CUSTOMER_FOLLOW_UP
            else "manager"
        ),
    )


def _checklist(
    decision: CallDecision,
    promoted_ids: set[str],
    evidence_preferences: set[str],
) -> list[PresentedCheck]:
    findings = [
        item
        for item in (
            decision.triggered_findings + decision.positive_findings
        )
        if item.detection_rule.detector
        == "structured_requirement_assessment"
        and item.visibility != Visibility.INTERNAL
    ]
    rows = []
    for finding in _ordered(findings):
        outcome = _finding_outcome(finding)
        status = {
            FindingOutcome.EFFECTIVE: CheckStatus.DEMONSTRATED,
            FindingOutcome.INCORRECT: CheckStatus.INCORRECT,
            FindingOutcome.MISSED: CheckStatus.NOT_DEMONSTRATED,
            FindingOutcome.OBSERVED_CONCERN: (
                CheckStatus.UNABLE_TO_DETERMINE
            ),
        }[outcome]
        evidence = _selected_evidence(
            finding,
            evidence_preferences,
            limit=1,
        )
        rows.append(
            PresentedCheck(
                requirement_id=finding.applicability.rule_id,
                title=finding.title,
                category=finding.category,
                status=status,
                summary=finding.summary,
                evidence_ids=[item.evidence_id for item in evidence],
                promoted=finding.finding_id in promoted_ids,
            )
        )
    return rows


def _signal_value(bundle: SignalBundle, name: str):
    return next(
        (
            signal.value
            for signal in bundle.signals
            if signal.name == name
        ),
        None,
    )


def _acoustic_context(
    bundle: SignalBundle | None,
) -> AcousticContext:
    if bundle is None:
        return AcousticContext(
            status=AcousticStatus.UNAVAILABLE,
            coverage_label="Audio support unavailable",
            conclusion=(
                "The transcript was evaluated without acoustic support."
            ),
        )
    coverage = next(
        (
            item
            for item in bundle.coverage
            if item.modality == Modality.ACOUSTIC
            and item.source == "audio_features"
        ),
        None,
    )
    if coverage is None:
        coverage = next(
            (
                item
                for item in bundle.coverage
                if item.modality == Modality.ACOUSTIC
                and item.source == "emotion_model"
            ),
            None,
        )
    if coverage is None or coverage.usable_units == 0:
        return AcousticContext(
            status=AcousticStatus.UNAVAILABLE,
            coverage_label="Audio support unavailable",
            conclusion=(
                "No reliable acoustic observations were available for this "
                "call."
            ),
        )

    status = (
        AcousticStatus.AVAILABLE
        if coverage.coverage_ratio >= 0.5
        else AcousticStatus.LIMITED
    )
    observations = [
        AcousticObservation(
            observation_id=episode.episode_id,
            label="Sustained customer vocal strain",
            summary=(
                "Multiple acoustic cues rose together during this interval. "
                "This supports review of the surrounding conversation but "
                "does not establish an agent failure on its own."
            ),
            start_seconds=episode.start_seconds,
            end_seconds=episode.end_seconds,
        )
        for episode in bundle.episodes
        if (
            episode.episode_type == "acoustic.customer_elevation"
            and episode.speaker == Speaker.CUSTOMER
            and episode.reliability.status
            not in (
                ReliabilityStatus.UNUSABLE,
                ReliabilityStatus.UNAVAILABLE,
            )
        )
    ]
    direction = _signal_value(
        bundle,
        "acoustic.dynamics.trajectory_direction",
    )
    unresolved = _signal_value(
        bundle,
        "acoustic.dynamics.unresolved_end_candidate",
    )
    if unresolved:
        conclusion = (
            "Customer vocal strain remained elevated near the end of the "
            "call. Read this with the transcript context."
        )
    elif direction == "decreasing":
        conclusion = (
            "Customer vocal strain eased over the call; the audio supports "
            "a recovery pattern."
        )
    elif observations:
        conclusion = (
            "The audio contains sustained customer-strain intervals that "
            "should be read with the transcript."
        )
    else:
        conclusion = (
            "No sustained vocal-friction pattern crossed the provisional "
            "support gate."
        )
    return AcousticContext(
        status=status,
        coverage_label=(
            f"Audio support on {coverage.usable_units} of "
            f"{coverage.expected_units} segments"
        ),
        conclusion=conclusion,
        observations=observations[:3],
    )


def _finding_evidence_ids(findings: list[Finding]) -> list[str]:
    return list(dict.fromkeys(
        evidence.evidence_id
        for finding in findings
        for evidence in finding.evidence
    ))[:3]


def _question_summary(
    findings: list[Finding],
    fallback: str,
) -> str:
    if not findings:
        return fallback
    return findings[0].summary


def _manager_questions(
    decision: CallDecision,
    _acoustic: AcousticContext,
) -> list[ManagerQuestion]:
    all_findings = (
        decision.triggered_findings + decision.positive_findings
    )
    request_ids = {
        "request.intent_confirmed",
        "loan.purpose_and_stage_confirmed",
    }
    request = [
        item
        for item in all_findings
        if item.applicability.rule_id in request_ids
    ]
    process = [
        item
        for item in all_findings
        if (
            item.detection_rule.detector
            == "structured_requirement_assessment"
            and item.applicability.rule_id not in request_ids
            and item.category
            not in (
                FindingCategory.OUTCOME,
                FindingCategory.AGENT_BEHAVIOR,
            )
        )
    ]
    experience = [
        item
        for item in all_findings
        if (
            item.category == FindingCategory.ESCALATION
            or item.finding_type.startswith("experience.")
            or (
                item.category == FindingCategory.AGENT_BEHAVIOR
                and item.detection_rule.detector
                == "structured_requirement_assessment"
            )
        )
    ]
    outcome = [
        item
        for item in all_findings
        if item.category == FindingCategory.OUTCOME
    ]

    def answer(
        findings: list[Finding],
        *,
        clear_when_empty: bool = False,
    ) -> tuple[ManagerAnswer, str]:
        negative = [
            item
            for item in findings
            if item.polarity.value == "negative"
        ]
        positive = [
            item
            for item in findings
            if item.polarity.value == "positive"
        ]
        if any(
            item.severity.value == "critical"
            for item in negative
        ):
            return ManagerAnswer.NO, "No"
        if negative:
            return ManagerAnswer.PARTLY, "Partly"
        if positive or (
            clear_when_empty
            and decision.decision_status == DecisionStatus.COMPLETE
        ):
            return ManagerAnswer.YES, "Yes"
        return ManagerAnswer.UNCLEAR, "Unable to determine"

    request_answer, request_label = answer(request)
    process_answer, process_label = answer(process)
    experience_answer, experience_label = answer(
        experience,
        clear_when_empty=True,
    )
    outcome_answer, outcome_label = answer(outcome)

    request_negative = [
        item for item in request if item.polarity.value == "negative"
    ]
    request_positive = [
        item for item in request if item.polarity.value == "positive"
    ]
    process_negative = [
        item for item in process if item.polarity.value == "negative"
    ]
    experience_negative = [
        item for item in experience if item.polarity.value == "negative"
    ]
    experience_positive = [
        item for item in experience if item.polarity.value == "positive"
    ]
    outcome_negative = [
        item for item in outcome if item.polarity.value == "negative"
    ]
    outcome_positive = [
        item for item in outcome if item.polarity.value == "positive"
    ]
    return [
        ManagerQuestion(
            question_id="call.request",
            question="Was the customer's request understood?",
            answer=request_answer,
            answer_label=request_label,
            summary=_question_summary(
                request_negative or request_positive,
                "The available evidence did not resolve the request.",
            ),
            evidence_ids=_finding_evidence_ids(request),
        ),
        ManagerQuestion(
            question_id="call.process",
            question="Was the applicable process followed?",
            answer=process_answer,
            answer_label=process_label,
            summary=(
                _question_summary(
                    process_negative,
                    "No applicable process checks were available.",
                )
                if process_negative
                else (
                    f"{len(process)} applicable process checks were "
                    "supported by transcript evidence."
                    if process
                    else "No applicable process checks were available."
                )
            ),
            evidence_ids=_finding_evidence_ids(process),
        ),
        ManagerQuestion(
            question_id="call.experience",
            question="Was the customer experience handled appropriately?",
            answer=experience_answer,
            answer_label=experience_label,
            summary=_question_summary(
                experience_negative or experience_positive,
                (
                    "No customer-experience concern requiring review was "
                    "identified in the transcript."
                ),
            ),
            evidence_ids=_finding_evidence_ids(experience),
        ),
        ManagerQuestion(
            question_id="call.outcome",
            question="Was the outcome clear and complete?",
            answer=outcome_answer,
            answer_label=outcome_label,
            summary=_question_summary(
                outcome_negative or outcome_positive,
                "The available evidence did not resolve the call outcome.",
            ),
            evidence_ids=_finding_evidence_ids(outcome),
        ),
    ]


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
    bundle: SignalBundle | None = None,
) -> CallEvaluationView:
    """Project a decision into the manager-facing call evaluator contract."""
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
    visible_positive = _supervisor_positive_order(
        visible_positive,
        supervisor.positive_finding_ids if supervisor else [],
    )
    positive_source = visible_positive[:MAX_POSITIVE_HIGHLIGHTS]
    positive_ids = {item.finding_id for item in positive_source}
    promoted_ids = primary_ids | positive_ids
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
    acoustic = _acoustic_context(bundle)
    return CallEvaluationView(
        presentation_version=PRESENTATION_VERSION,
        call_id=decision.call_id,
        decision_sha256=decision_sha256(decision),
        state=_state(decision),
        evaluation_status=decision.decision_status,
        attention_required=decision.attention_required,
        headline=headline,
        summary=summary,
        manager_questions=_manager_questions(decision, acoustic),
        checklist=_checklist(
            decision,
            promoted_ids,
            evidence_preferences,
        ),
        acoustic_context=acoustic,
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
