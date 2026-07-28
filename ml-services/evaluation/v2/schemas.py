"""Strict contracts for evaluator v2 signals, findings, and decisions."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class Speaker(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    TRANSCRIPT = "transcript"
    TEXT = "text"
    ACOUSTIC = "acoustic"
    MULTIMODAL = "multimodal"
    POLICY = "policy"
    HUMAN = "human"


class SignalScope(str, Enum):
    SEGMENT = "segment"
    EPISODE = "episode"
    CALL = "call"


class Visibility(str, Enum):
    PRIMARY = "primary"
    EVIDENCE = "evidence"
    DETAILS = "details"
    INTERNAL = "internal"


class ReliabilityStatus(str, Enum):
    USABLE = "usable"
    LIMITED = "limited"
    UNUSABLE = "unusable"
    UNAVAILABLE = "unavailable"


class FindingPolarity(str, Enum):
    NEGATIVE = "negative"
    POSITIVE = "positive"


class FindingCategory(str, Enum):
    REQUIRED_CONTROL = "required_control"
    PROCESS = "process"
    ESCALATION = "escalation"
    OUTCOME = "outcome"
    AGENT_BEHAVIOR = "agent_behavior"
    DATA_QUALITY = "data_quality"


class FindingSeverity(str, Enum):
    INFO = "info"
    REVIEW = "review"
    CRITICAL = "critical"


class ActionType(str, Enum):
    NONE = "none"
    CREATE_REVIEW_CASE = "create_review_case"
    RECOMMEND_COACHING = "recommend_coaching"
    REQUEST_CUSTOMER_FOLLOW_UP = "request_customer_follow_up"
    POLICY_REVIEW = "policy_review"


class ActionExecution(str, Enum):
    NO_ACTION = "no_action"
    AUTOMATIC = "automatic"
    REQUIRES_APPROVAL = "requires_approval"


class SourceProvenance(ContractModel):
    producer: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    source_artifact: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime | None = None
    method: str | None = None


class ReliabilityAssessment(ContractModel):
    status: ReliabilityStatus
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class TranscriptSegment(ContractModel):
    segment_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    seq_id: int | None = Field(default=None, ge=0)
    speaker: Speaker
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("segment end_seconds must not precede start_seconds")
        return self


SignalValue = bool | int | float | str


class SignalRecord(ContractModel):
    signal_id: str = Field(min_length=1)
    name: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    modality: Modality
    scope: SignalScope
    value: SignalValue
    unit: str | None = None
    segment_id: str | None = None
    seq_id: int | None = Field(default=None, ge=0)
    speaker: Speaker | None = None
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    reliability: ReliabilityAssessment
    provenance: SourceProvenance
    visibility: Visibility = Visibility.INTERNAL

    @model_validator(mode="after")
    def validate_signal(self):
        if self.scope == SignalScope.SEGMENT and not self.segment_id:
            raise ValueError("segment signals require the canonical segment_id")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("signal end_seconds must not precede start_seconds")
        if self.visibility == Visibility.PRIMARY:
            raise ValueError(
                "raw signals cannot be primary page content; expose a finding instead"
            )
        return self


class SignalEpisode(ContractModel):
    episode_id: str = Field(min_length=1)
    episode_type: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    modality: Modality
    speaker: Speaker
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    segment_ids: list[str] = Field(min_length=1)
    signal_ids: list[str] = Field(default_factory=list)
    observation: str = Field(min_length=1)
    reliability: ReliabilityAssessment
    provenance: SourceProvenance
    visibility: Visibility = Visibility.INTERNAL

    @model_validator(mode="after")
    def validate_episode(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("episode end_seconds must not precede start_seconds")
        if self.visibility == Visibility.PRIMARY:
            raise ValueError(
                "episodes cannot be primary page content; expose a finding instead"
            )
        return self


class ModalityCoverage(ContractModel):
    modality: Modality
    source: str = Field(min_length=1)
    expected_units: int = Field(ge=0)
    usable_units: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.usable_units > self.expected_units:
            raise ValueError("usable_units cannot exceed expected_units")
        expected_ratio = (
            self.usable_units / self.expected_units
            if self.expected_units
            else 0.0
        )
        if abs(self.coverage_ratio - expected_ratio) > 0.000001:
            raise ValueError("coverage_ratio must match the unit counts")
        return self


class SignalBundle(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    call_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    accent: str | None = None
    duration_seconds: float = Field(ge=0.0)
    transcript_model: str | None = None
    segments: list[TranscriptSegment]
    signals: list[SignalRecord] = Field(default_factory=list)
    episodes: list[SignalEpisode] = Field(default_factory=list)
    coverage: list[ModalityCoverage] = Field(default_factory=list)
    sources: list[SourceProvenance] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self):
        segment_ids = [segment.segment_id for segment in self.segments]
        signal_ids = [signal.signal_id for signal in self.signals]
        episode_ids = [episode.episode_id for episode in self.episodes]

        for values, label in (
            (segment_ids, "segment"),
            (signal_ids, "signal"),
            (episode_ids, "episode"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} id")

        known_segments = set(segment_ids)
        known_signals = set(signal_ids)
        for signal in self.signals:
            if signal.segment_id and signal.segment_id not in known_segments:
                raise ValueError(
                    f"signal {signal.signal_id!r} references unknown segment"
                )
        for episode in self.episodes:
            if not set(episode.segment_ids).issubset(known_segments):
                raise ValueError(
                    f"episode {episode.episode_id!r} references unknown segment"
                )
            if not set(episode.signal_ids).issubset(known_signals):
                raise ValueError(
                    f"episode {episode.episode_id!r} references unknown signal"
                )
        return self


class ThresholdSpec(ContractModel):
    signal_name: str = Field(min_length=1)
    operator: Literal[
        "gt", "gte", "lt", "lte", "eq", "contains", "present"
    ]
    value: SignalValue | None = None
    unit: str | None = None
    window: str | None = None


class DetectionRule(ContractModel):
    rule_id: str = Field(min_length=1)
    rule_version: str = Field(min_length=1)
    detector: str = Field(min_length=1)
    signal_ids: list[str] = Field(default_factory=list)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)


class Applicability(ContractModel):
    applies: bool
    profile_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    selected_branch: str | None = None


class EvidenceRef(ContractModel):
    evidence_id: str = Field(min_length=1)
    modality: Modality
    segment_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    speaker: Speaker | None = None
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    quote: str | None = None
    observation: str | None = None

    @model_validator(mode="after")
    def validate_evidence(self):
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("evidence end_seconds must not precede start_seconds")
        if not any(
            (
                self.segment_ids,
                self.signal_ids,
                self.episode_ids,
                self.start_seconds is not None,
                self.quote,
                self.observation,
            )
        ):
            raise ValueError("evidence requires a source reference or observation")
        return self


class Finding(ContractModel):
    finding_id: str = Field(min_length=1)
    finding_type: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    category: FindingCategory
    polarity: FindingPolarity
    severity: FindingSeverity
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    business_definition: str = Field(min_length=1)
    applicability: Applicability
    detection_rule: DetectionRule
    evidence: list[EvidenceRef] = Field(min_length=1)
    counter_evidence: list[EvidenceRef] = Field(default_factory=list)
    reliability: ReliabilityAssessment
    visibility: Visibility
    display_priority: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_finding(self):
        if not self.applicability.applies:
            raise ValueError("non-applicable rules must not produce findings")
        return self


class DecisionUncertainty(ContractModel):
    code: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    message: str = Field(min_length=1)
    modality: Modality | None = None
    affected_signal_ids: list[str] = Field(default_factory=list)
    affected_finding_ids: list[str] = Field(default_factory=list)
    visibility: Visibility = Visibility.DETAILS

    @model_validator(mode="after")
    def validate_visibility(self):
        if self.visibility == Visibility.PRIMARY:
            raise ValueError("uncertainty is supporting context, not the decision")
        return self


class RecommendedAction(ContractModel):
    action_type: ActionType
    execution: ActionExecution
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    finding_ids: list[str] = Field(default_factory=list)
    automation_allowed: bool = False
    requires_human_approval: bool = False

    @model_validator(mode="after")
    def validate_execution(self):
        if len(self.finding_ids) != len(set(self.finding_ids)):
            raise ValueError("recommended action has duplicate finding ids")
        if self.action_type == ActionType.NONE:
            if self.execution != ActionExecution.NO_ACTION:
                raise ValueError("none action_type requires no_action execution")
            if self.finding_ids:
                raise ValueError("none action_type cannot cite findings")
            if self.automation_allowed or self.requires_human_approval:
                raise ValueError("none action_type cannot require execution")
            return self

        if self.execution == ActionExecution.NO_ACTION:
            raise ValueError("non-none actions require an execution policy")
        if (
            self.execution == ActionExecution.REQUIRES_APPROVAL
            and (
                self.automation_allowed
                or not self.requires_human_approval
            )
        ):
            raise ValueError(
                "approval actions must disable automation and require approval"
            )
        if self.execution == ActionExecution.AUTOMATIC:
            if self.action_type != ActionType.CREATE_REVIEW_CASE:
                raise ValueError(
                    "only review-case creation may initially run automatically"
                )
            if not self.automation_allowed or self.requires_human_approval:
                raise ValueError(
                    "automatic review-case creation must be allowed without approval"
                )
        return self


class PresentationSelection(ContractModel):
    primary_finding_ids: list[str] = Field(default_factory=list)
    positive_finding_ids: list[str] = Field(default_factory=list)
    detail_finding_ids: list[str] = Field(default_factory=list)


class QualificationReason(str, Enum):
    QUALIFIES_CRITICAL = "qualifies_critical"
    QUALIFIES_REVIEW = "qualifies_review"
    QUALIFIES_LIMITED_REVIEW = "qualifies_limited_review"
    EXCLUDED_INFORMATIONAL = "excluded_informational"
    EXCLUDED_INTERNAL = "excluded_internal"
    EXCLUDED_UNRELIABLE = "excluded_unreliable"


class RecoveryEffect(str, Enum):
    NONE = "none"
    CONTEXT_ONLY = "context_only"


class DecisionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class FindingQualification(ContractModel):
    finding_id: str = Field(min_length=1)
    qualifies_for_attention: bool
    reason: QualificationReason
    precedence_group: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_.]+$",
    )

    @model_validator(mode="after")
    def validate_precedence(self):
        if self.qualifies_for_attention and not self.precedence_group:
            raise ValueError(
                "qualifying findings require a precedence group"
            )
        if not self.qualifies_for_attention and self.precedence_group:
            raise ValueError(
                "excluded findings cannot have a precedence group"
            )
        return self


class DecisionPolicyTrace(ContractModel):
    policy_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.-]+$")
    policy_version: str = Field(min_length=1)
    aggregation: Literal["any_qualifying_finding"]
    qualifications: list[FindingQualification] = Field(
        default_factory=list
    )
    controlling_finding_ids: list[str] = Field(default_factory=list)
    recovery_finding_ids: list[str] = Field(default_factory=list)
    recovery_effect: RecoveryEffect = RecoveryEffect.NONE

    @model_validator(mode="after")
    def validate_trace(self):
        qualification_ids = [
            item.finding_id for item in self.qualifications
        ]
        if len(qualification_ids) != len(set(qualification_ids)):
            raise ValueError("decision trace has duplicate finding ids")
        if len(self.controlling_finding_ids) != len(
            set(self.controlling_finding_ids)
        ):
            raise ValueError(
                "decision trace has duplicate controlling findings"
            )
        if len(self.recovery_finding_ids) != len(
            set(self.recovery_finding_ids)
        ):
            raise ValueError(
                "decision trace has duplicate recovery findings"
            )
        qualifying_ids = {
            item.finding_id
            for item in self.qualifications
            if item.qualifies_for_attention
        }
        if not set(self.controlling_finding_ids).issubset(
            qualifying_ids
        ):
            raise ValueError(
                "controlling findings must qualify for attention"
            )
        if qualifying_ids and not self.controlling_finding_ids:
            raise ValueError(
                "attention trace requires a controlling finding"
            )
        if not qualifying_ids and self.controlling_finding_ids:
            raise ValueError(
                "no-attention trace cannot have controlling findings"
            )
        if (
            self.recovery_finding_ids
            and self.recovery_effect == RecoveryEffect.NONE
        ):
            raise ValueError(
                "recovery findings require an explicit recovery effect"
            )
        if (
            not self.recovery_finding_ids
            and self.recovery_effect != RecoveryEffect.NONE
        ):
            raise ValueError(
                "recovery effect requires a recovery finding"
            )
        return self


class CallDecision(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    call_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    signal_bundle_schema_version: Literal["2.0"] = "2.0"
    signal_bundle_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    decision_status: DecisionStatus
    attention_required: bool
    triggered_findings: list[Finding] = Field(default_factory=list)
    positive_findings: list[Finding] = Field(default_factory=list)
    recommended_action: RecommendedAction
    uncertainties: list[DecisionUncertainty] = Field(default_factory=list)
    decision_trace: DecisionPolicyTrace
    presentation: PresentationSelection
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_decision(self):
        all_findings = self.triggered_findings + self.positive_findings
        ids = [finding.finding_id for finding in all_findings]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate finding id")

        if any(
            finding.polarity != FindingPolarity.NEGATIVE
            for finding in self.triggered_findings
        ):
            raise ValueError("triggered_findings must have negative polarity")
        if any(
            finding.polarity != FindingPolarity.POSITIVE
            for finding in self.positive_findings
        ):
            raise ValueError("positive_findings must have positive polarity")

        if self.attention_required and not self.triggered_findings:
            raise ValueError("attention requires at least one triggered finding")
        qualification_ids = {
            item.finding_id
            for item in self.decision_trace.qualifications
        }
        triggered_ids = {
            finding.finding_id
            for finding in self.triggered_findings
        }
        if qualification_ids != triggered_ids:
            raise ValueError(
                "decision trace must classify every triggered finding"
            )
        trace_attention = any(
            item.qualifies_for_attention
            for item in self.decision_trace.qualifications
        )
        if self.attention_required != trace_attention:
            raise ValueError(
                "attention must equal any qualifying finding"
            )
        has_requirement_uncertainty = any(
            uncertainty.code.startswith("requirement.")
            for uncertainty in self.uncertainties
        )
        expected_status = (
            DecisionStatus.PARTIAL
            if has_requirement_uncertainty and all_findings
            else DecisionStatus.INSUFFICIENT_EVIDENCE
            if has_requirement_uncertainty
            else DecisionStatus.COMPLETE
        )
        if self.decision_status != expected_status:
            raise ValueError(
                "decision status does not match finding and "
                "requirement-assessment coverage"
            )
        if self.attention_required:
            if self.recommended_action.action_type == ActionType.NONE:
                raise ValueError("attention requires a recommended action")
            if not self.recommended_action.finding_ids:
                raise ValueError("recommended action must cite a finding")
        elif self.recommended_action.action_type != ActionType.NONE:
            raise ValueError("no-attention decisions must use the none action")

        known_ids = set(ids)
        if not set(self.recommended_action.finding_ids).issubset(known_ids):
            raise ValueError("recommended action references unknown finding")
        if not set(self.recommended_action.finding_ids).issubset(
            self.decision_trace.controlling_finding_ids
        ):
            raise ValueError(
                "recommended action must cite controlling findings"
            )
        positive_ids = {
            finding.finding_id
            for finding in self.positive_findings
        }
        if not set(
            self.decision_trace.recovery_finding_ids
        ).issubset(positive_ids):
            raise ValueError(
                "decision trace references unknown recovery findings"
            )

        for finding_id in self.presentation.primary_finding_ids:
            matching = [
                finding
                for finding in self.triggered_findings
                if finding.finding_id == finding_id
            ]
            if not matching:
                raise ValueError("primary presentation references unknown finding")
            if matching[0].visibility != Visibility.PRIMARY:
                raise ValueError("primary presentation requires primary visibility")
        for finding_id in self.presentation.positive_finding_ids:
            matching = [
                finding
                for finding in self.positive_findings
                if finding.finding_id == finding_id
            ]
            if not matching:
                raise ValueError("positive presentation references unknown finding")
            if matching[0].visibility != Visibility.PRIMARY:
                raise ValueError("positive presentation requires primary visibility")
        if not set(self.presentation.detail_finding_ids).issubset(known_ids):
            raise ValueError("detail presentation references unknown finding")
        return self
