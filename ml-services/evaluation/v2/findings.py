"""Evidence-backed finding derivation for evaluator v2."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .domain_profiles import (
    RequirementLevel,
    ResolvedDomainPlan,
    ResolvedRequirement,
)
from .schemas import (
    Applicability,
    ContractModel,
    DecisionUncertainty,
    DetectionRule,
    EvidenceRef,
    Finding,
    FindingCategory,
    FindingPolarity,
    FindingSeverity,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SignalEpisode,
    SignalRecord,
    SourceProvenance,
    Speaker,
    ThresholdSpec,
    Visibility,
)

FINDINGS_VERSION = "0.1.0"
ACOUSTIC_EPISODE_TYPE = "acoustic.customer_elevation"
RECOVERY_SIGNAL_NAME = "acoustic.dynamics.recovery_candidate"


class RequirementVerdict(str, Enum):
    MET = "met"
    INCORRECT = "incorrect"
    MISSED = "missed"
    UNCERTAIN = "uncertain"


class RequirementAssessment(ContractModel):
    """A grounded semantic verdict supplied to deterministic finding logic."""

    assessment_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    requirement_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_.]+$",
    )
    verdict: RequirementVerdict
    rationale: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)
    counter_evidence: list[EvidenceRef] = Field(default_factory=list)
    reliability: ReliabilityAssessment
    provenance: SourceProvenance


class RequirementAssessmentBatch(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    call_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    assessments: list[RequirementAssessment] = Field(default_factory=list)
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_unique_requirements(self):
        assessment_ids = [
            assessment.assessment_id
            for assessment in self.assessments
        ]
        ids = [
            assessment.requirement_id
            for assessment in self.assessments
        ]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("assessment ids must be unique")
        if len(ids) != len(set(ids)):
            raise ValueError("requirement assessments must be unique")
        return self


class FindingDerivation(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    call_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    findings_version: str = Field(min_length=1)
    triggered_findings: list[Finding] = Field(default_factory=list)
    positive_findings: list[Finding] = Field(default_factory=list)
    uncertainties: list[DecisionUncertainty] = Field(default_factory=list)
    suppressed_duplicates: list[str] = Field(default_factory=list)
    provenance: SourceProvenance

    @model_validator(mode="after")
    def validate_findings(self):
        all_findings = self.triggered_findings + self.positive_findings
        ids = [finding.finding_id for finding in all_findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding derivation contains duplicate ids")
        if any(
            finding.polarity != FindingPolarity.NEGATIVE
            for finding in self.triggered_findings
        ):
            raise ValueError("triggered_findings must be negative")
        if any(
            finding.polarity != FindingPolarity.POSITIVE
            for finding in self.positive_findings
        ):
            raise ValueError("positive_findings must be positive")
        if any(not finding.evidence for finding in all_findings):
            raise ValueError(
                "derived findings require evidence"
            )
        return self


@dataclass(frozen=True)
class TextEscalationRule:
    finding_type: str
    title: str
    business_definition: str
    severity: FindingSeverity
    category: FindingCategory
    patterns: tuple[re.Pattern[str], ...]


def _patterns(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


TEXT_ESCALATION_RULES = (
    TextEscalationRule(
        finding_type="escalation.manager_requested",
        title="Customer requested a manager",
        business_definition=(
            "The customer explicitly asks to speak with a manager or "
            "supervisor."
        ),
        severity=FindingSeverity.CRITICAL,
        category=FindingCategory.ESCALATION,
        patterns=_patterns(
            r"\b(?:speak|talk)\s+(?:with|to)\s+"
            r"(?:(?:a|the|your)\s+)?(?:manager|supervisor)\b",
            r"\b(?:get|put|transfer)\s+me\s+"
            r"(?:to|through\s+to|with)\s+"
            r"(?:(?:a|the|your)\s+)?(?:manager|supervisor)\b",
        ),
    ),
    TextEscalationRule(
        finding_type="escalation.formal_complaint",
        title="Customer raised a formal complaint",
        business_definition=(
            "The customer explicitly states an intention to make a "
            "complaint."
        ),
        severity=FindingSeverity.CRITICAL,
        category=FindingCategory.ESCALATION,
        patterns=_patterns(
            r"\b(?:file|make|submit|raise|lodge)\s+"
            r"(?:a\s+)?(?:formal\s+)?complaint\b",
        ),
    ),
    TextEscalationRule(
        finding_type="escalation.legal_or_regulatory",
        title="Customer threatened external escalation",
        business_definition=(
            "The customer explicitly threatens legal, regulatory, or "
            "ombudsman escalation."
        ),
        severity=FindingSeverity.CRITICAL,
        category=FindingCategory.ESCALATION,
        patterns=_patterns(
            r"\b(?:contact|call|involve|speak\s+to)\s+"
            r"(?:(?:my|a|an)\s+)?(?:lawyer|attorney|ombudsman)\b",
            r"\breport\s+(?:this|you|the\s+bank)\s+to\s+"
            r"(?:the\s+)?(?:regulator|ombudsman|authorities)\b",
        ),
    ),
    TextEscalationRule(
        finding_type="process.repeat_contact_unresolved",
        title="Customer reported repeated contact",
        business_definition=(
            "The customer explicitly reports repeated attempts to resolve "
            "the same issue."
        ),
        severity=FindingSeverity.REVIEW,
        category=FindingCategory.PROCESS,
        patterns=_patterns(
            r"\b(?:called|contacted|spoken\s+to)\b.{0,35}\b"
            r"(?:twice|three\s+times|four\s+times|multiple\s+times|"
            r"several\s+times|\d+\s+times)\b",
            r"\b(?:second|third|fourth)\s+time\s+"
            r"(?:i(?:'m|\s+am)\s+)?(?:calling|contacting)\b",
        ),
    ),
    TextEscalationRule(
        finding_type="escalation.explicit_dissatisfaction",
        title="Customer explicitly expressed dissatisfaction",
        business_definition=(
            "The customer directly describes strong dissatisfaction rather "
            "than the evaluator inferring emotion from wording."
        ),
        severity=FindingSeverity.REVIEW,
        category=FindingCategory.ESCALATION,
        patterns=_patterns(
            r"\b(?:i\s+am|i'm)\s+(?:really\s+|very\s+|extremely\s+)?"
            r"(?:frustrated|angry|upset)\b",
            r"\bthis\s+is\s+(?:completely\s+|absolutely\s+)?"
            r"unacceptable\b",
            r"\b(?:this|it)\s+is\s+(?:really\s+|absolutely\s+)?"
            r"ridiculous\b",
            r"\bthis\s+is\s+(?:honestly\s+)?(?:a\s+)?nightmare\b",
            r"\b(?:was|is|has\s+been)\s+stressing\s+me\s+out\s+"
            r"(?:tremendously|badly)\b",
        ),
    ),
    TextEscalationRule(
        finding_type="experience.accessibility_barrier",
        title="Customer reported an access barrier",
        business_definition=(
            "The customer explicitly states that a physical-access barrier "
            "prevents them from using the offered service path."
        ),
        severity=FindingSeverity.REVIEW,
        category=FindingCategory.AGENT_BEHAVIOR,
        patterns=_patterns(
            r"\b(?:can't|cannot|unable\s+to)\s+(?:go|visit)\s+"
            r"(?:to\s+)?(?:the\s+)?(?:bank|branch)\s+physically\b",
            r"\bi(?:'m|\s+am)\s+(?:currently\s+)?in\s+a\s+wheelchair\b",
        ),
    ),
)


_CATEGORY_MAP = {
    "security": FindingCategory.REQUIRED_CONTROL,
    "transaction_accuracy": FindingCategory.REQUIRED_CONTROL,
    "authorization": FindingCategory.REQUIRED_CONTROL,
    "process": FindingCategory.PROCESS,
    "outcome": FindingCategory.OUTCOME,
    "service": FindingCategory.AGENT_BEHAVIOR,
}

_SEVERITY_RANK = {
    FindingSeverity.INFO: 0,
    FindingSeverity.REVIEW: 1,
    FindingSeverity.CRITICAL: 2,
}

_RELIABILITY_RANK = {
    ReliabilityStatus.USABLE: 0,
    ReliabilityStatus.LIMITED: 1,
    ReliabilityStatus.UNUSABLE: 2,
    ReliabilityStatus.UNAVAILABLE: 3,
}


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.findings",
        producer_version=FINDINGS_VERSION,
        method=(
            "deterministic_requirement_mapping_and_high_precision_"
            "explicit_text_rules"
        ),
    )


def _finding_id(polarity: FindingPolarity, finding_type: str) -> str:
    return (
        f"finding-{polarity.value}-"
        f"{finding_type.replace('.', '-')}"
    )


def _branch_for_requirement(
    plan: ResolvedDomainPlan,
    requirement_id: str,
) -> str | None:
    branches = sorted(
        {
            step.branch_id
            for step in plan.workflow_steps
            if step.requirement_id == requirement_id
        }
    )
    return ",".join(branches) if branches else None


def _requirement_category(
    requirement: ResolvedRequirement,
) -> FindingCategory:
    try:
        return _CATEGORY_MAP[requirement.category]
    except KeyError as exc:
        raise ValueError(
            f"unsupported requirement category: {requirement.category!r}"
        ) from exc


def _level_severity(
    level: RequirementLevel,
    polarity: FindingPolarity,
) -> FindingSeverity:
    if polarity == FindingPolarity.POSITIVE:
        return FindingSeverity.INFO
    if level == RequirementLevel.CRITICAL:
        return FindingSeverity.CRITICAL
    if level == RequirementLevel.REQUIRED:
        return FindingSeverity.REVIEW
    return FindingSeverity.INFO


def _display_priority(
    severity: FindingSeverity,
    polarity: FindingPolarity,
) -> int:
    if polarity == FindingPolarity.POSITIVE:
        return 200
    return {
        FindingSeverity.CRITICAL: 10,
        FindingSeverity.REVIEW: 30,
        FindingSeverity.INFO: 80,
    }[severity]


def _validate_evidence(
    evidence: EvidenceRef,
    bundle: SignalBundle,
) -> None:
    segments = {
        segment.segment_id: segment
        for segment in bundle.segments
    }
    signal_ids = {signal.signal_id for signal in bundle.signals}
    episode_ids = {episode.episode_id for episode in bundle.episodes}
    if not set(evidence.segment_ids).issubset(segments):
        raise ValueError(
            f"evidence {evidence.evidence_id!r} references unknown segments"
        )
    if not set(evidence.signal_ids).issubset(signal_ids):
        raise ValueError(
            f"evidence {evidence.evidence_id!r} references unknown signals"
        )
    if not set(evidence.episode_ids).issubset(episode_ids):
        raise ValueError(
            f"evidence {evidence.evidence_id!r} references unknown episodes"
        )
    if evidence.quote:
        referenced = [
            segments[segment_id]
            for segment_id in evidence.segment_ids
        ]
        if not referenced:
            raise ValueError("quoted evidence requires a segment reference")
        if not any(
            evidence.quote.strip() in segment.text
            for segment in referenced
        ):
            raise ValueError(
                f"evidence {evidence.evidence_id!r} quote is not verbatim"
            )
    if (
        evidence.speaker
        and evidence.segment_ids
        and any(
            segments[segment_id].speaker != evidence.speaker
            for segment_id in evidence.segment_ids
        )
    ):
        raise ValueError(
            f"evidence {evidence.evidence_id!r} speaker does not match"
        )
    if evidence.segment_ids:
        referenced = [
            segments[segment_id]
            for segment_id in evidence.segment_ids
        ]
        expected_start = min(
            segment.start_seconds for segment in referenced
        )
        expected_end = max(
            segment.end_seconds for segment in referenced
        )
        if (
            evidence.start_seconds is not None
            and abs(evidence.start_seconds - expected_start) > 0.001
        ):
            raise ValueError(
                f"evidence {evidence.evidence_id!r} start time does not match"
            )
        if (
            evidence.end_seconds is not None
            and abs(evidence.end_seconds - expected_end) > 0.001
        ):
            raise ValueError(
                f"evidence {evidence.evidence_id!r} end time does not match"
            )


def _validate_assessments(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    batch: RequirementAssessmentBatch | None,
) -> dict[str, RequirementAssessment]:
    if batch is None:
        return {}
    if batch.call_id != bundle.call_id or batch.call_id != plan.call_id:
        raise ValueError(
            "assessment, plan, and bundle call_id values must match"
        )
    if batch.profile_id != plan.profile_id:
        raise ValueError("assessment profile_id does not match the plan")

    known = {
        requirement.requirement_id
        for requirement in plan.requirements
    }
    supplied = {
        assessment.requirement_id
        for assessment in batch.assessments
    }
    if not supplied.issubset(known):
        raise ValueError(
            "assessment batch contains non-applicable requirements"
        )
    for assessment in batch.assessments:
        for evidence in assessment.evidence:
            if not (
                evidence.segment_ids
                or evidence.signal_ids
                or evidence.episode_ids
            ):
                raise ValueError(
                    "requirement assessment evidence must reference "
                    "the signal bundle"
                )
        for evidence in (
            assessment.evidence + assessment.counter_evidence
        ):
            _validate_evidence(evidence, bundle)
    return {
        assessment.requirement_id: assessment
        for assessment in batch.assessments
    }


def _assessment_uncertainty(
    requirement: ResolvedRequirement,
    assessment: RequirementAssessment | None,
) -> DecisionUncertainty:
    if assessment is None:
        return DecisionUncertainty(
            code=f"requirement.not_assessed.{requirement.requirement_id}",
            message=(
                f"{requirement.title} has no grounded requirement "
                "assessment."
            ),
            modality=Modality.TEXT,
            visibility=Visibility.INTERNAL,
        )
    return DecisionUncertainty(
        code=(
            f"requirement.assessment_uncertain."
            f"{requirement.requirement_id}"
        ),
        message=(
            f"{requirement.title} could not be determined from the "
            "available evidence."
        ),
        modality=Modality.TEXT,
        affected_signal_ids=[
            signal_id
            for evidence in (
                assessment.evidence + assessment.counter_evidence
            )
            for signal_id in evidence.signal_ids
        ],
        visibility=Visibility.DETAILS,
    )


def _requirement_finding(
    *,
    plan: ResolvedDomainPlan,
    requirement: ResolvedRequirement,
    assessment: RequirementAssessment,
    polarity: FindingPolarity,
    finding_type: str,
    category: FindingCategory,
) -> Finding:
    severity = _level_severity(requirement.level, polarity)
    title = requirement.title
    summary = assessment.rationale
    return Finding(
        finding_id=_finding_id(polarity, finding_type),
        finding_type=finding_type,
        category=category,
        polarity=polarity,
        severity=severity,
        title=title,
        summary=summary,
        business_definition=requirement.business_definition,
        applicability=Applicability(
            applies=True,
            profile_id=plan.profile_id,
            rule_id=requirement.requirement_id,
            reason=requirement.applicability_reason,
            selected_branch=_branch_for_requirement(
                plan,
                requirement.requirement_id,
            ),
        ),
        detection_rule=DetectionRule(
            rule_id=requirement.requirement_id,
            rule_version=plan.profile_version,
            detector="structured_requirement_assessment",
            thresholds=[
                ThresholdSpec(
                    signal_name="assessment.requirement_verdict",
                    operator="eq",
                    value=assessment.verdict.value,
                )
            ],
        ),
        evidence=assessment.evidence,
        counter_evidence=assessment.counter_evidence,
        reliability=assessment.reliability,
        visibility=Visibility.DETAILS,
        display_priority=_display_priority(severity, polarity),
    )


def _requirement_findings(
    plan: ResolvedDomainPlan,
    assessments: dict[str, RequirementAssessment],
) -> tuple[list[Finding], list[Finding], list[DecisionUncertainty]]:
    negative = []
    positive = []
    uncertainties = []
    for requirement in plan.requirements:
        assessment = assessments.get(requirement.requirement_id)
        if assessment is None:
            uncertainties.append(
                _assessment_uncertainty(requirement, None)
            )
            continue
        if assessment.verdict == RequirementVerdict.UNCERTAIN:
            uncertainties.append(
                _assessment_uncertainty(requirement, assessment)
            )
            continue
        if assessment.reliability.status in (
            ReliabilityStatus.UNUSABLE,
            ReliabilityStatus.UNAVAILABLE,
        ):
            uncertainties.append(
                _assessment_uncertainty(requirement, assessment)
            )
            continue

        category = _requirement_category(requirement)
        if (
            assessment.verdict in (
                RequirementVerdict.INCORRECT,
                RequirementVerdict.MISSED,
            )
            and requirement.failure_finding_type
        ):
            negative.append(
                _requirement_finding(
                    plan=plan,
                    requirement=requirement,
                    assessment=assessment,
                    polarity=FindingPolarity.NEGATIVE,
                    finding_type=requirement.failure_finding_type,
                    category=category,
                )
            )
        elif (
            assessment.verdict == RequirementVerdict.MET
            and requirement.positive_finding_type
        ):
            positive.append(
                _requirement_finding(
                    plan=plan,
                    requirement=requirement,
                    assessment=assessment,
                    polarity=FindingPolarity.POSITIVE,
                    finding_type=requirement.positive_finding_type,
                    category=category,
                )
            )
    return negative, positive, uncertainties


def _transcript_evidence(
    prefix: str,
    segment,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"{prefix}:{segment.segment_id}",
        modality=Modality.TRANSCRIPT,
        segment_ids=[segment.segment_id],
        speaker=segment.speaker,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        quote=segment.text,
    )


def _next_agent_response(
    bundle: SignalBundle,
    after_seconds: float,
    finding_type: str,
) -> EvidenceRef:
    response = next(
        (
            segment
            for segment in sorted(
                bundle.segments,
                key=lambda item: (
                    item.start_seconds,
                    item.segment_index,
                ),
            )
            if segment.speaker == Speaker.AGENT
            and segment.start_seconds >= after_seconds
        ),
        None,
    )
    if response is None:
        return EvidenceRef(
            evidence_id=f"counter:{finding_type}:no-response",
            modality=Modality.TRANSCRIPT,
            observation=(
                "No subsequent agent response was available as "
                "counter-evidence."
            ),
        )
    return _transcript_evidence(
        f"counter:{finding_type}",
        response,
    )


def _episode_distance(
    evidence: EvidenceRef,
    episode: SignalEpisode,
) -> float:
    start = evidence.start_seconds or 0.0
    end = evidence.end_seconds or start
    if start <= episode.end_seconds and end >= episode.start_seconds:
        return 0.0
    if end < episode.start_seconds:
        return episode.start_seconds - end
    return start - episode.end_seconds


def _acoustic_support(
    finding: Finding,
    bundle: SignalBundle,
    window_seconds: float,
) -> Finding:
    if finding.category != FindingCategory.ESCALATION:
        return finding
    episodes = [
        episode
        for episode in bundle.episodes
        if episode.episode_type == ACOUSTIC_EPISODE_TYPE
        and episode.speaker == Speaker.CUSTOMER
        and episode.reliability.status
        not in (
            ReliabilityStatus.UNUSABLE,
            ReliabilityStatus.UNAVAILABLE,
        )
    ]
    matched = [
        episode
        for episode in episodes
        if any(
            _episode_distance(evidence, episode) <= window_seconds
            for evidence in finding.evidence
            if evidence.start_seconds is not None
        )
    ]
    if not matched:
        return finding

    enriched = finding.model_copy(deep=True)
    for episode in matched:
        enriched.evidence.append(
            EvidenceRef(
                evidence_id=(
                    f"evidence:{finding.finding_type}:"
                    f"{episode.episode_id}"
                ),
                modality=Modality.ACOUSTIC,
                signal_ids=episode.signal_ids,
                episode_ids=[episode.episode_id],
                speaker=episode.speaker,
                start_seconds=episode.start_seconds,
                end_seconds=episode.end_seconds,
                observation=(
                    "A provisional persistent acoustic-elevation "
                    "candidate occurs near the explicit text event."
                ),
            )
        )
    enriched.detection_rule.signal_ids = sorted(
        {
            *enriched.detection_rule.signal_ids,
            *(
                signal_id
                for episode in matched
                for signal_id in episode.signal_ids
            ),
        }
    )
    enriched.detection_rule.detector = (
        "explicit_text_with_provisional_acoustic_support"
    )
    enriched.reliability.reasons = sorted(
        {
            *enriched.reliability.reasons,
            "acoustic_elevation_is_provisional_support",
        }
    )
    return Finding.model_validate(enriched.model_dump())


def _explicit_text_findings(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    acoustic_window_seconds: float,
) -> list[Finding]:
    segments = sorted(
        (
            segment
            for segment in bundle.segments
            if segment.speaker == Speaker.CUSTOMER
        ),
        key=lambda item: (item.start_seconds, item.segment_index),
    )
    output = []
    for rule in TEXT_ESCALATION_RULES:
        matched = [
            segment
            for segment in segments
            if any(pattern.search(segment.text) for pattern in rule.patterns)
        ]
        if not matched:
            continue
        last_end = max(segment.end_seconds for segment in matched)
        finding = Finding(
            finding_id=_finding_id(
                FindingPolarity.NEGATIVE,
                rule.finding_type,
            ),
            finding_type=rule.finding_type,
            category=rule.category,
            polarity=FindingPolarity.NEGATIVE,
            severity=rule.severity,
            title=rule.title,
            summary=(
                f"{len(matched)} customer segment"
                f"{'s' if len(matched) != 1 else ''} matched a "
                "high-precision explicit-text rule."
            ),
            business_definition=rule.business_definition,
            applicability=Applicability(
                applies=True,
                profile_id=plan.profile_id,
                rule_id=rule.finding_type,
                reason=(
                    "Explicit escalation and repeat-contact rules apply "
                    "to every supported call."
                ),
            ),
            detection_rule=DetectionRule(
                rule_id=rule.finding_type,
                rule_version=FINDINGS_VERSION,
                detector="high_precision_explicit_text",
                thresholds=[
                    ThresholdSpec(
                        signal_name="transcript.customer_text",
                        operator="contains",
                        value="versioned_explicit_phrase_pattern",
                    )
                ],
            ),
            evidence=[
                _transcript_evidence(
                    f"evidence:{rule.finding_type}",
                    segment,
                )
                for segment in matched
            ],
            counter_evidence=[
                _next_agent_response(
                    bundle,
                    last_end,
                    rule.finding_type,
                )
            ],
            reliability=ReliabilityAssessment(
                status=ReliabilityStatus.USABLE,
                reasons=["high_precision_explicit_phrase_match"],
            ),
            visibility=Visibility.DETAILS,
            display_priority=_display_priority(
                rule.severity,
                FindingPolarity.NEGATIVE,
            ),
        )
        output.append(
            _acoustic_support(
                finding,
                bundle,
                acoustic_window_seconds,
            )
        )
    return output


def _unique_evidence(values: list[EvidenceRef]) -> list[EvidenceRef]:
    output = []
    seen = set()
    for evidence in values:
        key = evidence.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        output.append(evidence)
    return output


def _worst_reliability(
    assessments: list[ReliabilityAssessment],
) -> ReliabilityAssessment:
    status = max(
        (assessment.status for assessment in assessments),
        key=lambda value: _RELIABILITY_RANK[value],
    )
    coverage = [
        assessment.coverage_ratio
        for assessment in assessments
        if assessment.coverage_ratio is not None
    ]
    return ReliabilityAssessment(
        status=status,
        coverage_ratio=min(coverage) if coverage else None,
        reasons=sorted(
            {
                reason
                for assessment in assessments
                for reason in assessment.reasons
            }
        ),
        quality_flags=sorted(
            {
                flag
                for assessment in assessments
                for flag in assessment.quality_flags
            }
        ),
    )


def _merge_findings(
    findings: list[Finding],
) -> tuple[list[Finding], list[str]]:
    grouped: dict[
        tuple[FindingPolarity, str],
        list[Finding],
    ] = {}
    for finding in findings:
        grouped.setdefault(
            (finding.polarity, finding.finding_type),
            [],
        ).append(finding)

    output = []
    suppressed = []
    for group in grouped.values():
        merged = group[0].model_copy(deep=True)
        if len(group) > 1:
            suppressed.extend(
                (
                    f"{finding.finding_id}:"
                    f"{finding.applicability.rule_id}"
                )
                for finding in group[1:]
            )
            requirement_ids = sorted(
                {
                    finding.applicability.rule_id
                    for finding in group
                }
            )
            merged.summary = (
                f"{len(group)} applicable assessments produced the "
                f"same finding type: {', '.join(requirement_ids)}."
            )
            merged.business_definition = " ".join(
                dict.fromkeys(
                    finding.business_definition
                    for finding in group
                )
            )
            merged.applicability.rule_id = ",".join(
                requirement_ids
            )
            merged.detection_rule.rule_id = ",".join(
                requirement_ids
            )
            merged.applicability.selected_branch = ",".join(
                sorted(
                    {
                        branch
                        for finding in group
                        for branch in (
                            finding.applicability.selected_branch or ""
                        ).split(",")
                        if branch
                    }
                )
            ) or None
        merged.evidence = _unique_evidence(
            [
                evidence
                for finding in group
                for evidence in finding.evidence
            ]
        )
        merged.counter_evidence = _unique_evidence(
            [
                evidence
                for finding in group
                for evidence in finding.counter_evidence
            ]
        )
        merged.detection_rule.signal_ids = sorted(
            {
                signal_id
                for finding in group
                for signal_id in finding.detection_rule.signal_ids
            }
        )
        merged.severity = max(
            (finding.severity for finding in group),
            key=lambda value: _SEVERITY_RANK[value],
        )
        merged.display_priority = min(
            finding.display_priority for finding in group
        )
        merged.reliability = _worst_reliability(
            [finding.reliability for finding in group]
        )
        output.append(Finding.model_validate(merged.model_dump()))
    output.sort(
        key=lambda finding: (
            finding.display_priority,
            finding.finding_type,
        )
    )
    return output, suppressed


def _call_signal(
    bundle: SignalBundle,
    name: str,
) -> SignalRecord | None:
    return next(
        (
            signal
            for signal in bundle.signals
            if signal.name == name
        ),
        None,
    )


def _recovery_finding(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    negative: list[Finding],
) -> Finding | None:
    supported = [
        finding
        for finding in negative
        if finding.category == FindingCategory.ESCALATION
        and any(
            evidence.episode_ids
            for evidence in finding.evidence
        )
    ]
    recovery = _call_signal(bundle, RECOVERY_SIGNAL_NAME)
    if (
        not supported
        or recovery is None
        or recovery.value is not True
    ):
        return None

    episode_ids = sorted(
        {
            episode_id
            for finding in supported
            for evidence in finding.evidence
            for episode_id in evidence.episode_ids
        }
    )
    signal_ids = sorted(
        {
            recovery.signal_id,
            *(
                signal_id
                for finding in supported
                for evidence in finding.evidence
                for signal_id in evidence.signal_ids
            ),
        }
    )
    return Finding(
        finding_id=_finding_id(
            FindingPolarity.POSITIVE,
            "escalation.recovery_observed",
        ),
        finding_type="escalation.recovery_observed",
        category=FindingCategory.ESCALATION,
        polarity=FindingPolarity.POSITIVE,
        severity=FindingSeverity.INFO,
        title="Customer delivery recovered after an elevation episode",
        summary=(
            "A provisional acoustic recovery candidate follows an explicit "
            "text event with nearby persistent acoustic elevation."
        ),
        business_definition=(
            "The customer's delivery settles after a supported elevation "
            "episode; this does not by itself prove issue resolution."
        ),
        applicability=Applicability(
            applies=True,
            profile_id=plan.profile_id,
            rule_id="escalation.recovery_observed",
            reason=(
                "Recovery is considered only after an explicit escalation "
                "event has nearby acoustic support."
            ),
        ),
        detection_rule=DetectionRule(
            rule_id="escalation.recovery_observed",
            rule_version=FINDINGS_VERSION,
            detector="explicit_text_acoustic_recovery",
            signal_ids=signal_ids,
            thresholds=[
                ThresholdSpec(
                    signal_name=RECOVERY_SIGNAL_NAME,
                    operator="eq",
                    value=True,
                )
            ],
        ),
        evidence=[
            EvidenceRef(
                evidence_id="evidence:escalation.recovery_observed",
                modality=Modality.ACOUSTIC,
                signal_ids=signal_ids,
                episode_ids=episode_ids,
                observation=(
                    "The provisional acoustic derivation reports recovery "
                    "after the supported episode."
                ),
            )
        ],
        counter_evidence=_unique_evidence(
            [
                evidence
                for finding in supported
                for evidence in finding.evidence
                if evidence.modality == Modality.TRANSCRIPT
            ]
        ),
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.LIMITED,
            reasons=[
                "provisional_acoustic_recovery_rule_not_validated"
            ],
        ),
        visibility=Visibility.INTERNAL,
        display_priority=220,
    )


def derive_findings(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    assessments: RequirementAssessmentBatch | None = None,
    *,
    acoustic_window_seconds: float = 20.0,
) -> FindingDerivation:
    """Derive grounded findings without making an attention decision."""
    if bundle.call_id != plan.call_id:
        raise ValueError("plan and bundle call_id values must match")
    if acoustic_window_seconds < 0:
        raise ValueError("acoustic_window_seconds cannot be negative")

    assessment_by_requirement = _validate_assessments(
        bundle,
        plan,
        assessments,
    )
    negative, positive, uncertainties = _requirement_findings(
        plan,
        assessment_by_requirement,
    )
    negative.extend(
        _explicit_text_findings(
            bundle,
            plan,
            acoustic_window_seconds,
        )
    )
    negative, negative_duplicates = _merge_findings(negative)
    recovery = _recovery_finding(bundle, plan, negative)
    if recovery:
        positive.append(recovery)
    positive, positive_duplicates = _merge_findings(positive)

    known_signal_ids = {
        signal.signal_id
        for signal in bundle.signals
    }
    for finding in negative + positive:
        if not set(
            finding.detection_rule.signal_ids
        ).issubset(known_signal_ids):
            raise ValueError(
                f"finding {finding.finding_id!r} rule references "
                "unknown signals"
            )
        for evidence in finding.evidence + finding.counter_evidence:
            _validate_evidence(evidence, bundle)

    return FindingDerivation(
        call_id=bundle.call_id,
        profile_id=plan.profile_id,
        findings_version=FINDINGS_VERSION,
        triggered_findings=negative,
        positive_findings=positive,
        uncertainties=uncertainties,
        suppressed_duplicates=sorted(
            set(negative_duplicates + positive_duplicates)
        ),
        provenance=_source(),
    )
