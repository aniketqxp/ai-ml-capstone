"""Reproducible modality ablations for evaluator-v2 signal value."""
from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .decisions import build_call_decision
from .domain_profiles import (
    DomainProfile,
    ResolvedDomainPlan,
    ResolvedRequirement,
)
from .evaluation_data import (
    AnnotationEvidence,
    CallAnnotation,
    ChallengeCall,
    CounterfactualAxis,
    EvaluationDataset,
    ExpectedAttention,
    ExpectedPairEffect,
)
from .findings import (
    RequirementAssessment,
    RequirementAssessmentBatch,
    RequirementVerdict,
    derive_findings,
)
from .schemas import (
    CallDecision,
    ContractModel,
    EvidenceRef,
    Finding,
    FindingPolarity,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SignalEpisode,
    SignalRecord,
    SignalScope,
    SourceProvenance,
    Speaker,
    Visibility,
)

SIGNAL_VALIDATION_VERSION = "0.1.0"
RECOVERY_FINDING_TYPE = "escalation.recovery_observed"
ELEVATION_EPISODE_TYPE = "acoustic.customer_elevation"
RECOVERY_SIGNAL_NAME = "acoustic.dynamics.recovery_candidate"


class AblationVariant(str, Enum):
    TEXT_ONLY = "text_only"
    AUDIO_ONLY = "audio_only"
    MULTIMODAL = "multimodal"


class ValidationPopulation(str, Enum):
    EXISTING_CALLS = "existing_calls"
    CONTROLLED_CHALLENGES = "controlled_challenges"


class ClaimLevel(str, Enum):
    DETECTOR_ONLY = "detector_only"
    KNOWN_ANSWER_POLICY_VALIDATION = (
        "known_answer_policy_validation"
    )


class AgreementStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ThresholdFitStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"


class AcousticPolicy(str, Enum):
    SUPPORT_ONLY = "support_only"
    ELIGIBLE_FOR_PRIMARY_DECISION = "eligible_for_primary_decision"


class CallValidationResult(ContractModel):
    call_id: str = Field(min_length=1)
    population: ValidationPopulation
    claim_level: ClaimLevel
    variant: AblationVariant
    expected_attention: ExpectedAttention
    predicted_attention: bool
    decision_status: str = Field(min_length=1)
    expected_negative_finding_types: list[str] = Field(
        default_factory=list
    )
    predicted_negative_finding_types: list[str] = Field(
        default_factory=list
    )
    expected_positive_finding_types: list[str] = Field(
        default_factory=list
    )
    predicted_positive_finding_types: list[str] = Field(
        default_factory=list
    )
    attention_match: bool
    false_positive: bool
    false_negative: bool
    matched_reason_count: int = Field(ge=0)
    expected_reason_count: int = Field(ge=0)
    localized_reason_count: int = Field(ge=0)
    acoustically_supported_finding_types: list[str] = Field(
        default_factory=list
    )
    recovery_observed: bool
    candidate_episode_count: int = Field(ge=0)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.matched_reason_count > self.expected_reason_count:
            raise ValueError(
                "matched reasons cannot exceed expected reasons"
            )
        if self.localized_reason_count > self.matched_reason_count:
            raise ValueError(
                "localized reasons cannot exceed matched reasons"
            )
        if self.false_positive and self.false_negative:
            raise ValueError(
                "a result cannot be both false positive and false negative"
            )
        return self


class VariantMetrics(ContractModel):
    population: ValidationPopulation
    claim_level: ClaimLevel
    variant: AblationVariant
    call_count: int = Field(ge=0)
    expected_attention_count: int = Field(ge=0)
    predicted_attention_count: int = Field(ge=0)
    attention_match_count: int = Field(ge=0)
    attention_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    true_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    true_negative_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    false_positive_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    expected_reason_count: int = Field(ge=0)
    matched_reason_count: int = Field(ge=0)
    reason_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    localized_reason_count: int = Field(ge=0)
    evidence_localization_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    acoustically_supported_finding_count: int = Field(ge=0)
    recovery_observed_count: int = Field(ge=0)
    candidate_episode_count: int = Field(ge=0)


class PairValidationResult(ContractModel):
    pair_id: str = Field(min_length=1)
    axis: CounterfactualAxis
    variant: AblationVariant
    expected_effect: ExpectedPairEffect
    passed: bool
    baseline_attention: bool
    variant_attention: bool
    baseline_acoustic_support_count: int = Field(ge=0)
    variant_acoustic_support_count: int = Field(ge=0)
    baseline_recovery_observed: bool
    variant_recovery_observed: bool


class PairMetrics(ContractModel):
    variant: AblationVariant
    pair_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    pass_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    passed_by_axis: dict[str, int]
    total_by_axis: dict[str, int]


class AgreementAssessment(ContractModel):
    status: AgreementStatus
    human_adjudicated_call_count: int = Field(ge=0)
    human_attention_agreement: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    researcher_seed_call_count: int = Field(ge=0)
    researcher_seed_attention_agreement: dict[str, float | None]
    limitation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_availability(self):
        if (
            self.status == AgreementStatus.UNAVAILABLE
            and self.human_attention_agreement is not None
        ):
            raise ValueError(
                "unavailable human agreement cannot have a score"
            )
        if (
            self.status == AgreementStatus.AVAILABLE
            and self.human_attention_agreement is None
        ):
            raise ValueError(
                "available human agreement requires a score"
            )
        return self


class ThresholdAssessment(ContractModel):
    status: ThresholdFitStatus
    fitted_threshold_count: Literal[0] = 0
    reason: str = Field(min_length=1)


class AcousticValueAssessment(ContractModel):
    selected_policy: AcousticPolicy
    attention_changes_from_text_only: int = Field(ge=0)
    challenge_support_pair_passed: bool
    challenge_recovery_pair_passed: bool
    real_call_supported_finding_count: int = Field(ge=0)
    real_call_recovery_count: int = Field(ge=0)
    rationale: list[str] = Field(min_length=1)


class SignalValidationReport(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    validation_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    source_revisions: dict[str, str]
    call_results: list[CallValidationResult]
    variant_metrics: list[VariantMetrics]
    pair_results: list[PairValidationResult]
    pair_metrics: list[PairMetrics]
    human_agreement: AgreementAssessment
    threshold_assessment: ThresholdAssessment
    acoustic_value: AcousticValueAssessment
    conclusions: list[str] = Field(min_length=1)


def _source() -> SourceProvenance:
    return SourceProvenance(
        producer="evaluator_v2.signal_validation",
        producer_version=SIGNAL_VALIDATION_VERSION,
        method="versioned_modality_ablation",
    )


def challenge_transcript(call: ChallengeCall) -> dict:
    return {
        "call_id": call.call_id,
        "domain": "banking",
        "accent": "controlled",
        "model": "authored-counterfactual-v1",
        "sentences": [
            {
                "id": segment.segment_id,
                "seq_id": segment.seq_id,
                "speaker": segment.speaker.value.upper(),
                "start": segment.start_seconds,
                "end": segment.end_seconds,
                "text": segment.text,
            }
            for segment in call.segments
        ],
    }


def make_audio_only_bundle(bundle: SignalBundle) -> SignalBundle:
    """Retain timing and acoustic references while hiding all words."""
    output = bundle.model_copy(deep=True)
    for segment in output.segments:
        segment.text = ""
    output.coverage = [
        coverage
        for coverage in output.coverage
        if coverage.modality != Modality.TRANSCRIPT
    ]
    return output


def _segment_evidence(
    evidence: AnnotationEvidence,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence.evidence_id,
        modality=Modality.TRANSCRIPT,
        segment_ids=[evidence.segment_id],
        speaker=evidence.speaker,
        start_seconds=evidence.start_seconds,
        end_seconds=evidence.end_seconds,
        quote=evidence.quote,
    )


def _fallback_segment(
    bundle: SignalBundle,
    requirement: ResolvedRequirement,
) -> EvidenceRef:
    preferred_index = {
        "interaction.agent_identification": 0,
        "request.intent_confirmed": 1,
        "outcome.summary_and_next_steps": -1,
        "security.transaction_identity_verified": 2,
        "transfer.accounts_confirmed": 4,
        "transfer.amount_confirmed": 4,
        "transfer.terms_explained": 4,
        "transfer.authorization_obtained": 5,
        "transfer.completion_confirmed": -1,
        "transfer.one_time_timing_confirmed": 4,
    }.get(requirement.requirement_id, 0)
    segment = bundle.segments[preferred_index]
    return EvidenceRef(
        evidence_id=(
            f"controlled:{requirement.requirement_id}:evidence"
        ),
        modality=Modality.TRANSCRIPT,
        segment_ids=[segment.segment_id],
        speaker=segment.speaker,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        quote=segment.text,
    )


def controlled_assessments(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    annotation: CallAnnotation,
) -> RequirementAssessmentBatch:
    """Materialize known-answer semantic assessments for policy tests."""
    findings = {
        finding.finding_type: finding
        for finding in annotation.expected_findings
    }
    evidence = {
        item.evidence_id: item for item in annotation.evidence
    }
    assessments = []
    for requirement in plan.requirements:
        failure = (
            findings.get(requirement.failure_finding_type)
            if requirement.failure_finding_type
            else None
        )
        positive = (
            findings.get(requirement.positive_finding_type)
            if requirement.positive_finding_type
            else None
        )
        verdict = (
            RequirementVerdict.MISSED
            if failure
            else RequirementVerdict.MET
        )
        selected = failure or positive
        selected_evidence = (
            [
                _segment_evidence(evidence[evidence_id])
                for evidence_id in selected.evidence_ids
            ]
            if selected and selected.evidence_ids
            else [_fallback_segment(bundle, requirement)]
        )
        assessments.append(
            RequirementAssessment(
                assessment_id=(
                    f"controlled:{bundle.call_id}:"
                    f"{requirement.requirement_id}"
                ),
                requirement_id=requirement.requirement_id,
                verdict=verdict,
                rationale=(
                    selected.rationale
                    if selected
                    else (
                        "The controlled clean-call template satisfies "
                        "this applicable requirement."
                    )
                ),
                evidence=selected_evidence,
                counter_evidence=[
                    EvidenceRef(
                        evidence_id=(
                            f"controlled:{requirement.requirement_id}:"
                            "counter"
                        ),
                        modality=Modality.POLICY,
                        observation=(
                            "No contradictory exchange is authored "
                            "elsewhere in this controlled call."
                        ),
                    )
                ],
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.USABLE,
                    reasons=["controlled_known_answer"],
                ),
                provenance=_source(),
            )
        )
    return RequirementAssessmentBatch(
        call_id=bundle.call_id,
        profile_id=plan.profile_id,
        assessments=assessments,
        provenance=_source(),
    )


def _controlled_episode_segment(
    call: ChallengeCall,
) -> str | None:
    negative_ids = {
        evidence_id
        for finding in call.annotation.expected_findings
        if finding.polarity == FindingPolarity.NEGATIVE
        and finding.finding_type.startswith("escalation.")
        for evidence_id in finding.evidence_ids
    }
    evidence_by_id = {
        evidence.evidence_id: evidence
        for evidence in call.annotation.evidence
    }
    for evidence_id in negative_ids:
        return evidence_by_id[evidence_id].segment_id
    return None


def apply_controlled_acoustics(
    bundle: SignalBundle,
    call: ChallengeCall,
) -> SignalBundle:
    output = bundle.model_copy(deep=True)
    condition = call.acoustic_condition.value
    if condition in {"not_modeled", "calm"}:
        return output
    segment_id = _controlled_episode_segment(call)
    if segment_id is None:
        return output
    segment = next(
        item
        for item in output.segments
        if item.segment_id == segment_id
    )
    signal = SignalRecord(
        signal_id=f"{segment_id}:controlled-elevation",
        name="acoustic.escalation.score",
        modality=Modality.ACOUSTIC,
        scope=SignalScope.SEGMENT,
        value=0.9,
        unit="ratio",
        segment_id=segment.segment_id,
        seq_id=segment.seq_id,
        speaker=Speaker.CUSTOMER,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.USABLE,
            reasons=["controlled_acoustic_condition"],
        ),
        provenance=_source(),
        visibility=Visibility.INTERNAL,
    )
    output.signals.append(signal)
    output.episodes.append(
        SignalEpisode(
            episode_id=f"{call.call_id}:controlled-elevation",
            episode_type=ELEVATION_EPISODE_TYPE,
            modality=Modality.ACOUSTIC,
            speaker=Speaker.CUSTOMER,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            segment_ids=[segment.segment_id],
            signal_ids=[signal.signal_id],
            observation=(
                "Authored persistent customer-elevation condition."
            ),
            reliability=ReliabilityAssessment(
                status=ReliabilityStatus.USABLE,
                reasons=["controlled_acoustic_condition"],
            ),
            provenance=_source(),
            visibility=Visibility.INTERNAL,
        )
    )
    if condition in {
        "recovered_elevation",
        "unresolved_elevation",
    }:
        output.signals.append(
            SignalRecord(
                signal_id=f"{call.call_id}:controlled-recovery",
                name=RECOVERY_SIGNAL_NAME,
                modality=Modality.ACOUSTIC,
                scope=SignalScope.CALL,
                value=condition == "recovered_elevation",
                reliability=ReliabilityAssessment(
                    status=ReliabilityStatus.USABLE,
                    reasons=["controlled_acoustic_condition"],
                ),
                provenance=_source(),
                visibility=Visibility.INTERNAL,
            )
        )
    output.sources.append(_source())
    return output


def _finding_types(
    findings: list[Finding],
) -> list[str]:
    return sorted(
        {finding.finding_type for finding in findings}
    )


def _acoustic_finding_types(
    decision: CallDecision,
) -> list[str]:
    return sorted(
        {
            finding.finding_type
            for finding in (
                decision.triggered_findings
                + decision.positive_findings
            )
            if any(
                evidence.modality == Modality.ACOUSTIC
                for evidence in finding.evidence
            )
        }
    )


def _predicted_segment_ids(finding: Finding) -> set[str]:
    return {
        segment_id
        for evidence in finding.evidence
        if evidence.modality == Modality.TRANSCRIPT
        for segment_id in evidence.segment_ids
    }


def _expected_evidence_by_finding(
    annotation: CallAnnotation,
) -> dict[str, set[str]]:
    evidence = {
        item.evidence_id: item.segment_id
        for item in annotation.evidence
    }
    return {
        finding.finding_type: {
            evidence[evidence_id]
            for evidence_id in finding.evidence_ids
        }
        for finding in annotation.expected_findings
        if finding.polarity == FindingPolarity.NEGATIVE
    }


def evaluate_call(
    *,
    annotation: CallAnnotation,
    decision: CallDecision,
    bundle: SignalBundle,
    population: ValidationPopulation,
    claim_level: ClaimLevel,
    variant: AblationVariant,
    limitations: list[str] | None = None,
) -> CallValidationResult:
    expected_negative = sorted(
        finding.finding_type
        for finding in annotation.expected_findings
        if finding.polarity == FindingPolarity.NEGATIVE
    )
    expected_positive = sorted(
        finding.finding_type
        for finding in annotation.expected_findings
        if finding.polarity == FindingPolarity.POSITIVE
    )
    predicted_negative = _finding_types(
        decision.triggered_findings
    )
    predicted_positive = _finding_types(
        decision.positive_findings
    )
    expected_attention = (
        annotation.expected_attention == ExpectedAttention.REQUIRED
    )
    match = decision.attention_required == expected_attention
    predicted_by_type = {
        finding.finding_type: finding
        for finding in decision.triggered_findings
    }
    expected_segments = _expected_evidence_by_finding(annotation)
    matched = set(expected_negative).intersection(
        predicted_negative
    )
    localized = sum(
        bool(
            _predicted_segment_ids(predicted_by_type[finding_type])
            .intersection(expected_segments[finding_type])
        )
        for finding_type in matched
    )
    return CallValidationResult(
        call_id=annotation.call_id,
        population=population,
        claim_level=claim_level,
        variant=variant,
        expected_attention=annotation.expected_attention,
        predicted_attention=decision.attention_required,
        decision_status=decision.decision_status.value,
        expected_negative_finding_types=expected_negative,
        predicted_negative_finding_types=predicted_negative,
        expected_positive_finding_types=expected_positive,
        predicted_positive_finding_types=predicted_positive,
        attention_match=match,
        false_positive=(
            not expected_attention and decision.attention_required
        ),
        false_negative=(
            expected_attention and not decision.attention_required
        ),
        matched_reason_count=len(matched),
        expected_reason_count=len(expected_negative),
        localized_reason_count=localized,
        acoustically_supported_finding_types=(
            _acoustic_finding_types(decision)
        ),
        recovery_observed=(
            RECOVERY_FINDING_TYPE in predicted_positive
        ),
        candidate_episode_count=sum(
            episode.episode_type == ELEVATION_EPISODE_TYPE
            for episode in bundle.episodes
        ),
        limitations=limitations or [],
    )


def evaluate_bundle(
    *,
    annotation: CallAnnotation,
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    assessments: RequirementAssessmentBatch | None,
    population: ValidationPopulation,
    claim_level: ClaimLevel,
    variant: AblationVariant,
    limitations: list[str] | None = None,
) -> tuple[CallValidationResult, CallDecision]:
    derivation = derive_findings(bundle, plan, assessments)
    decision = build_call_decision(bundle, derivation)
    return (
        evaluate_call(
            annotation=annotation,
            decision=decision,
            bundle=bundle,
            population=population,
            claim_level=claim_level,
            variant=variant,
            limitations=limitations,
        ),
        decision,
    )


def aggregate_variant_metrics(
    results: list[CallValidationResult],
) -> list[VariantMetrics]:
    groups: dict[
        tuple[
            ValidationPopulation,
            ClaimLevel,
            AblationVariant,
        ],
        list[CallValidationResult],
    ] = {}
    for result in results:
        key = (
            result.population,
            result.claim_level,
            result.variant,
        )
        groups.setdefault(key, []).append(result)

    output = []
    for (population, claim_level, variant), rows in groups.items():
        expected_positive = sum(
            row.expected_attention == ExpectedAttention.REQUIRED
            for row in rows
        )
        expected_negative = len(rows) - expected_positive
        true_positive = sum(
            row.predicted_attention
            and row.expected_attention == ExpectedAttention.REQUIRED
            for row in rows
        )
        true_negative = sum(
            not row.predicted_attention
            and row.expected_attention
            == ExpectedAttention.NOT_REQUIRED
            for row in rows
        )
        false_positive = sum(row.false_positive for row in rows)
        false_negative = sum(row.false_negative for row in rows)
        expected_reasons = sum(
            row.expected_reason_count for row in rows
        )
        matched_reasons = sum(
            row.matched_reason_count for row in rows
        )
        localized_reasons = sum(
            row.localized_reason_count for row in rows
        )
        output.append(
            VariantMetrics(
                population=population,
                claim_level=claim_level,
                variant=variant,
                call_count=len(rows),
                expected_attention_count=expected_positive,
                predicted_attention_count=sum(
                    row.predicted_attention for row in rows
                ),
                attention_match_count=sum(
                    row.attention_match for row in rows
                ),
                attention_accuracy=(
                    sum(row.attention_match for row in rows)
                    / len(rows)
                    if rows
                    else None
                ),
                true_positive_count=true_positive,
                false_negative_count=false_negative,
                sensitivity=(
                    true_positive / expected_positive
                    if expected_positive
                    else None
                ),
                true_negative_count=true_negative,
                false_positive_count=false_positive,
                false_positive_rate=(
                    false_positive / expected_negative
                    if expected_negative
                    else None
                ),
                expected_reason_count=expected_reasons,
                matched_reason_count=matched_reasons,
                reason_recall=(
                    matched_reasons / expected_reasons
                    if expected_reasons
                    else None
                ),
                localized_reason_count=localized_reasons,
                evidence_localization_rate=(
                    localized_reasons / matched_reasons
                    if matched_reasons
                    else None
                ),
                acoustically_supported_finding_count=sum(
                    len(row.acoustically_supported_finding_types)
                    for row in rows
                ),
                recovery_observed_count=sum(
                    row.recovery_observed for row in rows
                ),
                candidate_episode_count=sum(
                    row.candidate_episode_count for row in rows
                ),
            )
        )
    return sorted(
        output,
        key=lambda item: (
            item.population.value,
            item.variant.value,
        ),
    )


def evaluate_pairs(
    dataset: EvaluationDataset,
    challenge_results: list[CallValidationResult],
) -> list[PairValidationResult]:
    by_key = {
        (row.call_id, row.variant): row
        for row in challenge_results
    }
    output = []
    for pair in dataset.counterfactual_pairs:
        for variant in AblationVariant:
            baseline = by_key[(pair.baseline_call_id, variant)]
            changed = by_key[(pair.variant_call_id, variant)]
            baseline_support = len(
                baseline.acoustically_supported_finding_types
            )
            changed_support = len(
                changed.acoustically_supported_finding_types
            )
            attention_matches_labels = (
                baseline.attention_match and changed.attention_match
            )
            if (
                pair.expected_effect
                == ExpectedPairEffect.ATTENTION_CHANGES
            ):
                passed = (
                    attention_matches_labels
                    and baseline.predicted_attention
                    != changed.predicted_attention
                )
            elif (
                pair.expected_effect
                == ExpectedPairEffect.SUPPORT_CHANGES
            ):
                passed = (
                    attention_matches_labels
                    and baseline_support != changed_support
                )
            elif (
                pair.expected_effect
                == ExpectedPairEffect.RECOVERY_CHANGES
            ):
                passed = (
                    attention_matches_labels
                    and baseline.recovery_observed
                    != changed.recovery_observed
                )
            elif (
                pair.expected_effect
                == ExpectedPairEffect.NO_DECISION_CHANGE
            ):
                passed = (
                    attention_matches_labels
                    and baseline.predicted_attention
                    == changed.predicted_attention
                )
            else:
                passed = (
                    attention_matches_labels
                    and baseline.predicted_negative_finding_types
                    != changed.predicted_negative_finding_types
                )
            output.append(
                PairValidationResult(
                    pair_id=pair.pair_id,
                    axis=pair.axis,
                    variant=variant,
                    expected_effect=pair.expected_effect,
                    passed=passed,
                    baseline_attention=baseline.predicted_attention,
                    variant_attention=changed.predicted_attention,
                    baseline_acoustic_support_count=baseline_support,
                    variant_acoustic_support_count=changed_support,
                    baseline_recovery_observed=(
                        baseline.recovery_observed
                    ),
                    variant_recovery_observed=(
                        changed.recovery_observed
                    ),
                )
            )
    return output


def aggregate_pair_metrics(
    results: list[PairValidationResult],
) -> list[PairMetrics]:
    output = []
    for variant in AblationVariant:
        rows = [
            row for row in results if row.variant == variant
        ]
        total = Counter(row.axis.value for row in rows)
        passed = Counter(
            row.axis.value for row in rows if row.passed
        )
        output.append(
            PairMetrics(
                variant=variant,
                pair_count=len(rows),
                passed_count=sum(row.passed for row in rows),
                pass_rate=(
                    sum(row.passed for row in rows) / len(rows)
                    if rows
                    else None
                ),
                passed_by_axis=dict(sorted(passed.items())),
                total_by_axis=dict(sorted(total.items())),
            )
        )
    return output


def agreement_assessment(
    dataset: EvaluationDataset,
    existing_results: list[CallValidationResult],
) -> AgreementAssessment:
    human = [
        record
        for record in dataset.existing_calls
        if (
            record.annotation.provenance.annotator_type.value
            == "human"
            and record.annotation.provenance.status.value
            == "adjudicated"
        )
    ]
    seed_by_variant = {}
    for variant in AblationVariant:
        rows = [
            row
            for row in existing_results
            if row.variant == variant
        ]
        seed_by_variant[variant.value] = (
            sum(row.attention_match for row in rows) / len(rows)
            if rows
            else None
        )
    human_ids = {record.call_id for record in human}
    human_rows = [
        row
        for row in existing_results
        if (
            row.variant == AblationVariant.MULTIMODAL
            and row.call_id in human_ids
        )
    ]
    human_score = (
        sum(row.attention_match for row in human_rows)
        / len(human_rows)
        if human_rows
        else None
    )
    return AgreementAssessment(
        status=(
            AgreementStatus.AVAILABLE
            if human
            else AgreementStatus.UNAVAILABLE
        ),
        human_adjudicated_call_count=len(human),
        human_attention_agreement=human_score,
        researcher_seed_call_count=len(dataset.existing_calls),
        researcher_seed_attention_agreement=seed_by_variant,
        limitation=(
            "No existing-call annotation has human-adjudicated "
            "provenance, so human agreement cannot be calculated. "
            "Researcher-seed agreement is reported separately and is "
            "not an accuracy claim."
        ),
    )


def acoustic_value_assessment(
    existing_results: list[CallValidationResult],
    pair_results: list[PairValidationResult],
) -> AcousticValueAssessment:
    existing_text = {
        row.call_id: row
        for row in existing_results
        if row.variant == AblationVariant.TEXT_ONLY
    }
    existing_multi = {
        row.call_id: row
        for row in existing_results
        if row.variant == AblationVariant.MULTIMODAL
    }
    attention_changes = sum(
        existing_text[call_id].predicted_attention
        != existing_multi[call_id].predicted_attention
        for call_id in existing_text
    )
    support_pair = next(
        row
        for row in pair_results
        if (
            row.axis == CounterfactualAxis.DELIVERY
            and row.variant == AblationVariant.MULTIMODAL
        )
    )
    recovery_pair = next(
        row
        for row in pair_results
        if (
            row.axis == CounterfactualAxis.RECOVERY
            and row.variant == AblationVariant.MULTIMODAL
        )
    )
    supported = sum(
        len(row.acoustically_supported_finding_types)
        for row in existing_multi.values()
    )
    recovery = sum(
        row.recovery_observed for row in existing_multi.values()
    )
    return AcousticValueAssessment(
        selected_policy=AcousticPolicy.SUPPORT_ONLY,
        attention_changes_from_text_only=attention_changes,
        challenge_support_pair_passed=support_pair.passed,
        challenge_recovery_pair_passed=recovery_pair.passed,
        real_call_supported_finding_count=supported,
        real_call_recovery_count=recovery,
        rationale=[
            (
                "Audio-only has no grounded route to an attention "
                "finding under the current policy."
            ),
            (
                "Controlled multimodal cases verify acoustic "
                "corroboration and recovery context without allowing "
                "either to erase or create the deterministic decision."
            ),
            (
                "The real-call batch has no human-adjudicated labels "
                "showing that acoustic candidates improve attention "
                "accuracy."
            ),
        ],
    )


def build_report(
    *,
    dataset: EvaluationDataset,
    profile: DomainProfile,
    source_revisions: dict[str, str],
    call_results: list[CallValidationResult],
) -> SignalValidationReport:
    existing_results = [
        row
        for row in call_results
        if row.population == ValidationPopulation.EXISTING_CALLS
    ]
    challenge_results = [
        row
        for row in call_results
        if (
            row.population
            == ValidationPopulation.CONTROLLED_CHALLENGES
        )
    ]
    pair_results = evaluate_pairs(dataset, challenge_results)
    acoustic_value = acoustic_value_assessment(
        existing_results,
        pair_results,
    )
    return SignalValidationReport(
        validation_version=SIGNAL_VALIDATION_VERSION,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        profile_id=profile.profile_id,
        source_revisions=source_revisions,
        call_results=call_results,
        variant_metrics=aggregate_variant_metrics(call_results),
        pair_results=pair_results,
        pair_metrics=aggregate_pair_metrics(pair_results),
        human_agreement=agreement_assessment(
            dataset,
            existing_results,
        ),
        threshold_assessment=ThresholdAssessment(
            status=ThresholdFitStatus.NOT_ATTEMPTED,
            reason=(
                "The existing calls have no human-adjudicated labels "
                "and the controlled cases are authored, so fitting "
                "thresholds would overstate the available evidence."
            ),
        ),
        acoustic_value=acoustic_value,
        conclusions=[
            (
                "Controlled text-only decisions test deterministic "
                "policy behavior, not semantic-model accuracy."
            ),
            (
                "Existing-call detector runs remain incomplete until "
                "grounded requirement assessments are available."
            ),
            (
                "Acoustic output remains supporting evidence and "
                "recovery context, not an independent attention trigger."
            ),
            (
                "Human agreement and threshold fitting remain blocked "
                "by the absence of adjudicated real-call labels."
            ),
        ],
    )
