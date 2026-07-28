"""Provenance-safe annotations and controlled evaluation cases."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schemas import ContractModel, FindingPolarity, Speaker

EVALUATION_DATASET_ID = "banking-evaluator-v2"
EVALUATION_DATASET_VERSION = "0.1.0"


class EvaluationSourceType(str, Enum):
    EXISTING_CALL = "existing_call"
    CONTROLLED_CHALLENGE = "controlled_challenge"


class AnnotatorType(str, Enum):
    HUMAN = "human"
    MODEL_ASSISTED_RESEARCHER_SEED = (
        "model_assisted_researcher_seed"
    )
    SYNTHETIC_AUTHOR = "synthetic_author"


class AnnotationStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"


class ExpectedAttention(str, Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    UNCERTAIN = "uncertain"


class CounterfactualAxis(str, Enum):
    TRANSCRIPT = "transcript"
    DELIVERY = "delivery"
    OUTCOME = "outcome"
    CONTROL = "control"
    RECOVERY = "recovery"
    DURATION = "duration"


class ExpectedPairEffect(str, Enum):
    ATTENTION_CHANGES = "attention_changes"
    REASONS_CHANGE = "reasons_change"
    SUPPORT_CHANGES = "support_changes"
    RECOVERY_CHANGES = "recovery_changes"
    NO_DECISION_CHANGE = "no_decision_change"


class AcousticCondition(str, Enum):
    NOT_MODELED = "not_modeled"
    CALM = "calm"
    PERSISTENT_ELEVATION = "persistent_elevation"
    RECOVERED_ELEVATION = "recovered_elevation"
    UNRESOLVED_ELEVATION = "unresolved_elevation"


class AnnotationProvenance(ContractModel):
    annotator_id: str = Field(min_length=1)
    annotator_type: AnnotatorType
    status: AnnotationStatus
    method: str = Field(min_length=1)
    requires_human_review: bool
    ground_truth_basis: str | None = None

    @model_validator(mode="after")
    def validate_claim_strength(self):
        if (
            self.annotator_type
            == AnnotatorType.MODEL_ASSISTED_RESEARCHER_SEED
        ):
            if self.status != AnnotationStatus.DRAFT:
                raise ValueError(
                    "model-assisted seed annotations must remain draft"
                )
            if not self.requires_human_review:
                raise ValueError(
                    "model-assisted seed annotations require human review"
                )
            if self.ground_truth_basis:
                raise ValueError(
                    "model-assisted seed annotations are not ground truth"
                )
        if self.status == AnnotationStatus.ADJUDICATED:
            if (
                self.annotator_type == AnnotatorType.HUMAN
                and self.requires_human_review
            ):
                raise ValueError(
                    "human-adjudicated annotations cannot remain pending"
                )
            if (
                self.annotator_type == AnnotatorType.SYNTHETIC_AUTHOR
                and not self.ground_truth_basis
            ):
                raise ValueError(
                    "synthetic known answers require a controlled basis"
                )
        return self


class AnnotationEvidence(ContractModel):
    evidence_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    segment_id: str = Field(min_length=1)
    seq_id: int = Field(ge=0)
    speaker: Speaker
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    quote: str = Field(min_length=1)
    purpose: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "annotation evidence end must not precede start"
            )
        return self


class ExpectedFindingLabel(ContractModel):
    finding_type: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_.]+$",
    )
    polarity: FindingPolarity
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class CallAnnotation(ContractModel):
    annotation_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    call_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    source_type: EvaluationSourceType
    expected_attention: ExpectedAttention
    expected_findings: list[ExpectedFindingLabel] = Field(
        default_factory=list
    )
    evidence: list[AnnotationEvidence] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    provenance: AnnotationProvenance

    @model_validator(mode="after")
    def validate_annotation(self):
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("annotation evidence ids must be unique")
        known_evidence = set(evidence_ids)
        finding_types = [
            finding.finding_type for finding in self.expected_findings
        ]
        if len(finding_types) != len(set(finding_types)):
            raise ValueError("expected finding types must be unique")
        for finding in self.expected_findings:
            if not set(finding.evidence_ids).issubset(known_evidence):
                raise ValueError(
                    "expected finding references unknown evidence"
                )
        negative = [
            item
            for item in self.expected_findings
            if item.polarity == FindingPolarity.NEGATIVE
        ]
        if (
            self.expected_attention == ExpectedAttention.REQUIRED
            and not negative
        ):
            raise ValueError(
                "required attention needs an expected negative finding"
            )
        if (
            self.expected_attention == ExpectedAttention.NOT_REQUIRED
            and negative
        ):
            raise ValueError(
                "not-required attention cannot include negative findings"
            )
        if (
            self.expected_attention == ExpectedAttention.UNCERTAIN
            and not self.uncertainty_notes
        ):
            raise ValueError(
                "uncertain attention requires an uncertainty note"
            )
        if (
            self.source_type == EvaluationSourceType.EXISTING_CALL
            and self.provenance.annotator_type
            == AnnotatorType.SYNTHETIC_AUTHOR
        ):
            raise ValueError(
                "existing calls cannot use synthetic-author provenance"
            )
        return self


class ExistingCallRecord(ContractModel):
    call_id: str = Field(min_length=1)
    transcript_path: str = Field(min_length=1)
    transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation: CallAnnotation

    @model_validator(mode="after")
    def validate_call_id(self):
        if self.annotation.call_id != self.call_id:
            raise ValueError("existing annotation call id mismatch")
        if (
            self.annotation.source_type
            != EvaluationSourceType.EXISTING_CALL
        ):
            raise ValueError(
                "existing record requires existing-call annotation"
            )
        return self


class ChallengeSegment(ContractModel):
    segment_id: str = Field(min_length=1)
    seq_id: int = Field(ge=0)
    speaker: Speaker
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "challenge segment end must not precede start"
            )
        return self


class ChallengeCall(ContractModel):
    call_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    intent_ids: list[str] = Field(min_length=1)
    facts: dict[str, bool]
    acoustic_condition: AcousticCondition
    segments: list[ChallengeSegment] = Field(min_length=1)
    annotation: CallAnnotation

    @model_validator(mode="after")
    def validate_challenge(self):
        if self.annotation.call_id != self.call_id:
            raise ValueError("challenge annotation call id mismatch")
        if (
            self.annotation.source_type
            != EvaluationSourceType.CONTROLLED_CHALLENGE
        ):
            raise ValueError(
                "challenge requires controlled-challenge annotation"
            )
        segment_ids = [item.segment_id for item in self.segments]
        seq_ids = [item.seq_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("challenge segment ids must be unique")
        if len(seq_ids) != len(set(seq_ids)):
            raise ValueError("challenge seq ids must be unique")
        return self


class CounterfactualPair(ContractModel):
    pair_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
    )
    axis: CounterfactualAxis
    baseline_call_id: str = Field(min_length=1)
    variant_call_id: str = Field(min_length=1)
    controlled_change: str = Field(min_length=1)
    invariant_dimensions: list[
        Literal[
            "intent_and_facts",
            "transcript",
            "acoustic_condition",
            "attention",
        ]
    ] = Field(min_length=1)
    expected_effect: ExpectedPairEffect

    @model_validator(mode="after")
    def validate_distinct_calls(self):
        if self.baseline_call_id == self.variant_call_id:
            raise ValueError(
                "counterfactual pair requires two distinct calls"
            )
        if len(self.invariant_dimensions) != len(
            set(self.invariant_dimensions)
        ):
            raise ValueError(
                "counterfactual invariant dimensions must be unique"
            )
        return self


class EvaluationDataset(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["banking-evaluator-v2"] = (
        EVALUATION_DATASET_ID
    )
    dataset_version: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    existing_calls: list[ExistingCallRecord] = Field(
        default_factory=list
    )
    challenge_calls: list[ChallengeCall] = Field(
        default_factory=list
    )
    counterfactual_pairs: list[CounterfactualPair] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_dataset_references(self):
        all_calls = [
            item.call_id
            for item in self.existing_calls
        ] + [
            item.call_id
            for item in self.challenge_calls
        ]
        if len(all_calls) != len(set(all_calls)):
            raise ValueError("evaluation call ids must be unique")
        pair_ids = [item.pair_id for item in self.counterfactual_pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("counterfactual pair ids must be unique")
        challenge_ids = {
            item.call_id for item in self.challenge_calls
        }
        for pair in self.counterfactual_pairs:
            if (
                pair.baseline_call_id not in challenge_ids
                or pair.variant_call_id not in challenge_ids
            ):
                raise ValueError(
                    "counterfactual pair references unknown challenge"
                )
        represented = {
            item.axis for item in self.counterfactual_pairs
        }
        if self.counterfactual_pairs and represented != set(
            CounterfactualAxis
        ):
            missing = sorted(
                item.value
                for item in set(CounterfactualAxis) - represented
            )
            raise ValueError(
                f"counterfactual axes are incomplete: {missing}"
            )
        return self


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transcript_segments(path: Path) -> tuple[str, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["call_id"], {
        item["id"]: item for item in payload["sentences"]
    }


def _validate_evidence(
    annotation: CallAnnotation,
    segments: dict[str, dict],
) -> None:
    for evidence in annotation.evidence:
        try:
            segment = segments[evidence.segment_id]
        except KeyError as exc:
            raise ValueError(
                f"{annotation.call_id}: unknown evidence segment "
                f"{evidence.segment_id!r}"
            ) from exc
        expected = {
            "seq_id": evidence.seq_id,
            "speaker": evidence.speaker.value.upper(),
            "start": evidence.start_seconds,
            "end": evidence.end_seconds,
            "text": evidence.quote,
        }
        actual = {
            "seq_id": segment["seq_id"],
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }
        if actual != expected:
            raise ValueError(
                f"{annotation.call_id}: evidence "
                f"{evidence.evidence_id!r} does not exactly match "
                "the transcript"
            )


def _challenge_segments(call: ChallengeCall) -> dict[str, dict]:
    return {
        item.segment_id: {
            "seq_id": item.seq_id,
            "speaker": item.speaker.value.upper(),
            "start": item.start_seconds,
            "end": item.end_seconds,
            "text": item.text,
        }
        for item in call.segments
    }


def _transcript_content(call: ChallengeCall) -> list[tuple[str, str]]:
    return [
        (item.speaker.value, item.text)
        for item in call.segments
    ]


def validate_evaluation_dataset(
    dataset: EvaluationDataset,
    repo_root: Path,
) -> None:
    """Validate source fidelity and declared pair invariants."""
    for record in dataset.existing_calls:
        path = repo_root / record.transcript_path
        if file_sha256(path) != record.transcript_sha256:
            raise ValueError(
                f"{record.call_id}: transcript sha256 changed"
            )
        call_id, segments = _transcript_segments(path)
        if call_id != record.call_id:
            raise ValueError(
                f"{record.call_id}: transcript call id mismatch"
            )
        _validate_evidence(record.annotation, segments)

    challenges = {
        item.call_id: item for item in dataset.challenge_calls
    }
    for call in challenges.values():
        _validate_evidence(
            call.annotation,
            _challenge_segments(call),
        )

    for pair in dataset.counterfactual_pairs:
        baseline = challenges[pair.baseline_call_id]
        variant = challenges[pair.variant_call_id]
        invariants = set(pair.invariant_dimensions)
        if "intent_and_facts" in invariants and (
            baseline.intent_ids != variant.intent_ids
            or baseline.facts != variant.facts
        ):
            raise ValueError(
                f"{pair.pair_id}: intent_and_facts invariant failed"
            )
        if (
            "transcript" in invariants
            and _transcript_content(baseline)
            != _transcript_content(variant)
        ):
            raise ValueError(
                f"{pair.pair_id}: transcript invariant failed"
            )
        if (
            "acoustic_condition" in invariants
            and baseline.acoustic_condition
            != variant.acoustic_condition
        ):
            raise ValueError(
                f"{pair.pair_id}: acoustic invariant failed"
            )
        if (
            "attention" in invariants
            and baseline.annotation.expected_attention
            != variant.annotation.expected_attention
        ):
            raise ValueError(
                f"{pair.pair_id}: attention invariant failed"
            )


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(
        path.read_text(encoding="utf-8")
    )
