"""Bounded transcript assessment for applicable domain requirements."""
from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable

from pydantic import Field, ValidationError, model_validator

from .domain_profiles import ResolvedDomainPlan
from .findings import (
    RequirementAssessment,
    RequirementAssessmentBatch,
    RequirementVerdict,
)
from .schemas import (
    ContractModel,
    EvidenceRef,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SourceProvenance,
)

SEMANTIC_ASSESSOR_VERSION = "0.2.0"
_TIERS = ("qa-primary", "qa-fallback", "qa-safety")
SYSTEM_PROMPT = """You evaluate a customer-service transcript against a fixed
list of applicable business requirements.

Rules:
- Assess every supplied requirement exactly once.
- Use only the supplied requirement IDs and transcript segment IDs.
- MET requires direct transcript evidence that demonstrates the requirement.
- INCORRECT means the agent directly gave wrong information or performed the
  applicable behavior incorrectly. It requires direct transcript evidence of
  what the agent said or did.
- MISSED means the complete transcript lacks the required behavior or directly
  omits it. Cite the closest relevant context segment and explain what is
  absent; never invent a quote.
- UNCERTAIN means the transcript does not support a defensible MET or MISSED
  verdict.
- evidence_segment_ids must contain 1 to 3 segments that support the verdict.
- If a requirement has multiple elements, cite every segment needed to
  demonstrate those elements.
- Never use a generic acknowledgment such as "yes", "okay", or "for sure" as
  the sole evidence. Select the substantive statement it refers to as well.
- counter_evidence_segment_ids is optional and contains only genuine evidence
  that weighs against the verdict.
- An offered next step still counts when the customer declines it.
- When an agent explains current ineligibility and gives a concrete route or
  timeframe to try again, treat that as a follow-up path and outcome
  explanation when the requirement permits it.
- Judge only the stated evidence expectations. Do not demand documents, rates,
  or actions outside them.
- Do not infer policies beyond the supplied business definition.
- Do not score the call and do not recommend actions.

Return JSON only:
{"assessments":[{"requirement_id":"...","verdict":"met|incorrect|missed|uncertain",
"rationale":"call-specific explanation","evidence_segment_ids":["..."],
"counter_evidence_segment_ids":[]}]}"""


class SemanticVerdict(ContractModel):
    requirement_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9_.]+$",
    )
    verdict: RequirementVerdict
    rationale: str = Field(min_length=1, max_length=500)
    evidence_segment_ids: list[str] = Field(min_length=1, max_length=3)
    counter_evidence_segment_ids: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_segment_lists(self):
        if len(self.evidence_segment_ids) != len(
            set(self.evidence_segment_ids)
        ):
            raise ValueError("evidence segment IDs must be unique")
        if len(self.counter_evidence_segment_ids) != len(
            set(self.counter_evidence_segment_ids)
        ):
            raise ValueError("counter-evidence segment IDs must be unique")
        return self


class SemanticVerdictBatch(ContractModel):
    assessments: list[SemanticVerdict] = Field(min_length=1)


def _normalize_duplicates(
    payload: SemanticVerdictBatch,
) -> SemanticVerdictBatch:
    by_requirement: dict[str, SemanticVerdict] = {}
    for item in payload.assessments:
        existing = by_requirement.get(item.requirement_id)
        if existing is None:
            by_requirement[item.requirement_id] = item
            continue
        if existing.verdict != item.verdict:
            raise ValueError(
                "conflicting duplicate verdicts for "
                f"{item.requirement_id}"
            )

        evidence = list(dict.fromkeys(
            existing.evidence_segment_ids + item.evidence_segment_ids
        ))
        counter = list(dict.fromkeys(
            existing.counter_evidence_segment_ids
            + item.counter_evidence_segment_ids
        ))
        by_requirement[item.requirement_id] = existing.model_copy(
            update={
                "evidence_segment_ids": evidence[:3],
                "counter_evidence_segment_ids": counter[:3],
            }
        )
    return SemanticVerdictBatch(
        assessments=list(by_requirement.values())
    )


Completion = Callable[[str, str, str], tuple[str, str]]


def _complete(system: str, user: str, tier: str) -> tuple[str, str]:
    from router import chat_json_routed

    return chat_json_routed(
        system,
        user,
        temperature=0.0,
        max_tokens=7000,
        return_meta=True,
        tier=tier,
    )


def _prompt(bundle: SignalBundle, plan: ResolvedDomainPlan) -> str:
    def encoded(value) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    lines = [
        f"CALL\t{encoded(bundle.call_id)}",
        f"PROFILE\t{encoded(plan.profile_id)}",
        f"INTENTS\t{encoded(plan.selected_intent_ids)}",
        "REQUIREMENTS",
        "requirement_id\tlevel\ttitle\tdefinition\texpectations",
    ]
    lines.extend(
        "\t".join([
            item.requirement_id,
            item.level.value,
            encoded(item.title),
            encoded(item.business_definition),
            encoded(item.evidence_expectations),
        ])
        for item in plan.requirements
    )
    lines.extend([
        "TRANSCRIPT",
        "segment_id\tspeaker\tstart_seconds\ttext",
    ])
    lines.extend(
        "\t".join([
            segment.segment_id,
            segment.speaker.value,
            f"{segment.start_seconds:.2f}",
            encoded(segment.text),
        ])
        for segment in bundle.segments
    )
    return "\n".join(lines)


def _validate_response(
    payload: SemanticVerdictBatch,
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
) -> None:
    expected = {
        requirement.requirement_id
        for requirement in plan.requirements
    }
    supplied_ids = [
        assessment.requirement_id
        for assessment in payload.assessments
    ]
    supplied = set(supplied_ids)
    duplicates = sorted(
        item
        for item, count in Counter(supplied_ids).items()
        if count > 1
    )
    if supplied != expected or duplicates:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"semantic verdict coverage mismatch; missing={missing}, "
            f"extra={extra}, duplicates={duplicates}"
        )

    known_segments = {
        segment.segment_id
        for segment in bundle.segments
    }
    for assessment in payload.assessments:
        referenced = set(
            assessment.evidence_segment_ids
            + assessment.counter_evidence_segment_ids
        )
        unknown = sorted(referenced - known_segments)
        if unknown:
            raise ValueError(
                f"{assessment.requirement_id} references unknown "
                f"segments: {unknown}"
            )


def _evidence(
    *,
    bundle: SignalBundle,
    requirement_id: str,
    segment_ids: list[str],
    purpose: str,
) -> list[EvidenceRef]:
    by_id = {
        segment.segment_id: segment
        for segment in bundle.segments
    }
    if not segment_ids:
        return []
    segments = [by_id[segment_id] for segment_id in segment_ids]
    speakers = {segment.speaker for segment in segments}
    observation = "\n".join(
        f"{segment.speaker.value.upper()}: {segment.text}"
        for segment in segments
    )
    return [
        EvidenceRef(
            evidence_id=f"semantic:{requirement_id}:{purpose}",
            modality=Modality.TRANSCRIPT,
            segment_ids=segment_ids,
            speaker=segments[0].speaker if len(speakers) == 1 else None,
            start_seconds=min(
                segment.start_seconds
                for segment in segments
            ),
            end_seconds=max(
                segment.end_seconds
                for segment in segments
            ),
            observation=observation,
        )
    ]


def _materialize(
    *,
    payload: SemanticVerdictBatch,
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    served_model: str,
) -> RequirementAssessmentBatch:
    provenance = SourceProvenance(
        producer="evaluator_v2.semantic_assessment",
        producer_version=SEMANTIC_ASSESSOR_VERSION,
        method=(
            "bounded_requirement_verdict_with_validated_segment_references:"
            f"{served_model}+deterministic_policy_guards"
        ),
    )
    reliability = ReliabilityAssessment(
        status=ReliabilityStatus.LIMITED,
        coverage_ratio=1.0,
        reasons=[
            "llm_semantic_assessment",
            "not_human_adjudicated",
        ],
    )
    return RequirementAssessmentBatch(
        call_id=bundle.call_id,
        profile_id=plan.profile_id,
        assessments=[
            RequirementAssessment(
                assessment_id=f"assessment:{item.requirement_id}",
                requirement_id=item.requirement_id,
                verdict=item.verdict,
                rationale=item.rationale,
                evidence=_evidence(
                    bundle=bundle,
                    requirement_id=item.requirement_id,
                    segment_ids=item.evidence_segment_ids,
                    purpose="support",
                ),
                counter_evidence=_evidence(
                    bundle=bundle,
                    requirement_id=item.requirement_id,
                    segment_ids=item.counter_evidence_segment_ids,
                    purpose="counter",
                ),
                reliability=reliability,
                provenance=provenance,
            )
            for item in payload.assessments
        ],
        provenance=provenance,
    )


def assess_requirements(
    bundle: SignalBundle,
    plan: ResolvedDomainPlan,
    *,
    completion: Completion | None = None,
    max_attempts: int = 4,
) -> RequirementAssessmentBatch:
    if not bundle.segments:
        raise ValueError("semantic assessment requires transcript segments")
    completion = completion or _complete
    base_prompt = _prompt(bundle, plan)
    prompt = base_prompt
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        tier = _TIERS[min(attempt, len(_TIERS) - 1)]
        started = time.perf_counter()
        try:
            raw, served_model = completion(SYSTEM_PROMPT, prompt, tier)
        except Exception as exc:  # noqa: BLE001 - provider clients vary
            last_error = exc
            continue
        try:
            payload = SemanticVerdictBatch.model_validate_json(raw)
            payload = _normalize_duplicates(payload)
            _validate_response(payload, bundle, plan)
            result = _materialize(
                payload=payload,
                bundle=bundle,
                plan=plan,
                served_model=served_model,
            )
            elapsed = time.perf_counter() - started
            print(
                f"  [v2 semantic] {len(result.assessments)} requirements "
                f"in {elapsed:.1f}s via {served_model}"
            )
            return result
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            prompt = (
                f"{base_prompt}\n\nYour previous response was invalid: "
                f"{str(exc)[:600]}. Return corrected JSON only."
            )

    raise RuntimeError(
        f"semantic assessment failed after {max_attempts} attempts: "
        f"{last_error}"
    )
