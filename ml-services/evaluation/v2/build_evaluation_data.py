"""Build the checked evaluation dataset from reviewed source references."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluation_data import (
    EVALUATION_DATASET_ID,
    EVALUATION_DATASET_VERSION,
    AcousticCondition,
    AnnotationEvidence,
    AnnotationProvenance,
    AnnotationStatus,
    AnnotatorType,
    CallAnnotation,
    ChallengeCall,
    ChallengeSegment,
    CounterfactualAxis,
    CounterfactualPair,
    EvaluationDataset,
    EvaluationSourceType,
    ExistingCallRecord,
    ExpectedAttention,
    ExpectedFindingLabel,
    ExpectedPairEffect,
    file_sha256,
    validate_evaluation_dataset,
)
from .schemas import FindingPolarity, Speaker

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRANSCRIPT_DIR = REPO_ROOT / "data" / "sentence_segments" / "banking"
DEFAULT_OUTPUT = HERE / "research" / "evaluation_data_0_1.json"
DEFAULT_SUMMARY = (
    HERE / "research" / "evaluation_data_summary_0_1.json"
)
PROFILE_ID = "banking-demo-v1"


def _source_segment(call_id: str, seq_id: int) -> dict[str, Any]:
    path = TRANSCRIPT_DIR / f"{call_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(
        item
        for item in payload["sentences"]
        if item["seq_id"] == seq_id
    )


def _source_evidence(
    call_id: str,
    seq_id: int,
    purpose: str,
) -> AnnotationEvidence:
    item = _source_segment(call_id, seq_id)
    return AnnotationEvidence(
        evidence_id=f"evidence:{call_id}:{seq_id}",
        segment_id=item["id"],
        seq_id=item["seq_id"],
        speaker=Speaker(item["speaker"].lower()),
        start_seconds=item["start"],
        end_seconds=item["end"],
        quote=item["text"],
        purpose=purpose,
    )


def _seed_provenance() -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator_id="codex-research-seed",
        annotator_type=(
            AnnotatorType.MODEL_ASSISTED_RESEARCHER_SEED
        ),
        status=AnnotationStatus.DRAFT,
        method=(
            "profile-guided transcript review with exact segment "
            "citations; no claim of human ground truth"
        ),
        requires_human_review=True,
    )


def _finding(
    finding_type: str,
    polarity: FindingPolarity,
    evidence: list[AnnotationEvidence],
    rationale: str,
) -> ExpectedFindingLabel:
    return ExpectedFindingLabel(
        finding_type=finding_type,
        polarity=polarity,
        evidence_ids=[item.evidence_id for item in evidence],
        rationale=rationale,
    )


def _existing_record(
    call_id: str,
    attention: ExpectedAttention,
    findings: list[
        tuple[
            str,
            FindingPolarity,
            list[tuple[int, str]],
            str,
        ]
    ],
    uncertainty_notes: list[str] | None = None,
) -> ExistingCallRecord:
    evidence_by_key: dict[tuple[int, str], AnnotationEvidence] = {}
    labels = []
    for finding_type, polarity, evidence_specs, rationale in findings:
        finding_evidence = []
        for seq_id, purpose in evidence_specs:
            key = (seq_id, purpose)
            evidence_by_key.setdefault(
                key,
                _source_evidence(call_id, seq_id, purpose),
            )
            finding_evidence.append(evidence_by_key[key])
        labels.append(
            _finding(
                finding_type,
                polarity,
                finding_evidence,
                rationale,
            )
        )

    transcript_path = (
        Path("data")
        / "sentence_segments"
        / "banking"
        / f"{call_id}.json"
    )
    absolute_path = REPO_ROOT / transcript_path
    annotation = CallAnnotation(
        annotation_id=f"annotation:{call_id}:seed",
        call_id=call_id,
        profile_id=PROFILE_ID,
        source_type=EvaluationSourceType.EXISTING_CALL,
        expected_attention=attention,
        expected_findings=labels,
        evidence=list(evidence_by_key.values()),
        uncertainty_notes=uncertainty_notes or [],
        provenance=_seed_provenance(),
    )
    return ExistingCallRecord(
        call_id=call_id,
        transcript_path=transcript_path.as_posix(),
        transcript_sha256=file_sha256(absolute_path),
        annotation=annotation,
    )


def build_existing_records() -> list[ExistingCallRecord]:
    """Create draft labels whose citations remain tied to source files."""
    return [
        _existing_record(
            "en_CA_Banking_1586889",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(11, "Identity verification is explicitly framed.")],
                    "The agent requests verification before account action.",
                ),
                (
                    "control.recurring_schedule_confirmed",
                    FindingPolarity.POSITIVE,
                    [(28, "Amount, accounts, frequency, and start date are recapped.")],
                    "The recurring transfer schedule is explicit.",
                ),
                (
                    "outcome.transfer_confirmed",
                    FindingPolarity.POSITIVE,
                    [(104, "The completed setup is summarized.")],
                    "The call contains a clear transaction recap.",
                ),
            ],
        ),
        _existing_record(
            "en_CA_Banking_1588683",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(6, "The agent begins a multi-item verification exchange.")],
                    "The exchange occurs before the account is opened.",
                ),
                (
                    "control.recurring_schedule_confirmed",
                    FindingPolarity.POSITIVE,
                    [(69, "The recurring dates and setup are stated.")],
                    "The recurring schedule is explicit.",
                ),
                (
                    "outcome.transfer_confirmed",
                    FindingPolarity.POSITIVE,
                    [(83, "The agent verifies acceptance and scheduled dates.")],
                    "The customer receives confirmation of the setup.",
                ),
            ],
            [
                (
                    "The project profile treats the multi-item exchange as "
                    "verification; it does not assess whether requesting a "
                    "full card security code is appropriate operational "
                    "policy."
                )
            ],
        ),
        _existing_record(
            "en_CA_Banking_1590992",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(39, "Security questions begin before execution.")],
                    "The agent performs a security-question exchange.",
                ),
                (
                    "control.transfer_authorization_obtained",
                    FindingPolarity.POSITIVE,
                    [(158, "The customer accepts the immediate transfer.")],
                    "Approval follows the transfer discussion.",
                ),
                (
                    "outcome.transfer_confirmed",
                    FindingPolarity.POSITIVE,
                    [(177, "The customer confirms receipt of the notification.")],
                    "The notification provides transaction confirmation.",
                ),
            ],
        ),
        _existing_record(
            "en_CA_Banking_1592237",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(12, "The agent frames date of birth as security verification.")],
                    "Identity is checked before account opening.",
                ),
                (
                    "control.account_opening_consent_obtained",
                    FindingPolarity.POSITIVE,
                    [(80, "The customer agrees to open the checking account.")],
                    "The customer gives explicit opening consent.",
                ),
                (
                    "outcome.account_opening_confirmed",
                    FindingPolarity.POSITIVE,
                    [(89, "The agent confirms setup and next steps.")],
                    "The outcome and email follow-up are stated.",
                ),
            ],
            [
                (
                    "The transcript also introduces a savings account late "
                    "in the call; human review should decide whether it is "
                    "in scope for the selected checking-account intent."
                )
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1584540",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "process.loan_request_confirmed",
                    FindingPolarity.POSITIVE,
                    [(62, "The agent summarizes purpose and amount.")],
                    "The equipment-financing purpose is established.",
                ),
                (
                    "process.loan_eligibility_explained",
                    FindingPolarity.POSITIVE,
                    [(95, "The agent qualifies the likely eligibility outcome.")],
                    "The explanation avoids promising approval.",
                ),
                (
                    "outcome.loan_next_step_confirmed",
                    FindingPolarity.POSITIVE,
                    [(100, "The agent offers information and a later reapplication path.")],
                    "A route to improve and return is provided.",
                ),
            ],
            [
                (
                    "The current applicability fixture classifies this as "
                    "a general inquiry, although the agent collects "
                    "financial details and gives an eligibility opinion. "
                    "A human must adjudicate the inquiry-versus-application "
                    "boundary."
                )
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1586157",
            ExpectedAttention.REQUIRED,
            [
                (
                    "process.recurring_change_path_unclear",
                    FindingPolarity.NEGATIVE,
                    [(126, "Only expiration extension is explained after setup.")],
                    "The call does not explain how to change, pause, or cancel the recurring instruction.",
                ),
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(8, "Date of birth is requested before account action.")],
                    "The agent completes a verification exchange.",
                ),
                (
                    "control.recurring_schedule_confirmed",
                    FindingPolarity.POSITIVE,
                    [(121, "Amount, source, destination, frequency, and date are recapped.")],
                    "The recurring schedule is explicit.",
                ),
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1586678",
            ExpectedAttention.REQUIRED,
            [
                (
                    "control.identity_verification_missing",
                    FindingPolarity.NEGATIVE,
                    [(9, "The first account request asks for full account numbers.")],
                    "Account identifiers are collected without a customer-verification exchange.",
                ),
                (
                    "control.transfer_authorization_missing",
                    FindingPolarity.NEGATIVE,
                    [(46, "The agent selects a schedule without later obtaining explicit approval.")],
                    "No explicit authorization follows the material transfer details.",
                ),
                (
                    "process.recurring_change_path_explained",
                    FindingPolarity.POSITIVE,
                    [(51, "The agent explains available change and pause operations.")],
                    "The customer receives a usable modification path.",
                ),
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1586893",
            ExpectedAttention.NOT_REQUIRED,
            [
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(10, "Identity verification is explicitly requested.")],
                    "The verification exchange precedes account action.",
                ),
                (
                    "process.transfer_terms_explained",
                    FindingPolarity.POSITIVE,
                    [(60, "Fees and modification paths are stated.")],
                    "Material recurring-transfer terms are explained.",
                ),
                (
                    "outcome.transfer_confirmed",
                    FindingPolarity.POSITIVE,
                    [(102, "The agent confirms the schedule is active.")],
                    "The transfer setup has a clear completion statement.",
                ),
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1587139",
            ExpectedAttention.REQUIRED,
            [
                (
                    "escalation.explicit_dissatisfaction",
                    FindingPolarity.NEGATIVE,
                    [(60, "The customer directly calls the situation ridiculous.")],
                    "The explicit wording meets the review-level text rule.",
                ),
                (
                    "control.identity_verification_completed",
                    FindingPolarity.POSITIVE,
                    [(9, "Date of birth is requested before the transaction.")],
                    "The call contains a verification exchange.",
                ),
                (
                    "control.transfer_authorization_obtained",
                    FindingPolarity.POSITIVE,
                    [(79, "The customer accepts the expedited transfer.")],
                    "The customer gives explicit approval after timing is explained.",
                ),
            ],
            [
                (
                    "The existing acoustic result indicates later recovery, "
                    "but this seed does not treat that model-derived signal "
                    "as human-validated ground truth."
                )
            ],
        ),
        _existing_record(
            "en_US_General_Banking_1587700",
            ExpectedAttention.REQUIRED,
            [
                (
                    "control.transfer_authorization_missing",
                    FindingPolarity.NEGATIVE,
                    [(64, "The agent announces execution without a fresh approval after the details.")],
                    "The transcript lacks explicit authorization after amount, accounts, and timing are established.",
                ),
                (
                    "control.transfer_accounts_confirmed",
                    FindingPolarity.POSITIVE,
                    [(58, "The destination account is selected explicitly.")],
                    "The source and destination are made unambiguous.",
                ),
                (
                    "outcome.transfer_confirmed",
                    FindingPolarity.POSITIVE,
                    [(142, "The confirmation recaps amount and accounts.")],
                    "The customer receives a detailed completion record.",
                ),
            ],
            [
                (
                    "Human review should decide whether name plus account "
                    "number constitutes a clear identity-verification "
                    "exchange under the project profile."
                )
            ],
        ),
    ]


def _challenge_provenance(axis: CounterfactualAxis) -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator_id="controlled-challenge-author",
        annotator_type=AnnotatorType.SYNTHETIC_AUTHOR,
        status=AnnotationStatus.ADJUDICATED,
        method="known-answer counterfactual authored from one declared change",
        requires_human_review=False,
        ground_truth_basis=(
            f"the {axis.value} manipulation is explicit and all declared "
            "invariants are checked"
        ),
    )


def _challenge_segments(
    call_id: str,
    turns: list[tuple[Speaker, str]],
    *,
    seconds_per_turn: float = 4.0,
) -> list[ChallengeSegment]:
    segments = []
    for index, (speaker, text) in enumerate(turns, start=1):
        start = round((index - 1) * seconds_per_turn, 2)
        end = round(start + min(3.0, seconds_per_turn), 2)
        segments.append(
            ChallengeSegment(
                segment_id=f"{call_id}:segment:{index}",
                seq_id=index,
                speaker=speaker,
                start_seconds=start,
                end_seconds=end,
                text=text,
            )
        )
    return segments


def _challenge_annotation(
    call_id: str,
    axis: CounterfactualAxis,
    attention: ExpectedAttention,
    segments: list[ChallengeSegment],
    finding_specs: list[
        tuple[str, FindingPolarity, list[int], str]
    ],
) -> CallAnnotation:
    evidence = []
    findings = []
    by_seq = {item.seq_id: item for item in segments}
    for finding_type, polarity, seq_ids, rationale in finding_specs:
        finding_evidence = []
        for seq_id in seq_ids:
            segment = by_seq[seq_id]
            item = AnnotationEvidence(
                evidence_id=f"evidence:{call_id}:{seq_id}",
                segment_id=segment.segment_id,
                seq_id=segment.seq_id,
                speaker=segment.speaker,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                quote=segment.text,
                purpose=f"Known-answer evidence for {finding_type}.",
            )
            if item.evidence_id not in {
                known.evidence_id for known in evidence
            }:
                evidence.append(item)
            finding_evidence.append(item)
        findings.append(
            _finding(
                finding_type,
                polarity,
                finding_evidence,
                rationale,
            )
        )
    return CallAnnotation(
        annotation_id=f"annotation:{call_id}:known-answer",
        call_id=call_id,
        profile_id=PROFILE_ID,
        source_type=EvaluationSourceType.CONTROLLED_CHALLENGE,
        expected_attention=attention,
        expected_findings=findings,
        evidence=evidence,
        provenance=_challenge_provenance(axis),
    )


def _challenge_call(
    call_id: str,
    title: str,
    axis: CounterfactualAxis,
    turns: list[tuple[Speaker, str]],
    attention: ExpectedAttention,
    finding_specs: list[
        tuple[str, FindingPolarity, list[int], str]
    ],
    *,
    acoustic: AcousticCondition = AcousticCondition.CALM,
    seconds_per_turn: float = 4.0,
) -> ChallengeCall:
    segments = _challenge_segments(
        call_id,
        turns,
        seconds_per_turn=seconds_per_turn,
    )
    return ChallengeCall(
        call_id=call_id,
        title=title,
        intent_ids=["transfer.one_time"],
        facts={
            "request.executes_transaction": True,
            "transfer.executed_during_call": True,
        },
        acoustic_condition=acoustic,
        segments=segments,
        annotation=_challenge_annotation(
            call_id,
            axis,
            attention,
            segments,
            finding_specs,
        ),
    )


def _clean_turns() -> list[tuple[Speaker, str]]:
    return [
        (Speaker.AGENT, "Thank you for calling North Star Bank. My name is Maya."),
        (Speaker.CUSTOMER, "Please transfer 200 dollars from savings to checking today."),
        (Speaker.AGENT, "Before I access the accounts, please confirm your date of birth."),
        (Speaker.CUSTOMER, "It is May 4, 1988."),
        (Speaker.AGENT, "I have 200 dollars from savings to checking today, with no fee."),
        (Speaker.CUSTOMER, "Yes, please proceed."),
        (Speaker.AGENT, "The 200 dollar transfer is complete and available today."),
    ]


def build_challenges() -> tuple[
    list[ChallengeCall],
    list[CounterfactualPair],
]:
    positive_clean = [
        (
            "control.identity_verification_completed",
            FindingPolarity.POSITIVE,
            [3],
            "Identity is checked before account access.",
        ),
        (
            "control.transfer_authorization_obtained",
            FindingPolarity.POSITIVE,
            [6],
            "The customer explicitly authorizes the transfer.",
        ),
        (
            "outcome.transfer_confirmed",
            FindingPolarity.POSITIVE,
            [7],
            "The agent confirms the completed outcome.",
        ),
    ]
    calls = []
    pairs = []

    baseline = _clean_turns() + [
        (Speaker.AGENT, "A manager reviewed the account settings yesterday.")
    ]
    variant = _clean_turns() + [
        (Speaker.CUSTOMER, "I want to speak to a manager about this.")
    ]
    calls.extend(
        [
            _challenge_call(
                "challenge-transcript-context",
                "Manager word used as account context",
                CounterfactualAxis.TRANSCRIPT,
                baseline,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
            ),
            _challenge_call(
                "challenge-transcript-request",
                "Customer explicitly requests a manager",
                CounterfactualAxis.TRANSCRIPT,
                variant,
                ExpectedAttention.REQUIRED,
                [
                    (
                        "escalation.manager_requested",
                        FindingPolarity.NEGATIVE,
                        [8],
                        "The customer explicitly asks for a manager.",
                    ),
                    *positive_clean,
                ],
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:transcript:manager-attribution",
            axis=CounterfactualAxis.TRANSCRIPT,
            baseline_call_id="challenge-transcript-context",
            variant_call_id="challenge-transcript-request",
            controlled_change=(
                "A contextual agent statement becomes an explicit customer "
                "manager request."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "acoustic_condition",
            ],
            expected_effect=ExpectedPairEffect.ATTENTION_CHANGES,
        )
    )

    delivery_turns = _clean_turns() + [
        (Speaker.CUSTOMER, "I am very frustrated that this took so long.")
    ]
    delivery_findings = [
        (
            "escalation.explicit_dissatisfaction",
            FindingPolarity.NEGATIVE,
            [8],
            "The explicit text requires review in both delivery conditions.",
        ),
        *positive_clean,
    ]
    calls.extend(
        [
            _challenge_call(
                "challenge-delivery-calm",
                "Explicit dissatisfaction with calm delivery",
                CounterfactualAxis.DELIVERY,
                delivery_turns,
                ExpectedAttention.REQUIRED,
                delivery_findings,
                acoustic=AcousticCondition.CALM,
            ),
            _challenge_call(
                "challenge-delivery-elevated",
                "Explicit dissatisfaction with persistent elevation",
                CounterfactualAxis.DELIVERY,
                delivery_turns,
                ExpectedAttention.REQUIRED,
                delivery_findings,
                acoustic=AcousticCondition.PERSISTENT_ELEVATION,
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:delivery:acoustic-support",
            axis=CounterfactualAxis.DELIVERY,
            baseline_call_id="challenge-delivery-calm",
            variant_call_id="challenge-delivery-elevated",
            controlled_change=(
                "Only the authored acoustic condition changes from calm "
                "to persistent elevation."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "transcript",
                "attention",
            ],
            expected_effect=ExpectedPairEffect.SUPPORT_CHANGES,
        )
    )

    outcome_base = _clean_turns()
    outcome_variant = _clean_turns()
    outcome_variant[-1] = (
        Speaker.AGENT,
        "I have noted your transfer request.",
    )
    calls.extend(
        [
            _challenge_call(
                "challenge-outcome-confirmed",
                "Completed transfer has a clear confirmation",
                CounterfactualAxis.OUTCOME,
                outcome_base,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
            ),
            _challenge_call(
                "challenge-outcome-unclear",
                "Completed transfer lacks outcome confirmation",
                CounterfactualAxis.OUTCOME,
                outcome_variant,
                ExpectedAttention.REQUIRED,
                [
                    (
                        "outcome.transfer_confirmation_missing",
                        FindingPolarity.NEGATIVE,
                        [7],
                        "The closing statement does not say whether execution succeeded.",
                    ),
                    *positive_clean[:2],
                ],
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:outcome:completion-confirmation",
            axis=CounterfactualAxis.OUTCOME,
            baseline_call_id="challenge-outcome-confirmed",
            variant_call_id="challenge-outcome-unclear",
            controlled_change=(
                "The completion statement becomes an ambiguous request "
                "acknowledgement."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "acoustic_condition",
            ],
            expected_effect=ExpectedPairEffect.ATTENTION_CHANGES,
        )
    )

    identity_base = _clean_turns()
    identity_variant = _clean_turns()
    identity_variant[2] = (
        Speaker.AGENT,
        "I can see the savings and checking accounts on your profile.",
    )
    identity_variant[3] = (
        Speaker.CUSTOMER,
        "Those are the correct accounts.",
    )
    calls.extend(
        [
            _challenge_call(
                "challenge-control-identity-present",
                "Identity verification precedes account action",
                CounterfactualAxis.CONTROL,
                identity_base,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
            ),
            _challenge_call(
                "challenge-control-identity-missing",
                "Account action proceeds without identity verification",
                CounterfactualAxis.CONTROL,
                identity_variant,
                ExpectedAttention.REQUIRED,
                [
                    (
                        "control.identity_verification_missing",
                        FindingPolarity.NEGATIVE,
                        [3],
                        "The agent accesses account details without a verification exchange.",
                    ),
                    *positive_clean[1:],
                ],
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:control:identity-verification",
            axis=CounterfactualAxis.CONTROL,
            baseline_call_id="challenge-control-identity-present",
            variant_call_id="challenge-control-identity-missing",
            controlled_change=(
                "The identity request and response become unverified "
                "account acknowledgement."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "acoustic_condition",
            ],
            expected_effect=ExpectedPairEffect.ATTENTION_CHANGES,
        )
    )

    auth_base = _clean_turns()
    auth_variant = _clean_turns()
    auth_variant[5] = (
        Speaker.AGENT,
        "I am processing that transfer now.",
    )
    calls.extend(
        [
            _challenge_call(
                "challenge-control-authorization-present",
                "Explicit approval follows material terms",
                CounterfactualAxis.CONTROL,
                auth_base,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
            ),
            _challenge_call(
                "challenge-control-authorization-missing",
                "Execution proceeds without explicit approval",
                CounterfactualAxis.CONTROL,
                auth_variant,
                ExpectedAttention.REQUIRED,
                [
                    (
                        "control.transfer_authorization_missing",
                        FindingPolarity.NEGATIVE,
                        [6],
                        "The agent executes without customer approval after the recap.",
                    ),
                    positive_clean[0],
                    positive_clean[2],
                ],
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:control:transfer-authorization",
            axis=CounterfactualAxis.CONTROL,
            baseline_call_id="challenge-control-authorization-present",
            variant_call_id="challenge-control-authorization-missing",
            controlled_change=(
                "The customer's explicit approval becomes an agent "
                "execution statement."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "acoustic_condition",
            ],
            expected_effect=ExpectedPairEffect.ATTENTION_CHANGES,
        )
    )

    recovery_turns = _clean_turns() + [
        (Speaker.CUSTOMER, "I am very frustrated about the delay."),
        (Speaker.AGENT, "I understand. The transfer is complete now."),
        (Speaker.CUSTOMER, "Thank you for resolving it."),
    ]
    recovery_negative = (
        "escalation.explicit_dissatisfaction",
        FindingPolarity.NEGATIVE,
        [8],
        "Explicit dissatisfaction remains reviewable after recovery.",
    )
    calls.extend(
        [
            _challenge_call(
                "challenge-recovery-observed",
                "Elevated customer delivery settles before close",
                CounterfactualAxis.RECOVERY,
                recovery_turns,
                ExpectedAttention.REQUIRED,
                [
                    recovery_negative,
                    *positive_clean,
                    (
                        "escalation.recovery_observed",
                        FindingPolarity.POSITIVE,
                        [10],
                        "The authored delivery condition settles after resolution.",
                    ),
                ],
                acoustic=AcousticCondition.RECOVERED_ELEVATION,
            ),
            _challenge_call(
                "challenge-recovery-unresolved",
                "Elevated customer delivery remains unresolved",
                CounterfactualAxis.RECOVERY,
                recovery_turns,
                ExpectedAttention.REQUIRED,
                [recovery_negative, *positive_clean],
                acoustic=AcousticCondition.UNRESOLVED_ELEVATION,
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:recovery:settled-versus-unresolved",
            axis=CounterfactualAxis.RECOVERY,
            baseline_call_id="challenge-recovery-observed",
            variant_call_id="challenge-recovery-unresolved",
            controlled_change=(
                "Only the authored late-call acoustic trajectory changes "
                "from recovered to unresolved."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "transcript",
                "attention",
            ],
            expected_effect=ExpectedPairEffect.RECOVERY_CHANGES,
        )
    )

    duration_turns = _clean_turns()
    calls.extend(
        [
            _challenge_call(
                "challenge-duration-short",
                "Clean transfer at ordinary spacing",
                CounterfactualAxis.DURATION,
                duration_turns,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
                seconds_per_turn=4.0,
            ),
            _challenge_call(
                "challenge-duration-long",
                "Same clean transfer with long spacing",
                CounterfactualAxis.DURATION,
                duration_turns,
                ExpectedAttention.NOT_REQUIRED,
                positive_clean,
                seconds_per_turn=18.0,
            ),
        ]
    )
    pairs.append(
        CounterfactualPair(
            pair_id="pair:duration:spacing-invariance",
            axis=CounterfactualAxis.DURATION,
            baseline_call_id="challenge-duration-short",
            variant_call_id="challenge-duration-long",
            controlled_change=(
                "Only segment timing and total duration are expanded."
            ),
            invariant_dimensions=[
                "intent_and_facts",
                "transcript",
                "acoustic_condition",
                "attention",
            ],
            expected_effect=ExpectedPairEffect.NO_DECISION_CHANGE,
        )
    )
    return calls, pairs


def build_dataset() -> EvaluationDataset:
    challenges, pairs = build_challenges()
    return EvaluationDataset(
        dataset_id=EVALUATION_DATASET_ID,
        dataset_version=EVALUATION_DATASET_VERSION,
        profile_id=PROFILE_ID,
        existing_calls=build_existing_records(),
        challenge_calls=challenges,
        counterfactual_pairs=pairs,
    )


def build_summary(dataset: EvaluationDataset) -> dict[str, Any]:
    existing_statuses = Counter(
        item.annotation.provenance.status.value
        for item in dataset.existing_calls
    )
    existing_attention = Counter(
        item.annotation.expected_attention.value
        for item in dataset.existing_calls
    )
    axes = Counter(
        item.axis.value for item in dataset.counterfactual_pairs
    )
    challenge_attention = Counter(
        item.annotation.expected_attention.value
        for item in dataset.challenge_calls
    )
    return {
        "schema_version": "1.0",
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "profile_id": dataset.profile_id,
        "existing_calls": {
            "count": len(dataset.existing_calls),
            "annotation_status_counts": dict(
                sorted(existing_statuses.items())
            ),
            "expected_attention_counts": dict(
                sorted(existing_attention.items())
            ),
            "human_adjudicated_count": sum(
                item.annotation.provenance.annotator_type
                == AnnotatorType.HUMAN
                and item.annotation.provenance.status
                == AnnotationStatus.ADJUDICATED
                for item in dataset.existing_calls
            ),
            "requires_human_review_count": sum(
                item.annotation.provenance.requires_human_review
                for item in dataset.existing_calls
            ),
        },
        "controlled_challenges": {
            "call_count": len(dataset.challenge_calls),
            "pair_count": len(dataset.counterfactual_pairs),
            "pair_axis_counts": dict(sorted(axes.items())),
            "expected_attention_counts": dict(
                sorted(challenge_attention.items())
            ),
            "known_answer_count": sum(
                item.annotation.provenance.status
                == AnnotationStatus.ADJUDICATED
                for item in dataset.challenge_calls
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    args = parser.parse_args()

    dataset = build_dataset()
    validate_evaluation_dataset(dataset, REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(build_summary(dataset), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(dataset.existing_calls)} existing-call labels, "
        f"{len(dataset.challenge_calls)} challenge calls, and "
        f"{len(dataset.counterfactual_pairs)} pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
