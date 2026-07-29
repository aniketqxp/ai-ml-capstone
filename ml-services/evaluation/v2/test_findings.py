"""Behavioral tests for evidence-backed finding derivation."""
from __future__ import annotations

import unittest

from v2.domain_profiles import (
    ProfileSelection,
    load_banking_profile,
    resolve_domain_plan,
)
from v2.findings import (
    RequirementAssessment,
    RequirementAssessmentBatch,
    RequirementVerdict,
    derive_findings,
)
from v2.schemas import (
    EvidenceRef,
    FindingCategory,
    FindingPolarity,
    FindingSeverity,
    Modality,
    ReliabilityAssessment,
    ReliabilityStatus,
    SignalBundle,
    SignalEpisode,
    SignalRecord,
    SignalScope,
    SourceProvenance,
    Speaker,
    TranscriptSegment,
    Visibility,
)


def _source(producer: str = "test") -> SourceProvenance:
    return SourceProvenance(
        producer=producer,
        producer_version="1",
        method="synthetic_test_fixture",
    )


def _bundle(
    customer_text: str = "I want to speak to your manager.",
) -> SignalBundle:
    source = _source()
    rows = [
        (Speaker.CUSTOMER, "Please transfer one hundred dollars."),
        (Speaker.AGENT, "You want to transfer one hundred dollars."),
        (Speaker.CUSTOMER, customer_text),
        (Speaker.AGENT, "I can help resolve this with you."),
        (Speaker.CUSTOMER, "Okay, thank you for checking."),
        (Speaker.AGENT, "The transfer is complete."),
    ]
    segments = [
        TranscriptSegment(
            segment_id=f"segment-{index:02d}",
            segment_index=index,
            seq_id=index,
            speaker=speaker,
            start_seconds=float(index * 4),
            end_seconds=float(index * 4 + 2),
            text=text,
            provenance=source,
        )
        for index, (speaker, text) in enumerate(rows)
    ]
    return SignalBundle(
        call_id="test-call",
        domain="banking",
        duration_seconds=24.0,
        segments=segments,
        sources=[source],
    )


def _plan(
    *,
    intents: list[str] | None = None,
    facts: dict[str, bool] | None = None,
):
    intents = intents or ["transfer.one_time"]
    facts = facts or {
        "request.executes_transaction": True,
        "transfer.executed_during_call": True,
    }
    return resolve_domain_plan(
        load_banking_profile(),
        ProfileSelection(
            call_id="test-call",
            intent_ids=intents,
            facts=facts,
            selection_method="test",
        ),
    )


def _evidence(
    bundle: SignalBundle,
    segment_index: int,
    prefix: str,
) -> EvidenceRef:
    segment = bundle.segments[segment_index]
    return EvidenceRef(
        evidence_id=f"{prefix}-{segment.segment_id}",
        modality=Modality.TRANSCRIPT,
        segment_ids=[segment.segment_id],
        speaker=segment.speaker,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        quote=segment.text,
    )


def _assessment(
    bundle: SignalBundle,
    requirement_id: str,
    verdict: RequirementVerdict,
    *,
    evidence_index: int = 1,
    counter_index: int = 0,
) -> RequirementAssessment:
    return RequirementAssessment(
        assessment_id=f"assessment:{requirement_id}",
        requirement_id=requirement_id,
        verdict=verdict,
        rationale=f"Synthetic {verdict.value} assessment.",
        evidence=[
            _evidence(
                bundle,
                evidence_index,
                f"support-{requirement_id}",
            )
        ],
        counter_evidence=[
            _evidence(
                bundle,
                counter_index,
                f"counter-{requirement_id}",
            )
        ],
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.USABLE,
            coverage_ratio=1.0,
        ),
        provenance=_source("requirement_assessor"),
    )


def _batch(
    assessments: list[RequirementAssessment],
) -> RequirementAssessmentBatch:
    return RequirementAssessmentBatch(
        call_id="test-call",
        profile_id="banking-demo-v1",
        assessments=assessments,
        provenance=_source("requirement_assessor"),
    )


def _with_acoustic_episode(
    bundle: SignalBundle,
    *,
    recovery: bool,
) -> SignalBundle:
    output = bundle.model_copy(deep=True)
    source = _source("evaluator_v2.acoustic_dynamics")
    signal_ids = []
    for index in (2, 4):
        segment = output.segments[index]
        signal = SignalRecord(
            signal_id=(
                f"{segment.segment_id}:"
                "acoustic.dynamics.segment_elevated"
            ),
            name="acoustic.dynamics.segment_elevated",
            modality=Modality.ACOUSTIC,
            scope=SignalScope.SEGMENT,
            value=True,
            segment_id=segment.segment_id,
            seq_id=segment.seq_id,
            speaker=segment.speaker,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            reliability=ReliabilityAssessment(
                status=ReliabilityStatus.LIMITED,
                reasons=["test_episode"],
            ),
            provenance=source,
            visibility=Visibility.INTERNAL,
        )
        output.signals.append(signal)
        signal_ids.append(signal.signal_id)
    recovery_signal = SignalRecord(
        signal_id="call:acoustic.dynamics.recovery_candidate",
        name="acoustic.dynamics.recovery_candidate",
        modality=Modality.ACOUSTIC,
        scope=SignalScope.CALL,
        value=recovery,
        reliability=ReliabilityAssessment(
            status=ReliabilityStatus.LIMITED,
            reasons=["test_recovery"],
        ),
        provenance=source,
        visibility=Visibility.INTERNAL,
    )
    output.signals.append(recovery_signal)
    output.episodes.append(
        SignalEpisode(
            episode_id="customer-elevation-01",
            episode_type="acoustic.customer_elevation",
            modality=Modality.ACOUSTIC,
            speaker=Speaker.CUSTOMER,
            start_seconds=8.0,
            end_seconds=18.0,
            segment_ids=["segment-02", "segment-04"],
            signal_ids=signal_ids,
            observation="Synthetic persistent elevation.",
            reliability=ReliabilityAssessment(
                status=ReliabilityStatus.LIMITED,
                reasons=["test_episode"],
            ),
            provenance=source,
            visibility=Visibility.INTERNAL,
        )
    )
    output.sources.append(source)
    return SignalBundle.model_validate(output.model_dump())


class RequirementFindingTests(unittest.TestCase):
    def test_missed_critical_control_becomes_critical_finding(self):
        bundle = _bundle("The amount is correct.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.MISSED,
        )

        result = derive_findings(
            bundle,
            plan,
            _batch([assessment]),
        )
        finding = next(
            item
            for item in result.triggered_findings
            if item.finding_type
            == "control.transfer_amount_unconfirmed"
        )

        self.assertEqual(
            finding.category,
            FindingCategory.REQUIRED_CONTROL,
        )
        self.assertEqual(
            finding.severity,
            FindingSeverity.CRITICAL,
        )
        self.assertEqual(
            finding.polarity,
            FindingPolarity.NEGATIVE,
        )
        self.assertTrue(finding.evidence)
        self.assertTrue(finding.counter_evidence)

    def test_met_outcome_becomes_positive_finding(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.completion_confirmed",
            RequirementVerdict.MET,
            evidence_index=5,
        )

        result = derive_findings(
            bundle,
            plan,
            _batch([assessment]),
        )
        finding = next(
            item
            for item in result.positive_findings
            if item.finding_type == "outcome.transfer_confirmed"
        )

        self.assertEqual(finding.category, FindingCategory.OUTCOME)
        self.assertEqual(finding.severity, FindingSeverity.INFO)

    def test_incorrect_behavior_remains_distinct_from_an_omission(self):
        bundle = _bundle("The transfer amount is five hundred dollars.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.INCORRECT,
        )

        result = derive_findings(
            bundle,
            plan,
            _batch([assessment]),
        )
        finding = next(
            item
            for item in result.triggered_findings
            if item.finding_type
            == "control.transfer_amount_unconfirmed"
        )

        self.assertEqual(
            finding.detection_rule.thresholds[0].value,
            "incorrect",
        )
        self.assertEqual(finding.summary, assessment.rationale)

    def test_uncertain_and_missing_assessments_do_not_become_failures(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.UNCERTAIN,
        )

        result = derive_findings(
            bundle,
            plan,
            _batch([assessment]),
        )

        self.assertEqual(result.triggered_findings, [])
        self.assertEqual(result.positive_findings, [])
        self.assertEqual(
            len(result.uncertainties),
            len(plan.requirements),
        )

    def test_non_applicable_assessment_is_rejected(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "account.opening_consent",
            RequirementVerdict.MISSED,
        )

        with self.assertRaisesRegex(
            ValueError,
            "non-applicable",
        ):
            derive_findings(
                bundle,
                plan,
                _batch([assessment]),
            )

    def test_non_verbatim_quote_is_rejected(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.MISSED,
        )
        assessment.evidence[0].quote = "Text that was never spoken."

        with self.assertRaisesRegex(ValueError, "not verbatim"):
            derive_findings(
                bundle,
                plan,
                _batch([assessment]),
            )

    def test_observation_only_support_is_rejected(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.MISSED,
        )
        assessment.evidence = [
            EvidenceRef(
                evidence_id="unsupported-observation",
                modality=Modality.TEXT,
                observation="The amount was not confirmed.",
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            "must reference",
        ):
            derive_findings(
                bundle,
                plan,
                _batch([assessment]),
            )

    def test_evidence_timestamp_must_match_referenced_segment(self):
        bundle = _bundle("Thank you.")
        plan = _plan()
        assessment = _assessment(
            bundle,
            "transfer.amount_confirmed",
            RequirementVerdict.MISSED,
        )
        assessment.evidence[0].start_seconds = 5.0

        with self.assertRaisesRegex(
            ValueError,
            "start time does not match",
        ):
            derive_findings(
                bundle,
                plan,
                _batch([assessment]),
            )

    def test_shared_identity_failure_type_is_deduplicated(self):
        bundle = _bundle("Thank you.")
        plan = _plan(
            intents=[
                "transfer.one_time",
                "account.open_checking",
            ],
            facts={
                "request.executes_transaction": True,
                "transfer.executed_during_call": True,
                "request.opens_account": True,
                "account.link_existing_product": False,
                "policy.requires_identity_challenge": True,
            },
        )
        assessments = [
            _assessment(
                bundle,
                requirement_id,
                RequirementVerdict.MISSED,
            )
            for requirement_id in (
                "security.transaction_identity_verified",
                "security.account_opening_identity_verified",
            )
        ]

        result = derive_findings(
            bundle,
            plan,
            _batch(assessments),
        )
        identity_findings = [
            finding
            for finding in result.triggered_findings
            if finding.finding_type
            == "control.identity_verification_missing"
        ]

        self.assertEqual(len(identity_findings), 1)
        self.assertTrue(result.suppressed_duplicates)
        self.assertIn(
            "security.transaction_identity_verified",
            identity_findings[0].detection_rule.rule_id,
        )
        self.assertIn(
            "security.account_opening_identity_verified",
            identity_findings[0].detection_rule.rule_id,
        )


class ExplicitEscalationTests(unittest.TestCase):
    def test_manager_request_is_detected_but_manager_reference_is_not(self):
        plan = _plan()
        detected = derive_findings(_bundle(), plan)
        ignored = derive_findings(
            _bundle("The manager reviewed my account yesterday."),
            plan,
        )

        self.assertIn(
            "escalation.manager_requested",
            {
                finding.finding_type
                for finding in detected.triggered_findings
            },
        )
        self.assertNotIn(
            "escalation.manager_requested",
            {
                finding.finding_type
                for finding in ignored.triggered_findings
            },
        )

    def test_repeated_manager_requests_form_one_finding(self):
        bundle = _bundle()
        duplicate = bundle.model_copy(deep=True)
        duplicate.segments[4].text = (
            "Please put me through to a supervisor."
        )

        result = derive_findings(duplicate, _plan())
        findings = [
            finding
            for finding in result.triggered_findings
            if finding.finding_type
            == "escalation.manager_requested"
        ]

        self.assertEqual(len(findings), 1)
        transcript_evidence = [
            evidence
            for evidence in findings[0].evidence
            if evidence.modality == Modality.TRANSCRIPT
        ]
        self.assertEqual(len(transcript_evidence), 2)

    def test_nearby_acoustic_episode_supports_text_without_duplication(self):
        bundle = _with_acoustic_episode(_bundle(), recovery=False)
        result = derive_findings(bundle, _plan())
        manager = next(
            finding
            for finding in result.triggered_findings
            if finding.finding_type
            == "escalation.manager_requested"
        )

        self.assertEqual(
            sum(
                finding.finding_type
                == "escalation.manager_requested"
                for finding in result.triggered_findings
            ),
            1,
        )
        self.assertTrue(
            any(
                evidence.modality == Modality.ACOUSTIC
                for evidence in manager.evidence
            )
        )
        self.assertIn(
            "acoustic_elevation_is_provisional_support",
            manager.reliability.reasons,
        )

    def test_acoustic_episode_alone_does_not_create_escalation(self):
        bundle = _with_acoustic_episode(
            _bundle("Thank you for checking."),
            recovery=False,
        )

        result = derive_findings(bundle, _plan())

        self.assertFalse(
            any(
                finding.category == FindingCategory.ESCALATION
                for finding in result.triggered_findings
            )
        )

    def test_recovery_requires_explicit_text_and_acoustic_support(self):
        supported = derive_findings(
            _with_acoustic_episode(_bundle(), recovery=True),
            _plan(),
        )
        acoustic_only = derive_findings(
            _with_acoustic_episode(
                _bundle("Thank you for checking."),
                recovery=True,
            ),
            _plan(),
        )

        self.assertIn(
            "escalation.recovery_observed",
            {
                finding.finding_type
                for finding in supported.positive_findings
            },
        )
        self.assertNotIn(
            "escalation.recovery_observed",
            {
                finding.finding_type
                for finding in acoustic_only.positive_findings
            },
        )

    def test_every_generated_finding_has_both_evidence_sides(self):
        result = derive_findings(
            _with_acoustic_episode(_bundle(), recovery=True),
            _plan(),
        )

        for finding in (
            result.triggered_findings + result.positive_findings
        ):
            self.assertTrue(finding.evidence)
            self.assertTrue(finding.counter_evidence)


if __name__ == "__main__":
    unittest.main()
