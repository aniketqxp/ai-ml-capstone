from __future__ import annotations

import json
import unittest

from v2.adapters import build_signal_bundle
from v2.domain_profiles import (
    DomainProfile,
    ProfileSelection,
    resolve_domain_plan,
)
from v2.runtime import PROFILE_PATH
from v2.semantic_assessment import assess_requirements


def _bundle():
    return build_signal_bundle({
        "call_id": "semantic-test",
        "domain": "banking",
        "sentences": [
            {
                "id": "AGENT_001",
                "speaker": "AGENT",
                "start": 0.0,
                "end": 2.0,
                "text": "My name is Emily from North Star Bank.",
            },
            {
                "id": "CUSTOMER_001",
                "speaker": "CUSTOMER",
                "start": 2.1,
                "end": 4.0,
                "text": "Please set up a monthly transfer.",
            },
        ],
    })


def _plan():
    profile = DomainProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    return resolve_domain_plan(
        profile,
        ProfileSelection(
            call_id="semantic-test",
            intent_ids=["transfer.recurring"],
            facts={
                "request.executes_transaction": True,
                "transfer.executed_during_call": True,
            },
            selection_method="test",
        ),
    )


def _response(plan, segment_id="AGENT_001"):
    return json.dumps({
        "assessments": [
            {
                "requirement_id": item.requirement_id,
                "verdict": "met",
                "rationale": f"{item.title} is directly demonstrated.",
                "evidence_segment_ids": [segment_id],
                "counter_evidence_segment_ids": [],
            }
            for item in plan.requirements
        ],
    })


class SemanticAssessmentTests(unittest.TestCase):
    def test_materializes_exact_transcript_evidence(self):
        bundle = _bundle()
        plan = _plan()

        batch = assess_requirements(
            bundle,
            plan,
            completion=lambda _system, _user, _tier: (
                _response(plan),
                "test-model",
            ),
        )

        self.assertEqual(len(batch.assessments), len(plan.requirements))
        evidence = batch.assessments[0].evidence[0]
        self.assertEqual(evidence.segment_ids, ["AGENT_001"])
        self.assertEqual(
            evidence.observation,
            "AGENT: My name is Emily from North Star Bank.",
        )
        self.assertEqual(batch.assessments[0].counter_evidence, [])

    def test_retries_unknown_segment_references(self):
        bundle = _bundle()
        plan = _plan()
        responses = iter([
            (_response(plan, "UNKNOWN"), "bad-model"),
            (_response(plan), "good-model"),
        ])

        batch = assess_requirements(
            bundle,
            plan,
            completion=lambda _system, _user, _tier: next(responses),
        )

        self.assertEqual(len(batch.assessments), len(plan.requirements))

    def test_moves_to_next_tier_after_transport_failure(self):
        bundle = _bundle()
        plan = _plan()
        tiers = []

        def completion(_system, _user, tier):
            tiers.append(tier)
            if tier == "qa-primary":
                raise TimeoutError("primary timed out")
            return _response(plan), "fallback-model"

        batch = assess_requirements(
            bundle,
            plan,
            completion=completion,
        )

        self.assertEqual(len(batch.assessments), len(plan.requirements))
        self.assertEqual(tiers, ["qa-primary", "qa-fallback"])

    def test_normalizes_an_agreeing_duplicate_requirement(self):
        bundle = _bundle()
        plan = _plan()
        response = json.loads(_response(plan))
        response["assessments"].append(
            dict(response["assessments"][0])
        )

        batch = assess_requirements(
            bundle,
            plan,
            completion=lambda _system, _user, _tier: (
                json.dumps(response),
                "test-model",
            ),
        )

        self.assertEqual(len(batch.assessments), len(plan.requirements))

    def test_account_identifiers_do_not_satisfy_identity_verification(self):
        bundle = build_signal_bundle({
            "call_id": "semantic-test",
            "domain": "banking",
            "sentences": [
                {
                    "id": "AGENT_001",
                    "speaker": "AGENT",
                    "start": 0.0,
                    "end": 2.0,
                    "text": "May I have your checking account number?",
                },
                {
                    "id": "CUSTOMER_001",
                    "speaker": "CUSTOMER",
                    "start": 2.1,
                    "end": 4.0,
                    "text": "It is 123456.",
                },
            ],
        })
        plan = _plan()

        batch = assess_requirements(
            bundle,
            plan,
            completion=lambda _system, _user, _tier: (
                _response(plan),
                "test-model",
            ),
        )

        identity = next(
            item
            for item in batch.assessments
            if item.requirement_id
            == "security.transaction_identity_verified"
        )
        self.assertEqual(identity.verdict.value, "missed")
        self.assertEqual(
            identity.evidence[0].segment_ids,
            ["AGENT_001"],
        )

    def test_challenge_and_response_can_satisfy_identity_verification(self):
        bundle = build_signal_bundle({
            "call_id": "semantic-test",
            "domain": "banking",
            "sentences": [
                {
                    "id": "AGENT_001",
                    "speaker": "AGENT",
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Please confirm your date of birth.",
                },
                {
                    "id": "CUSTOMER_001",
                    "speaker": "CUSTOMER",
                    "start": 2.1,
                    "end": 4.0,
                    "text": "January 5, 1983.",
                },
            ],
        })
        plan = _plan()

        batch = assess_requirements(
            bundle,
            plan,
            completion=lambda _system, _user, _tier: (
                _response(plan),
                "test-model",
            ),
        )

        identity = next(
            item
            for item in batch.assessments
            if item.requirement_id
            == "security.transaction_identity_verified"
        )
        self.assertEqual(identity.verdict.value, "met")

    def test_rejects_conflicting_duplicate_requirement(self):
        bundle = _bundle()
        plan = _plan()
        response = json.loads(_response(plan))
        duplicate = dict(response["assessments"][0])
        duplicate["verdict"] = "missed"
        response["assessments"].append(duplicate)

        with self.assertRaises(RuntimeError):
            assess_requirements(
                bundle,
                plan,
                completion=lambda _system, _user, _tier: (
                    json.dumps(response),
                    "test-model",
                ),
                max_attempts=1,
            )

    def test_rejects_incomplete_requirement_coverage(self):
        bundle = _bundle()
        plan = _plan()
        response = json.loads(_response(plan))
        response["assessments"].pop()

        with self.assertRaises(RuntimeError):
            assess_requirements(
                bundle,
                plan,
                completion=lambda _system, _user, _tier: (
                    json.dumps(response),
                    "test-model",
                ),
                max_attempts=1,
            )


if __name__ == "__main__":
    unittest.main()
