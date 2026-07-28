import json
import unittest
from pathlib import Path

from v2.domain_profiles import (
    PolicyAuthority,
    ProfileSelection,
    load_banking_profile,
    resolve_domain_plan,
)

HERE = Path(__file__).resolve().parent
SELECTIONS = HERE / "profiles" / "banking_call_selections.json"


class BankingProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_banking_profile()
        cls.fixtures = json.loads(
            SELECTIONS.read_text(encoding="utf-8")
        )["calls"]

    def test_profile_is_research_policy_not_regulatory_claim(self):
        self.assertEqual(self.profile.profile_id, "banking-demo-v1")
        self.assertEqual(len(self.profile.intents), 4)
        self.assertNotIn(
            PolicyAuthority.REGULATION,
            {source.authority for source in self.profile.policy_sources},
        )
        self.assertIn("not", self.profile.disclaimer.lower())
        self.assertNotIn(
            "recording_disclosure",
            {
                requirement.requirement_id
                for requirement in self.profile.requirements
            },
        )
        requirement_ids = {
            requirement.requirement_id
            for requirement in self.profile.requirements
        }
        workflow_requirement_ids = {
            step.requirement_id
            for branch in self.profile.workflow_branches
            for step in branch.steps
        }
        self.assertEqual(requirement_ids, workflow_requirement_ids)

    def test_all_ten_banking_calls_resolve_without_unknown_applicability(self):
        plans = []
        for fixture in self.fixtures:
            selection = ProfileSelection.model_validate(fixture)
            plan = resolve_domain_plan(self.profile, selection)
            plans.append(plan)
            self.assertFalse(
                plan.unresolved_applicability,
                msg=selection.call_id,
            )
            self.assertGreater(len(plan.requirements), 0)
            self.assertGreater(len(plan.workflow_steps), 0)
            workflow_requirement_ids = [
                step.requirement_id for step in plan.workflow_steps
            ]
            self.assertEqual(
                len(workflow_requirement_ids),
                len(set(workflow_requirement_ids)),
            )
            self.assertEqual(
                {item.requirement_id for item in plan.requirements},
                set(workflow_requirement_ids),
            )

        self.assertEqual(len(plans), 10)
        self.assertEqual(
            {intent for plan in plans for intent in plan.selected_intent_ids},
            {
                "transfer.one_time",
                "transfer.recurring",
                "account.open_checking",
                "loan.agricultural_equipment",
            },
        )

    def test_general_loan_inquiry_does_not_require_identity_verification(self):
        selection = ProfileSelection(
            call_id="loan-inquiry",
            intent_ids=["loan.agricultural_equipment"],
            facts={"request.starts_application": False},
            selection_method="test",
        )
        plan = resolve_domain_plan(self.profile, selection)
        requirement_ids = {
            requirement.requirement_id
            for requirement in plan.requirements
        }
        step_ids = {step.step_id for step in plan.workflow_steps}

        self.assertNotIn(
            "security.loan_application_identity_verified",
            requirement_ids,
        )
        self.assertNotIn(
            "workflow.loan.verify_applicant",
            step_ids,
        )

    def test_loan_application_adds_identity_branch(self):
        selection = ProfileSelection(
            call_id="loan-application",
            intent_ids=["loan.agricultural_equipment"],
            facts={"request.starts_application": True},
            selection_method="test",
        )
        plan = resolve_domain_plan(self.profile, selection)
        requirement_ids = {
            requirement.requirement_id
            for requirement in plan.requirements
        }
        step_ids = {step.step_id for step in plan.workflow_steps}

        self.assertIn(
            "security.loan_application_identity_verified",
            requirement_ids,
        )
        self.assertIn(
            "workflow.loan.verify_applicant",
            step_ids,
        )

    def test_mixed_transfer_selects_both_detail_branches_without_duplicates(self):
        selection = ProfileSelection(
            call_id="mixed-transfer",
            intent_ids=["transfer.one_time", "transfer.recurring"],
            facts={
                "request.executes_transaction": True,
                "transfer.executed_during_call": True,
            },
            selection_method="test",
        )
        plan = resolve_domain_plan(self.profile, selection)
        self.assertIn(
            "transfer.one_time.details",
            plan.selected_branch_ids,
        )
        self.assertIn(
            "transfer.recurring.details",
            plan.selected_branch_ids,
        )
        step_ids = [step.requirement_id for step in plan.workflow_steps]
        self.assertEqual(len(step_ids), len(set(step_ids)))
        sequences = [step.sequence for step in plan.workflow_steps]
        self.assertEqual(sequences, sorted(sequences))

    def test_missing_branch_fact_is_unresolved_not_failed_or_applied(self):
        selection = ProfileSelection(
            call_id="transfer-with-missing-facts",
            intent_ids=["transfer.one_time"],
            facts={},
            selection_method="test",
        )
        plan = resolve_domain_plan(self.profile, selection)
        unresolved_ids = {
            item.subject_id for item in plan.unresolved_applicability
        }

        self.assertIn("transfer.execution", unresolved_ids)
        self.assertIn("transfer.confirmation", unresolved_ids)
        self.assertNotIn(
            "transfer.execution",
            plan.selected_branch_ids,
        )

    def test_linkage_requirement_depends_on_selected_fact(self):
        base = {
            "call_id": "checking",
            "intent_ids": ["account.open_checking"],
            "facts": {
                "request.opens_account": True,
                "account.link_existing_product": False,
            },
            "selection_method": "test",
        }
        without_linkage = resolve_domain_plan(
            self.profile,
            ProfileSelection.model_validate(base),
        )
        without_ids = {
            requirement.requirement_id
            for requirement in without_linkage.requirements
        }
        self.assertNotIn("account.linkage_confirmed", without_ids)

        base["facts"]["account.link_existing_product"] = True
        with_linkage = resolve_domain_plan(
            self.profile,
            ProfileSelection.model_validate(base),
        )
        with_ids = {
            requirement.requirement_id
            for requirement in with_linkage.requirements
        }
        self.assertIn("account.linkage_confirmed", with_ids)
        self.assertIn(
            "account.product_linkage",
            with_linkage.selected_branch_ids,
        )

    def test_unknown_intent_and_fact_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_domain_plan(
                self.profile,
                ProfileSelection(
                    call_id="unknown-intent",
                    intent_ids=["banking.unknown"],
                    selection_method="test",
                ),
            )
        with self.assertRaises(ValueError):
            resolve_domain_plan(
                self.profile,
                ProfileSelection(
                    call_id="unknown-fact",
                    intent_ids=["transfer.one_time"],
                    facts={"transfer.magic": True},
                    selection_method="test",
                ),
            )


if __name__ == "__main__":
    unittest.main()
