"""Versioned domain profiles and deterministic applicability resolution."""
from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from .schemas import ContractModel

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileStatus(str, Enum):
    RESEARCH = "research"
    VALIDATED = "validated"


class PolicyAuthority(str, Enum):
    PROJECT_POLICY = "project_policy"
    RESEARCH_EVIDENCE = "research_evidence"
    ORGANIZATION_POLICY = "organization_policy"
    REGULATION = "regulation"


class RequirementLevel(str, Enum):
    CRITICAL = "critical"
    REQUIRED = "required"
    SUPPORTIVE = "supportive"


class FactValueType(str, Enum):
    BOOLEAN = "boolean"
    ENUM = "enum"


class ConditionOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


FactValue = bool | str


class PolicySource(ContractModel):
    source_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.-]+$")
    authority: PolicyAuthority
    title: str = Field(min_length=1)
    reference: str | None = None
    notes: str = Field(min_length=1)


class IntentDefinition(ContractModel):
    intent_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    selection_guidance: list[str] = Field(min_length=1)


class FactDefinition(ContractModel):
    fact_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    description: str = Field(min_length=1)
    value_type: FactValueType
    allowed_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_allowed_values(self):
        if self.value_type == FactValueType.ENUM and not self.allowed_values:
            raise ValueError("enum facts require allowed_values")
        if self.value_type == FactValueType.BOOLEAN and self.allowed_values:
            raise ValueError("boolean facts cannot define allowed_values")
        return self


class ApplicabilityCondition(ContractModel):
    fact_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    operator: ConditionOperator
    value: FactValue | list[str] | None = None

    @model_validator(mode="after")
    def validate_operator_value(self):
        if self.operator in (
            ConditionOperator.EQUALS,
            ConditionOperator.NOT_EQUALS,
        ) and self.value is None:
            raise ValueError(f"{self.operator.value} requires a value")
        if self.operator == ConditionOperator.IN and not isinstance(
            self.value, list
        ):
            raise ValueError("in conditions require a list value")
        if self.operator in (
            ConditionOperator.IS_TRUE,
            ConditionOperator.IS_FALSE,
        ) and self.value is not None:
            raise ValueError(f"{self.operator.value} does not accept a value")
        return self


class RequirementDefinition(ContractModel):
    requirement_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    title: str = Field(min_length=1)
    business_definition: str = Field(min_length=1)
    level: RequirementLevel
    evidence_expectations: list[str] = Field(min_length=1)
    policy_source_ids: list[str] = Field(min_length=1)
    failure_finding_type: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_.]+$",
    )
    positive_finding_type: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_.]+$",
    )
    category: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    applies_to_intents: list[str] = Field(min_length=1)
    conditions: list[ApplicabilityCondition] = Field(default_factory=list)


class WorkflowStepDefinition(ContractModel):
    step_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    requirement_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    sequence: int = Field(ge=1)


class WorkflowBranch(ContractModel):
    branch_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    title: str = Field(min_length=1)
    applies_to_intents: list[str] = Field(min_length=1)
    conditions: list[ApplicabilityCondition] = Field(default_factory=list)
    steps: list[WorkflowStepDefinition] = Field(min_length=1)


class DomainProfile(ContractModel):
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    profile_id: str = Field(min_length=1, pattern=r"^[a-z0-9_.-]+$")
    version: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    status: ProfileStatus
    jurisdiction_scope: list[str] = Field(min_length=1)
    disclaimer: str = Field(min_length=1)
    policy_sources: list[PolicySource] = Field(min_length=1)
    intents: list[IntentDefinition] = Field(min_length=1)
    facts: list[FactDefinition] = Field(default_factory=list)
    requirements: list[RequirementDefinition] = Field(min_length=1)
    workflow_branches: list[WorkflowBranch] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile_references(self):
        source_ids = [source.source_id for source in self.policy_sources]
        intent_ids = [intent.intent_id for intent in self.intents]
        fact_ids = [fact.fact_id for fact in self.facts]
        requirement_ids = [
            requirement.requirement_id
            for requirement in self.requirements
        ]
        branch_ids = [branch.branch_id for branch in self.workflow_branches]
        step_ids = [
            step.step_id
            for branch in self.workflow_branches
            for step in branch.steps
        ]

        for values, label in (
            (source_ids, "policy source"),
            (intent_ids, "intent"),
            (fact_ids, "fact"),
            (requirement_ids, "requirement"),
            (branch_ids, "workflow branch"),
            (step_ids, "workflow step"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} id")

        known_sources = set(source_ids)
        known_intents = set(intent_ids)
        known_facts = set(fact_ids)
        known_requirements = set(requirement_ids)
        for requirement in self.requirements:
            _validate_intent_refs(
                requirement.applies_to_intents,
                known_intents,
            )
            _validate_conditions(requirement.conditions, known_facts)
            if not set(requirement.policy_source_ids).issubset(known_sources):
                raise ValueError(
                    f"requirement {requirement.requirement_id!r} "
                    "has unknown policy source"
                )
        for branch in self.workflow_branches:
            _validate_intent_refs(branch.applies_to_intents, known_intents)
            _validate_conditions(branch.conditions, known_facts)
            for step in branch.steps:
                if step.requirement_id not in known_requirements:
                    raise ValueError(
                        f"step {step.step_id!r} references unknown requirement"
                    )
        return self


class ProfileSelection(ContractModel):
    call_id: str = Field(min_length=1)
    intent_ids: list[str] = Field(min_length=1)
    facts: dict[str, FactValue] = Field(default_factory=dict)
    selection_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_intent_ids(self):
        if len(self.intent_ids) != len(set(self.intent_ids)):
            raise ValueError("intent_ids must be unique")
        return self


class ResolvedRequirement(ContractModel):
    requirement_id: str
    title: str
    business_definition: str
    level: RequirementLevel
    category: str = Field(min_length=1, pattern=r"^[a-z0-9_.]+$")
    applicability_reason: str
    evidence_expectations: list[str]
    policy_source_ids: list[str]
    failure_finding_type: str | None = None
    positive_finding_type: str | None = None


class ResolvedWorkflowStep(ContractModel):
    step_id: str
    requirement_id: str
    branch_id: str
    sequence: int


class UnresolvedApplicability(ContractModel):
    subject_id: str
    missing_fact_ids: list[str]
    reason: str


class ResolvedDomainPlan(ContractModel):
    profile_id: str
    profile_version: str
    call_id: str
    selected_intent_ids: list[str]
    selected_branch_ids: list[str]
    requirements: list[ResolvedRequirement]
    workflow_steps: list[ResolvedWorkflowStep]
    unresolved_applicability: list[UnresolvedApplicability]


def _validate_intent_refs(values: list[str], known: set[str]):
    unknown = set(values) - known - {"*"}
    if unknown:
        raise ValueError(f"unknown intent references: {sorted(unknown)}")


def _validate_conditions(
    conditions: list[ApplicabilityCondition],
    known_facts: set[str],
):
    unknown = {condition.fact_id for condition in conditions} - known_facts
    if unknown:
        raise ValueError(f"unknown fact references: {sorted(unknown)}")


def load_domain_profile(path: str | Path) -> DomainProfile:
    path = Path(path)
    return DomainProfile.model_validate_json(path.read_text(encoding="utf-8"))


def load_banking_profile() -> DomainProfile:
    return load_domain_profile(PROFILE_DIR / "banking_v1.json")


def _validate_selection(
    profile: DomainProfile,
    selection: ProfileSelection,
):
    known_intents = {intent.intent_id for intent in profile.intents}
    unknown_intents = set(selection.intent_ids) - known_intents
    if unknown_intents:
        raise ValueError(f"unknown selected intents: {sorted(unknown_intents)}")

    definitions = {fact.fact_id: fact for fact in profile.facts}
    unknown_facts = set(selection.facts) - set(definitions)
    if unknown_facts:
        raise ValueError(f"unknown selected facts: {sorted(unknown_facts)}")

    for fact_id, value in selection.facts.items():
        definition = definitions[fact_id]
        if definition.value_type == FactValueType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"fact {fact_id!r} requires a boolean")
        elif not isinstance(value, str) or value not in definition.allowed_values:
            raise ValueError(
                f"fact {fact_id!r} must be one of "
                f"{definition.allowed_values}"
            )


def _condition_state(
    condition: ApplicabilityCondition,
    facts: dict[str, FactValue],
) -> bool | None:
    if condition.fact_id not in facts:
        return None
    actual = facts[condition.fact_id]
    if condition.operator == ConditionOperator.EQUALS:
        return actual == condition.value
    if condition.operator == ConditionOperator.NOT_EQUALS:
        return actual != condition.value
    if condition.operator == ConditionOperator.IN:
        return actual in (condition.value or [])
    if condition.operator == ConditionOperator.IS_TRUE:
        return actual is True
    if condition.operator == ConditionOperator.IS_FALSE:
        return actual is False
    raise ValueError(f"unsupported condition operator: {condition.operator}")


def _applicability(
    conditions: list[ApplicabilityCondition],
    facts: dict[str, FactValue],
) -> tuple[bool, list[str]]:
    states = [
        (condition, _condition_state(condition, facts))
        for condition in conditions
    ]
    if any(state is False for _, state in states):
        return False, []
    missing = [
        condition.fact_id
        for condition, state in states
        if state is None
    ]
    if missing:
        return False, missing
    return True, []


def _intent_match(applies_to: list[str], selected: set[str]) -> bool:
    return "*" in applies_to or bool(set(applies_to).intersection(selected))


def _resolved(
    requirement: RequirementDefinition,
    *,
    reason: str,
) -> ResolvedRequirement:
    return ResolvedRequirement(
        requirement_id=requirement.requirement_id,
        title=requirement.title,
        business_definition=requirement.business_definition,
        level=requirement.level,
        category=requirement.category,
        applicability_reason=reason,
        evidence_expectations=requirement.evidence_expectations,
        policy_source_ids=requirement.policy_source_ids,
        failure_finding_type=requirement.failure_finding_type,
        positive_finding_type=requirement.positive_finding_type,
    )


def resolve_domain_plan(
    profile: DomainProfile,
    selection: ProfileSelection,
) -> ResolvedDomainPlan:
    """Resolve applicable requirements without evaluating call performance."""
    _validate_selection(profile, selection)
    selected = set(selection.intent_ids)
    requirements = []
    workflow_steps = []
    selected_branches = []
    unresolved = []

    for requirement in profile.requirements:
        if not _intent_match(requirement.applies_to_intents, selected):
            continue
        applies, missing = _applicability(
            requirement.conditions,
            selection.facts,
        )
        if missing:
            unresolved.append(
                UnresolvedApplicability(
                    subject_id=requirement.requirement_id,
                    missing_fact_ids=missing,
                    reason="Required applicability facts were not selected.",
                )
            )
        elif applies:
            requirements.append(
                _resolved(
                    requirement,
                    reason=(
                        "Selected intent and applicability facts match "
                        "this requirement."
                    ),
                )
            )

    resolved_requirement_ids = {
        requirement.requirement_id
        for requirement in requirements
    }
    seen_steps = set()
    seen_workflow_requirements = set()
    for branch in profile.workflow_branches:
        if not _intent_match(branch.applies_to_intents, selected):
            continue
        applies, missing = _applicability(
            branch.conditions,
            selection.facts,
        )
        if missing:
            unresolved.append(
                UnresolvedApplicability(
                    subject_id=branch.branch_id,
                    missing_fact_ids=missing,
                    reason="Required branch-selection facts were not selected.",
                )
            )
            continue
        if not applies:
            continue

        selected_branches.append(branch.branch_id)
        for step in sorted(branch.steps, key=lambda item: item.sequence):
            if step.step_id in seen_steps:
                continue
            if step.requirement_id not in resolved_requirement_ids:
                continue
            if step.requirement_id in seen_workflow_requirements:
                continue
            seen_steps.add(step.step_id)
            seen_workflow_requirements.add(step.requirement_id)
            workflow_steps.append(
                ResolvedWorkflowStep(
                    step_id=step.step_id,
                    requirement_id=step.requirement_id,
                    branch_id=branch.branch_id,
                    sequence=step.sequence,
                )
            )

    workflow_steps.sort(key=lambda item: item.sequence)
    return ResolvedDomainPlan(
        profile_id=profile.profile_id,
        profile_version=profile.version,
        call_id=selection.call_id,
        selected_intent_ids=selection.intent_ids,
        selected_branch_ids=selected_branches,
        requirements=requirements,
        workflow_steps=workflow_steps,
        unresolved_applicability=unresolved,
    )
