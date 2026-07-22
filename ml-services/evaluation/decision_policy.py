"""
Deterministic automation decision policy.

These rules convert finalized QA evaluation results into safe,
auditable operational recommendations.
"""

try:
    from evaluation.rubric import AutomationDecision
except ModuleNotFoundError:
    from rubric import AutomationDecision


_PRIORITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _higher_priority(
    current: str,
    incoming: str,
) -> str:
    """
    Return the higher of two priority levels.
    """
    if (
        _PRIORITY_ORDER[incoming]
        > _PRIORITY_ORDER[current]
    ):
        return incoming

    return current


def _add_or_merge_decision(
    decisions: list[AutomationDecision],
    *,
    action_id: str,
    action_type: str,
    decision: str,
    priority: str,
    reason: str,
    confidence: float,
    automation_allowed: bool,
    requires_human_approval: bool,
    rule_triggered: str,
    blocked_reason: str | None = None,
) -> None:
    """
    Add a decision or merge it with an existing decision
    of the same action type.

    This prevents several manager-review cards from being
    generated for the same call.
    """
    existing = next(
        (
            item
            for item in decisions
            if item.action_type == action_type
        ),
        None,
    )

    if existing is None:
        decisions.append(
            AutomationDecision(
                action_id=action_id,
                action_type=action_type,
                decision=decision,
                priority=priority,
                reason=reason,
                confidence=confidence,
                automation_allowed=automation_allowed,
                requires_human_approval=(
                    requires_human_approval
                ),
                rules_triggered=[rule_triggered],
                blocked_reasons=(
                    [blocked_reason]
                    if blocked_reason
                    else []
                ),
            )
        )
        return

    existing.priority = _higher_priority(
        existing.priority,
        priority,
    )

    existing.confidence = max(
        existing.confidence,
        confidence,
    )

    if reason not in existing.reason:
        existing.reason = (
            f"{existing.reason} {reason}"
        )

    if rule_triggered not in existing.rules_triggered:
        existing.rules_triggered.append(
            rule_triggered
        )

    if (
        blocked_reason
        and blocked_reason
        not in existing.blocked_reasons
    ):
        existing.blocked_reasons.append(
            blocked_reason
        )

    if decision == "blocked":
        existing.decision = "blocked"
        existing.automation_allowed = False
        existing.requires_human_approval = True

    elif (
        decision == "requires_approval"
        and existing.decision != "blocked"
    ):
        existing.decision = "requires_approval"
        existing.automation_allowed = False
        existing.requires_human_approval = True


def evaluate_automation_policy(
    context: dict,
) -> list[AutomationDecision]:
    """
    Evaluate deterministic automation rules.

    Expected context fields:
    - failed_compliance_controls
    - escalation_level
    - customer_satisfaction_score
    - unresolved_contradictions
    - total_contradictions
    """
    decisions: list[AutomationDecision] = []

    failed_controls = list(
        context.get(
            "failed_compliance_controls",
            [],
        )
        or []
    )

    escalation_level = str(
        context.get(
            "escalation_level",
            "none",
        )
        or "none"
    ).lower()

    satisfaction_score = context.get(
        "customer_satisfaction_score"
    )

    unresolved_contradictions = int(
        context.get(
            "unresolved_contradictions",
            0,
        )
        or 0
    )

    total_contradictions = int(
        context.get(
            "total_contradictions",
            0,
        )
        or 0
    )

    # ---------------------------------------------------------
    # Rule 1:
    # Missing recording disclosure -> agent coaching
    # ---------------------------------------------------------
    if "recording_disclosure" in failed_controls:
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-001",
            action_type="agent_coaching",
            decision="approved",
            priority="high",
            reason=(
                "Recording disclosure was not provided."
            ),
            confidence=0.98,
            automation_allowed=True,
            requires_human_approval=False,
            rule_triggered=(
                "RECORDING_DISCLOSURE_FAILURE"
            ),
        )

    # ---------------------------------------------------------
    # Rule 2:
    # Any compliance failure -> compliance alert
    # ---------------------------------------------------------
    if failed_controls:
        controls_text = ", ".join(
            control.replace("_", " ")
            for control in failed_controls
        )

        _add_or_merge_decision(
            decisions,
            action_id="ACTION-002",
            action_type="compliance_alert",
            decision="requires_approval",
            priority="high",
            reason=(
                "Compliance controls failed: "
                f"{controls_text}."
            ),
            confidence=0.97,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "COMPLIANCE_FAILURE_CONFIRMED"
            ),
        )

    # ---------------------------------------------------------
    # Rule 3:
    # Multiple compliance failures -> manager review
    # ---------------------------------------------------------
    if len(failed_controls) >= 2:
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-003",
            action_type="manager_review",
            decision="requires_approval",
            priority="high",
            reason=(
                f"{len(failed_controls)} compliance "
                "controls failed."
            ),
            confidence=0.97,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "MULTIPLE_COMPLIANCE_FAILURES"
            ),
        )

    # ---------------------------------------------------------
    # Rule 4:
    # Review-level escalation -> manager review
    # ---------------------------------------------------------
    if escalation_level == "review":
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-004",
            action_type="manager_review",
            decision="requires_approval",
            priority="medium",
            reason=(
                "The final escalation risk requires review."
            ),
            confidence=0.95,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "ESCALATION_REVIEW_REQUIRED"
            ),
        )

    # ---------------------------------------------------------
    # Rule 5:
    # Escalate-level risk -> critical manager review
    # ---------------------------------------------------------
    if escalation_level == "escalate":
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-005",
            action_type="manager_review",
            decision="requires_approval",
            priority="critical",
            reason=(
                "The final escalation risk is escalate."
            ),
            confidence=0.99,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "CRITICAL_ESCALATION_DETECTED"
            ),
        )

    # ---------------------------------------------------------
    # Rule 6:
    # Low customer satisfaction -> follow-up
    # ---------------------------------------------------------
    if (
        isinstance(
            satisfaction_score,
            (int, float),
        )
        and satisfaction_score <= 2
    ):
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-006",
            action_type="customer_follow_up",
            decision="requires_approval",
            priority="high",
            reason=(
                "Customer satisfaction was scored "
                f"{satisfaction_score}/5."
            ),
            confidence=0.94,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "LOW_CUSTOMER_SATISFACTION"
            ),
        )

    # ---------------------------------------------------------
    # Rule 7:
    # Unresolved contradictions -> block automation
    # and create a review case
    # ---------------------------------------------------------
    if unresolved_contradictions > 0:
        _add_or_merge_decision(
            decisions,
            action_id="ACTION-007",
            action_type="case_creation",
            decision="blocked",
            priority="high",
            reason=(
                f"{unresolved_contradictions} unresolved "
                "contradiction(s) may affect the final "
                "decision."
            ),
            confidence=0.96,
            automation_allowed=False,
            requires_human_approval=True,
            rule_triggered=(
                "UNRESOLVED_CONTRADICTION"
            ),
            blocked_reason=(
                "Automatic operational actions are blocked "
                "until the contradiction is reviewed."
            ),
        )

    # ---------------------------------------------------------
    # Rule 8:
    # No actionable issue -> no action
    # ---------------------------------------------------------
    if not decisions:
        decisions.append(
            AutomationDecision(
                action_id="ACTION-008",
                action_type="no_action",
                decision="approved",
                priority="low",
                reason=(
                    "No deterministic rule requiring an "
                    "operational action was triggered."
                ),
                confidence=0.95,
                automation_allowed=True,
                requires_human_approval=False,
                rules_triggered=[
                    "NO_ACTIONABLE_RISK"
                ],
                blocked_reasons=[],
            )
        )

    return decisions