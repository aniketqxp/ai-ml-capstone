"""
Deterministic action-routing: read an Evaluation's scores, set its flags.

Lives here (not in the FastAPI router) so both the request handler and the
background worker can call it without the worker importing the web layer.
"""


def apply_action_routing(evaluation, max_escalation_score=None):
    """Set escalation/coaching/manual-review flags from the scores in place."""
    # text escalation risk from the LLM (0-10 scale)
    if evaluation.escalation_risk is not None:
        if evaluation.escalation_risk >= 7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif evaluation.escalation_risk >= 4:
            evaluation.coaching_required = True

    # grade-based routing
    if evaluation.overall_grade == "F":
        evaluation.manual_review_required = True
        evaluation.coaching_required = True

    # acoustic escalation (wav2vec2 max segment score, 0.0-1.0)
    if max_escalation_score is not None:
        if max_escalation_score >= 0.7:
            evaluation.escalation_flag = True
            evaluation.coaching_required = True
        elif max_escalation_score >= 0.4:
            evaluation.coaching_required = True

    return evaluation
