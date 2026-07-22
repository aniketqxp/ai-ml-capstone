"""
Node functions for the graph-based evaluation pipeline.

Each node is a callable that takes EvalState and returns a partial state dict.
LLM nodes use a shared retry helper; deterministic nodes (assemble, aggregate,
anchor) do pure data transformation.
"""
import os
import json
import time

from pydantic import ValidationError

import paths
from env_util import load_env
from assemble import load_manifest, build_packet, assemble_turns, estimate_duration
from multimodal_context import build_multimodal_llm_context
from pydantic import BaseModel
from typing import List, Literal
try:
    from evaluation.decision_policy import evaluate_automation_policy
except ModuleNotFoundError:
    from decision_policy import evaluate_automation_policy

from rubric import (
    ComplianceChecklist,
    QualityDimensions,
    EscalationRisk,
    InvestigationReport,
    CallMetadata,
    CallEvaluation,
    RUBRIC_VERSION_GRAPH,
    CallWorkflow,
    WorkflowStep,
    Evidence,
    AIInsights,
    ContradictionFinding,
)
from prompts import (
    COMPLIANCE_SYSTEM,
    COMPLIANCE_SKELETON,
    QUALITY_SYSTEM,
    QUALITY_SKELETON,
    ESCALATION_SYSTEM,
    ESCALATION_SKELETON,
    INVESTIGATE_SYSTEM,
    INVESTIGATE_SKELETON,
    ARBITRATE_SYSTEM,
    ARBITRATE_SKELETON,
    SUBJECT_SYSTEM,
    SUBJECT_SKELETON,
    WORKFLOW_GEN_SYSTEM,
    WORKFLOW_GEN_SKELETON,
    WORKFLOW_CHECK_SYSTEM,
    WORKFLOW_CHECK_SKELETON,
    WORKFLOW_RECHECK_SYSTEM,
    WORKFLOW_RECHECK_SKELETON,
    AI_SUPERVISOR_SYSTEM_PROMPT,
    AI_SUPERVISOR_SKELETON,
    build_ai_supervisor_prompt,
)



load_env()
DATA = str(paths.NA_TESTSET)


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM call + retry helper
# ─────────────────────────────────────────────────────────────────────────────



def validate_ai_insights(
    insights: dict,
    evaluation: dict,
) -> dict:
    automation = insights.setdefault("automation", {})

    automation.setdefault("manager_review_required", False)
    automation.setdefault("customer_follow_up_required", False)
    automation.setdefault("compliance_alert_required", False)
    automation.setdefault("coaching_required", False)
    automation.setdefault("priority", "None")
    automation.setdefault("reasons", [])

    reasons = automation["reasons"]

    compliance = evaluation.get("compliance", {})

    mandatory_compliance_fields = [
        "name_announced",
        "company_announced",
        "recording_disclosure",
        "identity_verified",
        "resolution_provided",
    ]

    failed_controls = []

    for field_name in mandatory_compliance_fields:
        item = compliance.get(field_name) or {}

        if item.get("passed") is False:
            failed_controls.append(field_name)

    if failed_controls:
        automation["compliance_alert_required"] = True
        automation["coaching_required"] = True

        if automation.get("priority") in {None, "None", "Low"}:
            automation["priority"] = "High"

        reason = (
            "Failed compliance controls: "
            + ", ".join(name.replace("_", " ") for name in failed_controls)
        )

        if reason not in reasons:
            reasons.append(reason)

    business_risk = (
        insights.get("business_risk", {})
        .get("level", "Low")
    )

    if business_risk in {"High", "Critical"}:
        automation["manager_review_required"] = True

        reason = f"Business risk was assessed as {business_risk.lower()}."

        if reason not in reasons:
            reasons.append(reason)

    escalation = evaluation.get("escalation", {})
    escalation_level = escalation.get("risk_level", "none")

    if escalation_level == "escalate":
        automation["manager_review_required"] = True
        automation["priority"] = "Critical"

        reason = "The final escalation risk level is escalate."

        if reason not in reasons:
            reasons.append(reason)

    elif escalation_level == "review":
        automation["manager_review_required"] = True

        if automation.get("priority") in {None, "None", "Low"}:
            automation["priority"] = "Medium"

        reason = "The final escalation risk level requires manager review."

        if reason not in reasons:
            reasons.append(reason)

    quality = evaluation.get("quality", {})
    customer_satisfaction = quality.get("customer_satisfaction") or {}

    if customer_satisfaction.get("score", 5) <= 2:
        automation["customer_follow_up_required"] = True

        reason = "Customer satisfaction score was 2 or lower."

        if reason not in reasons:
            reasons.append(reason)

    if failed_controls and automation.get("priority") == "None":
        automation["priority"] = "Medium"

    acoustic = evaluation.get("_acoustic")

    if not acoustic:
        insights["audio_text_correlations"] = []

    return insights


def contradiction_check_node(state):
    """
    Detect contradictions between transcript, acoustic analysis,
    deterministic evaluation results, and final operational decisions.

    The first version performs deterministic checks. Semantic LLM checking
    will be added after the graph integration is working.
    """
    evaluation = dict(
        state.get("evaluation") or {}
    )

    contradictions = []

    # -------------------------------------------------------------
    # Check 1: compliance control passed without supporting evidence
    # -------------------------------------------------------------
    compliance = evaluation.get(
        "compliance",
        {},
    )

    for control_name, control in compliance.items():
        if not isinstance(control, dict):
            continue

        if (
            control.get("passed") is True
            and not control.get("evidence")
        ):
            contradictions.append({
                "contradiction_id": None,
                "category": "compliance",
                "title": (
                    "Compliance control passed without evidence"
                ),
                "description": (
                    f"The compliance control '{control_name}' was "
                    "marked as passed, but no supporting transcript "
                    "evidence was provided."
                ),
                "severity": "High",
                "status": "confirmed",
                "source_a": {
                    "source": "compliance",
                    "statement": (
                        f"{control_name} was marked as passed"
                    ),
                    "quote": None,
                    "speaker": None,
                    "timestamp": None,
                    "sec": None,
                },
                "source_b": {
                    "source": "transcript",
                    "statement": (
                        "No supporting evidence was attached"
                    ),
                    "quote": None,
                    "speaker": None,
                    "timestamp": None,
                    "sec": None,
                },
                "likely_explanation": (
                    "The compliance result may have been accepted "
                    "without an anchored transcript quote."
                ),
                "recommended_action": (
                    "Require verified transcript evidence before "
                    "accepting this compliance result."
                ),
                "affects_final_decision": True,
                "requires_confirmation": True,
                "confidence": 1.0,
            })

    
        # -------------------------------------------------------------
    # Check 2: text and acoustic escalation decisions disagree
    # -------------------------------------------------------------
    escalation = evaluation.get(
        "escalation",
        {},
    )

    hybrid = escalation.get(
        "hybrid",
        {},
    )

    text_risk = hybrid.get("text_risk")
    acoustic_risk = hybrid.get("acoustic_risk")
    final_risk = escalation.get("risk_level")

    if (
        text_risk
        and acoustic_risk
        and text_risk != acoustic_risk
    ):
        resolved = (
            hybrid.get("method") == "llm_arbitration"
            or hybrid.get("arbitration_rationale")
        )

        contradictions.append({
            "contradiction_id": None,
            "category": "escalation",
            "title": (
                "Transcript and acoustic escalation "
                "assessments disagree"
            ),
            "description": (
                f"The transcript-based escalation risk was "
                f"'{text_risk}', while the acoustic risk was "
                f"'{acoustic_risk}'. The final risk was "
                f"'{final_risk}'."
            ),
            "severity": (
                "Low" if resolved else "Medium"
            ),
            "status": (
                "resolved_by_fusion"
                if resolved
                else "possible"
            ),
            "source_a": {
                "source": "escalation",
                "statement": (
                    f"Transcript risk was {text_risk}"
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "acoustic_sentiment",
                "statement": (
                    f"Acoustic risk was {acoustic_risk}"
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                hybrid.get("arbitration_rationale")
                or (
                    "The spoken content and acoustic emotion "
                    "signals produced different risk levels."
                )
            ),
            "recommended_action": (
                "Use the final fused or arbitrated risk and "
                "review both channels when confidence is low."
            ),
            "affects_final_decision": not bool(resolved),
            "requires_confirmation": not bool(resolved),
            "confidence": 0.95,
        })

        # -------------------------------------------------------------
    # Check 3: raw acoustic sentiment differs from calibrated result
    # -------------------------------------------------------------
    sentiment = state.get("sentiment") or {}

    manager_review = sentiment.get(
        "manager_review_recommendation",
        {},
    )

    calibrated_sentiment = manager_review.get(
        "calibrated_customer_sentiment"
    )

    notes = manager_review.get("notes", [])

    raw_sentiment = None
    raw_emotion = None

    for note in notes:
        note_lower = str(note).lower()

        if (
            "raw model customer sentiment was"
            in note_lower
        ):
            raw_sentiment = (
                str(note).split(" was ", 1)[-1].strip()
            )

        if (
            "raw model customer dominant emotion was"
            in note_lower
        ):
            raw_emotion = (
                str(note).split(" was ", 1)[-1].strip()
            )

    if (
        raw_sentiment
        and calibrated_sentiment
        and raw_sentiment.lower()
        != calibrated_sentiment.lower()
    ):
        contradictions.append({
            "contradiction_id": None,
            "category": "sentiment",
            "title": (
                "Raw and calibrated customer sentiment differ"
            ),
            "description": (
                f"The raw acoustic model classified the "
                f"customer sentiment as '{raw_sentiment}', "
                f"while calibration changed it to "
                f"'{calibrated_sentiment}'."
            ),
            "severity": "Low",
            "status": "resolved_by_fusion",
            "source_a": {
                "source": "acoustic_sentiment",
                "statement": (
                    f"Raw customer sentiment was "
                    f"{raw_sentiment}"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "calibrated_sentiment",
                "statement": (
                    f"Calibrated customer sentiment was "
                    f"{calibrated_sentiment}"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                "Calibration reduced the influence of raw "
                "negative predictions because no strong "
                "negative or escalation indicators were found."
            ),
            "recommended_action": (
                "Use the calibrated sentiment for operational "
                "decisions while retaining the raw prediction "
                "for auditability."
            ),
            "affects_final_decision": False,
            "requires_confirmation": False,
            "confidence": 0.98,
        })    

        # -------------------------------------------------------------
    # Check 4: sentiment review recommendation conflicts with
    # the final escalation decision
    # -------------------------------------------------------------
    review_required = manager_review.get(
        "review_required"
    )

    review_level = manager_review.get(
        "review_level"
    )

    final_risk = (
        evaluation.get("escalation", {})
        .get("risk_level")
    )

    sentiment_requests_review = (
        review_required is True
        or review_level in {"medium", "high"}
    )

    final_requests_review = (
        final_risk in {"review", "escalate"}
    )

    if (
        review_required is not None
        and final_risk
        and sentiment_requests_review
        != final_requests_review
    ):
        contradictions.append({
            "contradiction_id": None,
            "category": "decision",
            "title": (
                "Sentiment review recommendation and final "
                "escalation decision disagree"
            ),
            "description": (
                f"The sentiment pipeline produced review level "
                f"'{review_level}' with review_required="
                f"{review_required}, while the final escalation "
                f"risk was '{final_risk}'."
            ),
            "severity": "Medium",
            "status": "possible",
            "source_a": {
                "source": "sentiment_manager_review",
                "statement": (
                    f"Sentiment review level was "
                    f"{review_level}; review_required="
                    f"{review_required}"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "final_escalation",
                "statement": (
                    f"Final escalation risk was {final_risk}"
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                "The sentiment recommendation and the final "
                "multimodal escalation logic used different "
                "signals or thresholds."
            ),
            "recommended_action": (
                "Review the fusion rationale and confirm that "
                "the final manager-review decision reflects all "
                "available sentiment and transcript evidence."
            ),
            "affects_final_decision": True,
            "requires_confirmation": True,
            "confidence": 0.95,
        })

        # -------------------------------------------------------------
    # Check 5: raw acoustic emotion differs from final text emotion
    # -------------------------------------------------------------
    final_text_emotion = (
        evaluation.get("escalation", {})
        .get("customer_emotion_text")
    )

    trajectory = sentiment.get(
        "temporal_emotion_trajectory",
        {},
    )

    end_state = trajectory.get("end_state")
    deescalation_detected = trajectory.get(
        "deescalation_detected",
        False,
    )

    if (
        raw_emotion
        and final_text_emotion
        and raw_emotion.lower()
        != final_text_emotion.lower()
    ):
        temporally_resolved = (
            deescalation_detected is True
            and end_state
            and end_state.lower()
            == final_text_emotion.lower()
        )

        contradictions.append({
            "contradiction_id": None,
            "category": "text_audio",
            "title": (
                "Acoustic and transcript-based customer "
                "emotion assessments differ"
            ),
            "description": (
                f"The raw acoustic model identified the "
                f"customer's dominant emotion as "
                f"'{raw_emotion}', while the transcript-based "
                f"evaluation described the customer as "
                f"'{final_text_emotion}'."
            ),
            "severity": (
                "Low"
                if temporally_resolved
                else "Medium"
            ),
            "status": (
                "resolved_by_fusion"
                if temporally_resolved
                else "possible"
            ),
            "source_a": {
                "source": "acoustic_sentiment",
                "statement": (
                    f"Raw acoustic emotion was "
                    f"{raw_emotion}"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "escalation",
                "statement": (
                    f"Transcript-based customer emotion was "
                    f"{final_text_emotion}"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                "The acoustic emotion summarizes emotional "
                "signals across the full call, while the final "
                "text assessment may reflect the customer's "
                "resolved end state."
                if temporally_resolved
                else
                "The acoustic and transcript channels produced "
                "different interpretations of the customer's "
                "emotional state."
            ),
            "recommended_action": (
                "Review the emotion trajectory and final call "
                "phase before using either emotion label as the "
                "sole operational decision."
            ),
            "affects_final_decision": not temporally_resolved,
            "requires_confirmation": not temporally_resolved,
            "confidence": 0.94,
        })

        # -------------------------------------------------------------
    # Check 6: high customer satisfaction conflicts with final
    # escalation or manager-review decision
    # -------------------------------------------------------------
    quality = evaluation.get(
        "quality",
        {},
    )

    satisfaction = quality.get(
        "customer_satisfaction",
        {},
    )

    satisfaction_score = satisfaction.get("score")

    final_risk = (
        evaluation.get("escalation", {})
        .get("risk_level")
    )

    if (
        isinstance(satisfaction_score, (int, float))
        and satisfaction_score >= 4
        and final_risk in {"review", "escalate"}
    ):
        trajectory = sentiment.get(
            "temporal_emotion_trajectory",
            {},
        )

        improved_resolution = (
            trajectory.get("overall_pattern")
            == "improved_resolution"
            and trajectory.get("unresolved_end_risk")
            is False
        )

        contradictions.append({
            "contradiction_id": None,
            "category": "quality",
            "title": (
                "High customer satisfaction conflicts with "
                "the final escalation decision"
            ),
            "description": (
                f"Customer satisfaction was scored "
                f"{satisfaction_score}/5, but the final "
                f"escalation risk was '{final_risk}'."
            ),
            "severity": (
                "Low"
                if improved_resolution
                else "Medium"
            ),
            "status": (
                "resolved_by_fusion"
                if improved_resolution
                else "possible"
            ),
            "source_a": {
                "source": "quality",
                "statement": (
                    f"Customer satisfaction score was "
                    f"{satisfaction_score}/5"
                ),
                "quote": None,
                "speaker": "CUSTOMER",
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "escalation",
                "statement": (
                    f"Final escalation risk was {final_risk}"
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                "The customer may have been satisfied with the "
                "resolution even though earlier portions of the "
                "call contained escalation indicators."
                if improved_resolution
                else
                "The quality and escalation assessments may be "
                "using inconsistent evidence or thresholds."
            ),
            "recommended_action": (
                "Review the customer satisfaction evidence, "
                "emotion trajectory, and escalation rationale "
                "before finalizing the manager-review decision."
            ),
            "affects_final_decision": not improved_resolution,
            "requires_confirmation": not improved_resolution,
            "confidence": 0.93,
        })

        # -------------------------------------------------------------
    # Check 7: high problem-resolution score conflicts with
    # incomplete required workflow steps
    # -------------------------------------------------------------
    workflow = evaluation.get(
        "workflow",
        {},
    )

    expected_steps = workflow.get(
        "expected_steps",
        [],
    )

    unmet_steps = [
        step
        for step in expected_steps
        if step.get("met") is False
    ]

    resolution_score = (
        evaluation.get("quality", {})
        .get("problem_resolution", {})
        .get("score")
    )

    if (
        isinstance(resolution_score, (int, float))
        and resolution_score >= 4
        and unmet_steps
    ):
        unmet_names = [
            step.get("step", "Unnamed workflow step")
            for step in unmet_steps
        ]

        contradictions.append({
            "contradiction_id": None,
            "category": "workflow",
            "title": (
                "High problem-resolution score conflicts with "
                "incomplete workflow steps"
            ),
            "description": (
                f"Problem resolution was scored "
                f"{resolution_score}/5, but "
                f"{len(unmet_steps)} required workflow "
                f"step(s) were not completed."
            ),
            "severity": "Medium",
            "status": "possible",
            "source_a": {
                "source": "quality",
                "statement": (
                    f"Problem resolution score was "
                    f"{resolution_score}/5"
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "source_b": {
                "source": "workflow",
                "statement": (
                    "Incomplete workflow steps: "
                    + "; ".join(unmet_names)
                ),
                "quote": None,
                "speaker": None,
                "timestamp": None,
                "sec": None,
            },
            "likely_explanation": (
                "The customer's immediate issue may have been "
                "resolved, while one or more required process "
                "steps were still omitted."
            ),
            "recommended_action": (
                "Review whether the missing workflow steps are "
                "mandatory controls or optional best practices, "
                "and adjust the final resolution assessment if "
                "needed."
            ),
            "affects_final_decision": True,
            "requires_confirmation": True,
            "confidence": 0.95,
        })
    
    # Assign stable IDs after all checks have run.
    for index, item in enumerate(
        contradictions,
        start=1,
    ):
        item["contradiction_id"] = (
            f"CONTRA-{index:03d}"
        )
    contradictions = [
            ContradictionFinding
            .model_validate(item)
            .model_dump(mode="json")
            for item in contradictions
    ]

    evaluation["contradictions"] = contradictions

    print(
        f"  [contradiction-check] found "
        f"{len(contradictions)} contradiction(s)"
    )

    return {
        "evaluation": evaluation,
        "contradictions": contradictions,
        "node_meta": {
            "contradiction_check": {
                "status": "success",
                "total_count": len(
                    contradictions
                ),
                "runs": _runs(
                    state,
                    "contradiction_check",
                ),
            }
        },
    }

def ai_supervisor_node(state):
    """
    Generate integrated, manager-ready insights from the finalized QA
    evaluation, transcript, workflow audit, and available acoustic results.
    """
    context = build_ai_insights_context(state)

    supervisor_packet = build_ai_supervisor_prompt(context)
    packet_chars = len(supervisor_packet)

    estimated_tokens = max(
        1,
        packet_chars // 4,
    )

    print(
        f"  [ai-supervisor] prompt size: "
        f"{packet_chars:,} chars "
        f"(~{estimated_tokens:,} tokens)"
    )

    evaluation = dict(
        state.get("evaluation") or {}
    )

    failed_controls = []

    compliance = evaluation.get(
        "compliance",
        {},
    )

    for control_name, control_data in compliance.items():
        if (
            isinstance(control_data, dict)
            and control_data.get("passed") is False
        ):
            failed_controls.append(control_name)

    escalation_level = (
        evaluation.get("escalation", {})
        .get("risk_level", "none")
    )

    customer_satisfaction_score = (
        evaluation.get("quality", {})
        .get("customer_satisfaction", {})
        .get("score")
    )

    contradictions = evaluation.get(
        "contradictions",
        [],
    )

    unresolved_contradictions = sum(
        1
        for contradiction in contradictions
        if (
            contradiction.get(
                "affects_final_decision"
            )
            is True
            and contradiction.get("status")
            not in {
                "resolved",
                "resolved_by_fusion",
            }
        )
    )

    automation_decisions = evaluate_automation_policy({
        "failed_compliance_controls": (
            failed_controls
        ),
        "escalation_level": escalation_level,
        "customer_satisfaction_score": (
            customer_satisfaction_score
        ),
        "unresolved_contradictions": (
            unresolved_contradictions
        ),
        "total_contradictions": len(
            contradictions
        ),
    })

    automation_decisions_dict = [
        decision.model_dump(mode="json")
        for decision in automation_decisions
    ]

    evaluation["automation_decisions"] = (
        automation_decisions_dict
    )

    try:
        result, dt, served = _llm_eval_with_retry(
            system=AI_SUPERVISOR_SYSTEM_PROMPT,
            skeleton=AI_SUPERVISOR_SKELETON,
            packet=supervisor_packet,
            validator=lambda d: AIInsights.model_validate(d),
        )

    except Exception as exc:

        evaluation["ai_insights"] = None

        evaluation["_multimodal_provenance"] = {
            "transcript_used": bool(
                state.get("turns")
            ),
            "sentiment_used": bool(
                state.get("sentiment")
            ),
            "sentiment_segment_count": len(
                (state.get("sentiment") or {}).get(
                    "segments",
                    [],
                )
            ),
            "data_availability": (
                state.get("data_availability")
                or {}
            ),
            "supervisor_model": None,
            "prompt_characters": packet_chars,
            "estimated_prompt_tokens": estimated_tokens,
            "supervisor_status": "failed",
            "error": (
                f"{type(exc).__name__}: {exc}"
            )[:1000],
        }

        print(
            f"  [ai-supervisor] unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "evaluation": evaluation,
            "ai_insights": None,
            "automation_decisions": automation_decisions_dict,
            "node_meta": {
                "ai_supervisor": {
                    "duration": 0.0,
                    "served_by": None,
                    "status": "failed",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    )[:1000],
                    "runs": _runs(
                        state,
                        "ai_supervisor",
                    ),
                }
            },
        }

    insights = validate_ai_insights(
        insights=result.model_dump(mode="json"),
        evaluation=state["evaluation"],
    )

    insights = validate_multimodal_ai_insights(
        insights=insights,
        state=state,
    )

    validated = AIInsights.model_validate(insights)
    validated.automation_decisions = (
        automation_decisions
    )

    

    insights_dict = validated.model_dump(mode="json")

    evaluation["ai_insights"] = insights_dict

    evaluation["automation_decisions"] = (
        automation_decisions_dict
    )

    evaluation["_multimodal_provenance"] = {
        "transcript_used": bool(
            state.get("turns")
        ),
        "sentiment_used": bool(
            state.get("sentiment")
        ),
        "sentiment_segment_count": len(
            (state.get("sentiment") or {}).get(
                "segments",
                [],
            )
        ),
        "data_availability": (
            state.get("data_availability")
            or {}
        ),
        "supervisor_model": served,
        "prompt_characters": packet_chars,
        "estimated_prompt_tokens": estimated_tokens,
        "supervisor_status": "success",
    }

    print(
        f"  [ai-supervisor] done in {dt:.1f}s via {served} "
        f"| risk={validated.business_risk.level} "
        f"| csat={validated.predicted_csat.score}/5"
    )

    return {
        "evaluation": evaluation,
        "ai_insights": insights_dict,
        "automation_decisions": (
            automation_decisions_dict
        ),
        "node_meta": {
            "ai_supervisor": {
                "duration": round(dt, 1),
                "served_by": served,
                "status": "success",
                "business_risk": validated.business_risk.level,
                "predicted_csat": validated.predicted_csat.score,
                "runs": _runs(
                    state,
                    "ai_supervisor",
                ),
            }
        },
    }

def build_ai_insights_context(state: dict) -> dict:
    """
    Build the complete multimodal input for the AI Supervisor.

    The AI Supervisor receives:
    - timestamped transcript
    - compliance results
    - quality results
    - workflow audit
    - escalation results
    - investigation results
    - complete acoustic sentiment intelligence
    """
    evaluation = state["evaluation"]

    transcript = []

    for turn in state.get("turns", []):
        transcript.append({
            "start": turn.get("start"),
            "end": turn.get("end"),
            "timestamp": turn.get("timestamp"),
            "speaker": turn.get("speaker"),
            "text": turn.get("text"),
        })

    context = build_multimodal_llm_context(
        call_id=state["call_id"],
        transcript=transcript,
        evaluation=evaluation,
        sentiment=state.get("sentiment", {}),
    )

    # Include graph-generated investigation separately because the
    # multimodal helper currently focuses on the core evaluation.
    context["deterministic_evaluation"]["investigation"] = (
        evaluation.get("investigation")
    )
    context["deterministic_evaluation"]["contradictions"] = (
        evaluation.get("contradictions", [])
    )

    context["metadata"] = state.get("meta", {})

    context["pipeline_data_availability"] = (
        state.get("data_availability", {})
    )

    return context
def _strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    return s.strip()


def validate_multimodal_ai_insights(
    insights: dict,
    state: dict,
) -> dict:
    """
    Apply deterministic safety and consistency checks to AI Supervisor output.
    """
    evaluation = state.get("evaluation", {})
    sentiment = state.get("sentiment", {})
    availability = state.get("data_availability", {})

    automation = insights.setdefault(
        "automation",
        {},
    )

    reasons = automation.setdefault(
        "reasons",
        [],
    )

    # Never allow audio claims when sentiment was unavailable.
    if not availability.get("has_sentiment"):
        insights["audio_text_correlations"] = []

    # Serious final escalation always requires review.
    risk_level = (
        evaluation.get("escalation", {})
        .get("risk_level", "none")
    )

    if risk_level == "escalate":
        automation["manager_review_required"] = True
        automation["priority"] = "Critical"

        reason = (
            "The final escalation risk level is escalate."
        )

        if reason not in reasons:
            reasons.append(reason)

    # Respect a strong deterministic manager-review recommendation.
    manager_review = sentiment.get(
        "manager_review_recommendation",
        {},
    )

    recommendation = str(
        manager_review.get(
            "recommendation",
            manager_review.get("decision", ""),
        )
    ).lower()

    if recommendation in {
        "review",
        "manager review",
        "manager_review",
        "required",
    }:
        automation["manager_review_required"] = True

        if automation.get("priority") in {
            None,
            "None",
            "Low",
        }:
            automation["priority"] = "Medium"

        reason = (
            "The sentiment pipeline recommended manager review."
        )

        if reason not in reasons:
            reasons.append(reason)

    # Failed mandatory controls require a compliance alert.
    compliance = evaluation.get(
        "compliance",
        {},
    )

    failed_controls = []

    for control_name, control in compliance.items():
        if not isinstance(control, dict):
            continue

        if control.get("passed") is False:
            failed_controls.append(control_name)

    if failed_controls:
        automation["compliance_alert_required"] = True

        reason = (
            "Compliance controls failed: "
            + ", ".join(failed_controls)
            + "."
        )

        if reason not in reasons:
            reasons.append(reason)

    return insights


def _feedback_text(quotes):
    """Build a revision note for a node whose cited quotes failed anchoring."""
    if not quotes:
        return None
    listed = "\n".join(f'- "{q}"' for q in quotes)
    return ("REVISION NOTE: in a previous evaluation of this call, the "
            "following cited quotes were NOT found verbatim in the "
            f"transcript:\n{listed}\n"
            "Re-evaluate. Every quote must be copied EXACTLY, character for "
            "character, from a single transcript turn. Do not paraphrase, "
            "do not merge text from multiple turns.")


# provider escalation order for eval nodes: a malformed/invalid 200-OK
# response is a model-quality failure the transport-level router never sees,
# so retrying the same provider on the same content fails identically. We
# escalate the PROVIDER per attempt instead -- a different model's JSON
# behaviour almost always parses where another's choked.
_TIER_SEQUENCE = ("qa-primary", "qa-safety", "qa-safety")


def _llm_eval_with_retry(system, skeleton, packet, validator, max_attempts=3,
                         feedback=None):
    """
    Call the LLM router, parse JSON, validate with `validator`, retry on
    failure with error feedback AND provider escalation. Returns
    (validated_model, dt, served_model). `feedback` is an optional revision
    note (e.g. failed anchor quotes) appended to the user prompt.
    """
    from router import chat_json_routed

    user = f"{packet}\n\n{skeleton}"
    if feedback:
        user += f"\n\n{feedback}"
    last_err = None

    for attempt in range(1, max_attempts + 1):
        tier = _TIER_SEQUENCE[min(attempt - 1, len(_TIER_SEQUENCE) - 1)]
        t0 = time.time()
        raw, served = chat_json_routed(system, user, return_meta=True, tier=tier)
        dt = time.time() - t0
        try:
            data = json.loads(_strip_fences(raw))
            result = validator(data)
            return result, dt, served
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            print(f"  [{system[:40]}...] attempt {attempt} ({tier}) failed "
                  f"({type(e).__name__}); escalating provider...")
            user = (f"{packet}\n\n{skeleton}"
                    + (f"\n\n{feedback}" if feedback else "")
                    + f"\n\nYour previous output was invalid: "
                    f"{str(e)[:400]}. Return corrected JSON only.")

    raise RuntimeError(f"Node failed after {max_attempts} attempts: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic nodes
# ─────────────────────────────────────────────────────────────────────────────

def _runs(state, node):
    """How many times `node` has executed in this invocation (1-based)."""
    return ((state.get("node_meta") or {}).get(node, {}).get("runs", 0)) + 1


def assemble_node(state):
    """
    Load all upstream pipeline artifacts required by the LangGraph workflow.

    Inputs loaded here:
    - transcription result
    - role-mapped transcript packet
    - chronological transcript turns
    - complete sentiment/audio payload, when available
    """
    from fusion import load_full_sentiment

    t0 = time.time()

    call_id = state["call_id"]
    results_dir = state.get("results_dir", "results_channels")

    manifest = load_manifest()
    meta = manifest[call_id]

    result_path = os.path.join(
        DATA,
        results_dir,
        meta["accent"],
        call_id + ".json",
    )

    with open(result_path, encoding="utf-8") as file:
        result = json.load(file)

    packet, stats = build_packet(result, meta)
    turns = assemble_turns(result)

    domain = result.get(
        "domain",
        meta.get("domain", "unknown"),
    )

    sentiment = load_full_sentiment(
        call_id=call_id,
        domain=domain,
    )

    cm = CallMetadata(
        call_id=call_id,
        domain=domain,
        accent=meta.get("accent"),
        duration_seconds=stats["duration_s"],
        transcript_model=result.get("model"),
    )

    data_availability = {
        "has_transcript": bool(turns),
        "has_sentiment": bool(sentiment),
        "has_sentiment_segments": bool(
            sentiment.get("segments")
        ),
        "has_audio_features": bool(
            sentiment.get("has_audio_features")
            or sentiment.get("audio_feature_summary")
        ),
        "has_calibrated_sentiment": bool(
            sentiment.get("calibrated_sentiment_summary")
        ),
        "has_multi_signal_escalation": bool(
            sentiment.get(
                "multi_signal_escalation_intelligence"
            )
        ),
        "has_temporal_emotion_trajectory": bool(
            sentiment.get("temporal_emotion_trajectory")
        ),
        "has_agent_empathy_tone_alignment": bool(
            sentiment.get("agent_empathy_tone_alignment")
        ),
    }

    print(
        f"  [assemble] transcript turns={len(turns)} | "
        f"sentiment={'available' if sentiment else 'unavailable'} | "
        f"audio_features={data_availability['has_audio_features']}"
    )

    return {
        "packet": packet,
        "meta": cm.model_dump(),
        "turns": turns,
        "transcript_result": result,
        "sentiment": sentiment,
        "data_availability": data_availability,
        "node_meta": {
            "assemble": {
                "duration": round(time.time() - t0, 2),
                "transcript_turns": len(turns),
                "sentiment_available": bool(sentiment),
                "sentiment_segments": len(
                    sentiment.get("segments", [])
                ),
                "data_availability": data_availability,
            }
        },
    }

def aggregate_node(state):
    """Merge parallel node outputs into a single CallEvaluation."""
    ev = CallEvaluation(
        rubric_version=RUBRIC_VERSION_GRAPH,
        metadata=CallMetadata(**state["meta"]),
        compliance=state["compliance"],
        quality=state["quality"],
        escalation=state["escalation"],
        workflow=state.get("workflow"),
        overall_summary=None,
    )
    return {"evaluation": ev.model_dump()}


def _collect_unanchored(node, out):
    """Recursively collect evidence quotes that anchoring could not locate."""
    if isinstance(node, dict):
        if "quote" in node and "speaker" in node and node.get("sec") is None:
            out.append(node["quote"])
        else:
            for v in node.values():
                _collect_unanchored(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_unanchored(v, out)


def anchor_node(state):
    """
    Attach real transcript timestamps to every evidence quote.
    Also reports WHICH sections cited quotes that could not be located, so
    the graph can route those sections back for re-evaluation (v0.3.0).
    """
    from eval_call_data import anchor_evidence

    ev_dict = state["evaluation"]
    if not isinstance(ev_dict, dict):
        ev_dict = ev_dict.model_dump()

    turns_indexed = [{"i": i, **t} for i, t in enumerate(state["turns"])]
    stats = {"total": 0, "anchored": 0}
    anchor_evidence(ev_dict, turns_indexed, stats)
    ev_dict["_anchor_stats"] = stats

    # per-section unanchored quotes -> feedback for the re-anchor loop
    feedback = {}
    for section in ("compliance", "quality", "escalation"):
        missing = []
        _collect_unanchored(ev_dict.get(section, {}), missing)
        if missing:
            feedback[section] = missing

    attempts = state.get("anchor_attempts", 0) + 1
    if feedback:
        n = sum(len(v) for v in feedback.values())
        print(f"  [anchor] pass {attempts}: {n} quote(s) not found verbatim "
              f"in sections: {', '.join(feedback)}")
    else:
        print(f"  [anchor] pass {attempts}: all quotes anchored")

    return {
        "evaluation": ev_dict,
        "anchor_attempts": attempts,
        "anchor_feedback": feedback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM nodes — each validates its own Pydantic fragment
# ─────────────────────────────────────────────────────────────────────────────

def _section_feedback(state, section):
    """Revision note if this section's quotes failed anchoring last pass."""
    fb = (state.get("anchor_feedback") or {}).get(section)
    if fb:
        print(f"  [{section}] re-running with {len(fb)} failed-quote note(s)")
    return _feedback_text(fb)


def compliance_node(state):
    """Evaluate 6 compliance checks."""
    result, dt, served = _llm_eval_with_retry(
        system=COMPLIANCE_SYSTEM,
        skeleton=COMPLIANCE_SKELETON,
        packet=state["packet"],
        validator=lambda d: ComplianceChecklist.model_validate(d),
        feedback=_section_feedback(state, "compliance"),
    )
    print(f"  [compliance] done in {dt:.1f}s via {served}")
    return {"compliance": result,
            "node_meta": {"compliance": {"duration": round(dt, 1),
                                         "served_by": served,
                                         "runs": _runs(state, "compliance")}}}


def quality_node(state):
    """Score 6 quality dimensions with enriched signal guidance."""
    result, dt, served = _llm_eval_with_retry(
        system=QUALITY_SYSTEM,
        skeleton=QUALITY_SKELETON,
        packet=state["packet"],
        validator=lambda d: QualityDimensions.model_validate(d),
        feedback=_section_feedback(state, "quality"),
    )
    print(f"  [quality] done in {dt:.1f}s via {served}")
    return {"quality": result,
            "node_meta": {"quality": {"duration": round(dt, 1),
                                      "served_by": served,
                                      "runs": _runs(state, "quality")}}}


def escalation_node(state):
    """Assess escalation risk: red flags, emotion, risk level."""
    result, dt, served = _llm_eval_with_retry(
        system=ESCALATION_SYSTEM,
        skeleton=ESCALATION_SKELETON,
        packet=state["packet"],
        validator=lambda d: EscalationRisk.model_validate(d),
        feedback=_section_feedback(state, "escalation"),
    )
    print(f"  [escalation] done in {dt:.1f}s via {served}")
    return {"escalation": result,
            "node_meta": {"escalation": {"duration": round(dt, 1),
                                         "served_by": served,
                                         "runs": _runs(state, "escalation")}}}


def investigate_node(state):
    """
    Conditional deep-dive (v0.3.0): runs ONLY when the anchored evaluation
    carries risk_level != "none". Produces a manager-ready incident report
    and attaches it to the evaluation.
    """
    ev = state["evaluation"]
    esc = ev["escalation"]
    context = (
        f"{state['packet']}\n\n"
        f"FIRST-PASS ESCALATION SCREENING RESULT:\n"
        f"{json.dumps({k: esc[k] for k in ('red_flags', 'customer_emotion_text', 'risk_level')}, indent=2)}"
    )
    report, dt, served = _llm_eval_with_retry(
        system=INVESTIGATE_SYSTEM,
        skeleton=INVESTIGATE_SKELETON,
        packet=context,
        validator=lambda d: InvestigationReport.model_validate(d),
    )
    print(f"  [investigate] done in {dt:.1f}s via {served} "
          f"(priority: {report.priority})")

    # anchor the report's own evidence quotes too
    from eval_call_data import anchor_evidence
    report_dict = report.model_dump()
    turns_indexed = [{"i": i, **t} for i, t in enumerate(state["turns"])]
    anchor_evidence(report_dict, turns_indexed, {"total": 0, "anchored": 0})

    ev["investigation"] = report_dict
    return {"evaluation": ev,
            "node_meta": {"investigate": {"duration": round(dt, 1),
                                          "served_by": served,
                                          "priority": report.priority}}}


# ── workflow track (v0.4.0) — three isolated LLM phases in one node ─────────

class _Subject(BaseModel):
    subject: str


class _StepsDraft(BaseModel):
    expected_steps: List[WorkflowStep]


def workflow_node(state):
    """
    Subject-derived expected workflow (v0.4.0). Three phases with strict
    prompt isolation:
      a. subject   — reads the transcript, states the REQUEST in one sentence
      b. expected  — sees ONLY domain + subject (never the transcript), lists
                     the boxes a competent agent should check for such a call
      c. check     — audits the transcript against that independent checklist
    The isolation in (b) is the point: expectations are formed before knowing
    what actually happened, so the audit cannot rationalize the observed call.
    """
    domain = state["meta"]["domain"]

    subj, dt_a, served_a = _llm_eval_with_retry(
        system=SUBJECT_SYSTEM, skeleton=SUBJECT_SKELETON,
        packet=state["packet"],
        validator=lambda d: _Subject.model_validate(d))
    print(f'  [workflow] subject in {dt_a:.1f}s: "{subj.subject}"')

    # phase b: NO transcript — only domain + subject
    gen_packet = f"DOMAIN: {domain}\nCALL SUBJECT: {subj.subject}"
    draft, dt_b, served_b = _llm_eval_with_retry(
        system=WORKFLOW_GEN_SYSTEM, skeleton=WORKFLOW_GEN_SKELETON,
        packet=gen_packet,
        validator=lambda d: _StepsDraft.model_validate(d))
    print(f"  [workflow] {len(draft.expected_steps)} expected steps in {dt_b:.1f}s")

    checklist = json.dumps(
        {"expected_steps": [{"step": s.step, "rationale": s.rationale}
                            for s in draft.expected_steps]}, indent=2)
    check_packet = (f"EXPECTED WORKFLOW (written without seeing this call):\n"
                    f"{checklist}\n\nTRANSCRIPT:\n{state['packet']}")
    # completeness is part of validity: a truncated audit (fewer entries than
    # the draft) would otherwise be silently padded with met=null at the join
    def _check_validator(d):
        r = _StepsDraft.model_validate(d)
        if len(r.expected_steps) != len(draft.expected_steps):
            raise ValueError(
                f"audit returned {len(r.expected_steps)} entries for "
                f"{len(draft.expected_steps)} steps; return one entry per "
                f"given step, same order")
        return r

    checked, dt_c, served_c = _llm_eval_with_retry(
        system=WORKFLOW_CHECK_SYSTEM, skeleton=WORKFLOW_CHECK_SKELETON,
        packet=check_packet, validator=_check_validator)

    # join by index against the DRAFT's step text — the draft is the contract;
    # the checker only contributes met/evidence. Missing entries stay unchecked.
    steps = []
    for i, ds in enumerate(draft.expected_steps):
        cs = checked.expected_steps[i] if i < len(checked.expected_steps) else None
        steps.append(WorkflowStep(
            step=ds.step, rationale=ds.rationale,
            met=cs.met if cs else None,
            evidence=cs.evidence if cs else None))

    # single-item recheck pass: fires ONLY on met=false steps, isolated from
    # the other steps so it can never perturb something the main pass already
    # got right. Catches the specific failure mode where the main pass wants
    # a direct question and misses that the value was established via a
    # later read-back/confirmation instead.
    def _recheck_validator(d):
        if "met" not in d:
            raise ValueError("missing met")
        return d

    dt_d = 0.0
    for s in steps:
        if s.met is not False:
            continue
        recheck_packet = (
            f"CHECKLIST ITEM: {s.step}\n"
            f"RATIONALE: {s.rationale}\n"
            f"FIRST-PASS VERDICT: met=false\n\n"
            f"TRANSCRIPT:\n{state['packet']}")
        try:
            rc, dt_r, _ = _llm_eval_with_retry(
                system=WORKFLOW_RECHECK_SYSTEM, skeleton=WORKFLOW_RECHECK_SKELETON,
                packet=recheck_packet, validator=_recheck_validator)
        except RuntimeError:
            continue
        dt_d += dt_r
        if rc.get("met") is True and rc.get("evidence"):
            s.met = True
            s.evidence = Evidence.model_validate(rc["evidence"])

    met = sum(1 for s in steps if s.met is True)
    missed = sum(1 for s in steps if s.met is False)
    print(f"  [workflow] audit in {dt_c:.1f}s: {met} met, {missed} missed, "
          f"{len(steps) - met - missed} n/a of {len(steps)}"
          + (f"  (+{dt_d:.1f}s recheck)" if dt_d else ""))

    dt = dt_a + dt_b + dt_c + dt_d
    return {"workflow": CallWorkflow(subject=subj.subject,
                                     expected_steps=steps).model_dump(),
            "node_meta": {"workflow": {"duration": round(dt, 1),
                                       "served_by": served_c,
                                       "phases": {"subject": round(dt_a, 1),
                                                  "expected": round(dt_b, 1),
                                                  "check": round(dt_c, 1)},
                                       "runs": _runs(state, "workflow")}}}


def fuse_node(state):
    """
    Deterministic acoustic-text fusion (v0.4.0). Loads the audio sentiment
    model's output (if any) and fuses it into the three audio-informed
    quality dimensions + the escalation risk level, attaching provenance.
    Runs between aggregate and anchor so the fused risk drives routing.
    """
    from fusion import load_acoustic, fuse_evaluation
    t0 = time.time()
    ev = state["evaluation"]
    rows = load_acoustic(state["call_id"], state["meta"]["domain"])
    summary = fuse_evaluation(ev, rows)
    if summary["acoustic_available"]:
        print(f"  [fuse] acoustic data found: fused "
              f"{', '.join(summary['fused_dims']) or 'nothing'}; "
              f"acoustic risk = {summary['acoustic_risk']}")
    else:
        print("  [fuse] no acoustic data for this call -> text-only scores")
    return {"evaluation": ev,
            "fusion_disputed": summary["disputed"],
            "node_meta": {"fuse": {"duration": round(time.time() - t0, 2),
                                   **summary}}}


class _Arbitration(BaseModel):
    risk_level: Literal["none", "review", "escalate"]
    rationale: str


def arbitrate_node(state):
    """
    Conditional arbitration (v0.4.1): runs ONLY when the text tier and the
    acoustic tier disagree about escalation risk. Deterministic merging is
    fine when the channels agree; when they disagree, a fixed rule has no
    basis to prefer one -- an LLM weighs both signals against the transcript
    (was the issue resolved? does vocal escalation RISE late in the call?).
    """
    ev = state["evaluation"]
    esc = ev["escalation"]
    h = esc.get("hybrid") or {}
    traj = (ev.get("_acoustic") or {}).get("trajectory") or []
    # compact trajectory line: customer escalation score per position bucket
    esc_series = " ".join(f"{t['p']:.2f}:{t['esc']:.2f}"
                          for t in traj if t.get("esc") is not None)

    context = (
        f"{state['packet']}\n\n"
        f"TEXT ASSESSMENT:\n"
        f"{json.dumps({k: esc[k] for k in ('red_flags', 'customer_emotion_text', 'risk_level')}, indent=2)}\n\n"
        f"ACOUSTIC ASSESSMENT (audio model, customer channel):\n"
        f"  tier: {h.get('acoustic_risk')}\n"
        f"  mean escalation over final third: {h.get('late_mean_escalation')}\n"
        f"  peak escalation: {h.get('peak_escalation')}\n"
        f"  trajectory (position:score, 0=start 1=end): {esc_series or 'n/a'}"
    )

    verdict, dt, served = _llm_eval_with_retry(
        system=ARBITRATE_SYSTEM, skeleton=ARBITRATE_SKELETON,
        packet=context,
        validator=lambda d: _Arbitration.model_validate(d))

    print(f"  [arbitrate] text={h.get('text_risk')} vs "
          f"acoustic={h.get('acoustic_risk')} -> {verdict.risk_level} "
          f"in {dt:.1f}s via {served}")

    esc["risk_level"] = verdict.risk_level
    esc["hybrid"] = {**h, "method": "llm_arbitration",
                     "arbitration_rationale": verdict.rationale}
    return {"evaluation": ev,
            "node_meta": {"arbitrate": {"duration": round(dt, 1),
                                        "served_by": served,
                                        "verdict": verdict.risk_level}}}


def chapter_node(state):
    """Segment the call into coherent chapters (reuses segment.py)."""
    from segment import segment
    cc, dt, issues = segment(
        state["call_id"],
        provider="router",
        results_dir=state.get("results_dir", "results_channels"),
    )
    if issues:
        print(f"  [chapters] {len(issues)} invariant issue(s): {issues}")
    print(f"  [chapters] {cc.n_chapters} chapters in {dt:.1f}s")
    return {"chapters": cc.model_dump(),
            "node_meta": {"chapters": {"duration": round(dt, 1),
                                       "served_by": "router",
                                       "runs": _runs(state, "chapters")}}}
