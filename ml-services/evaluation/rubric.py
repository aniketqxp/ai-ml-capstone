"""
Formal, versioned translation of the domain expert's QA rubric into a
machine-readable schema (Pydantic v2). This is the OUTPUT CONTRACT the LLM
evaluation layer must satisfy.

Design principle -- MODALITY SPLIT
----------------------------------
The expert's rubric mixes two kinds of judgment:
  * TEXT-derivable   : can be decided from the transcript alone.
  * AUDIO/paralinguistic: tone, pace, energy, "sounds satisfied" -- these
                          require the acoustic model (WP3) and must NOT be
                          guessed from text. They are represented here but
                          flagged `requires_audio=True` and left null by the
                          text LLM, to be filled by audio-text fusion (WBS 3.4).

Each leaf judgment carries EVIDENCE (a verbatim quote + speaker + timestamp) so
every score is provable -- no ungrounded compliance passes. This directly
supports the "Verbatim Evidence Extractor" track later.

RUBRIC_VERSION is bumped whenever criteria change so labelled data stays traceable.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field



class ExecutiveSummary(BaseModel):
    call_outcome: Literal[
        "Successful",
        "Partially successful",
        "Unsuccessful",
        "Unknown"
    ]
    resolution_status: Literal[
        "Resolved",
        "Partially resolved",
        "Unresolved",
        "Not applicable",
        "Unknown"
    ]
    overall_call_health: Literal[
        "Excellent",
        "Good",
        "Needs attention",
        "Critical",
        "Unknown"
    ]
    summary: str


class AutomationDecision(BaseModel):
    action_id: str
    action_type: Literal[
        "agent_coaching",
        "manager_review",
        "compliance_alert",
        "customer_follow_up",
        "case_creation",
        "no_action",
    ]

    decision: Literal[
        "approved",
        "requires_approval",
        "blocked",
    ]

    priority: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = "low"

    reason: str
    confidence: float = Field(ge=0.0, le=1.0)

    automation_allowed: bool = False
    requires_human_approval: bool = False

    rules_triggered: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)


class CustomerIntent(BaseModel):
    primary_intent: str
    secondary_intents: list[str] = Field(default_factory=list)
    customer_goal: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class RootCause(BaseModel):
    main_reason: str
    contributing_factors: list[str] = Field(default_factory=list)


class PredictedCSAT(BaseModel):
    score: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class BusinessRisk(BaseModel):
    level: Literal["Low", "Medium", "High", "Critical"]
    reasons: list[str] = Field(default_factory=list)


class NextBestAction(BaseModel):
    action: str
    priority: Literal["None", "Low", "Medium", "High", "Critical"]
    reason: str


class CoachingRecommendation(BaseModel):
    category: Literal[
        "Compliance",
        "Empathy",
        "Communication",
        "Efficiency",
        "Resolution",
        "Professionalism",
        "De-escalation",
        "Product knowledge",
        "Other"
    ]
    priority: Literal["Low", "Medium", "High", "Critical"]
    timestamp: Optional[str] = None
    observation: str
    recommendation: str
    suggested_phrase: Optional[str] = None


class AudioTextCorrelation(BaseModel):
    timestamp: Optional[str] = None
    speaker: Literal["agent", "customer", "unknown"]
    text_signal: str
    audio_signal: str
    interpretation: str
    recommended_response: Optional[str] = None


class ContradictionEvidence(BaseModel):
    source: Literal[
        "transcript",
        "acoustic_sentiment",
        "calibrated_sentiment",
        "compliance",
        "quality",
        "workflow",
        "escalation",
        "investigation",
        "automation",
        "ai_supervisor",
    ]

    statement: str

    quote: Optional[str] = None
    speaker: Optional[Literal["AGENT", "CUSTOMER"]] = None
    timestamp: Optional[str] = None
    sec: Optional[float] = None


class ContradictionFinding(BaseModel):
    contradiction_id: Optional[str] = None

    category: Literal[
        "text_audio",
        "sentiment",
        "compliance",
        "quality",
        "workflow",
        "resolution",
        "escalation",
        "decision",
        "evidence",
    ]

    title: str
    description: str

    severity: Literal["Low", "Medium", "High", "Critical"]

    status: Literal[
        "confirmed",
        "possible",
        "resolved_by_fusion",
    ]

    source_a: ContradictionEvidence
    source_b: ContradictionEvidence

    likely_explanation: Optional[str] = None
    recommended_action: Optional[str] = None

    affects_final_decision: bool = False
    requires_confirmation: bool = False

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class SupervisorDecisionFlags(BaseModel):
    manager_review_required: bool
    customer_follow_up_required: bool
    compliance_alert_required: bool
    coaching_required: bool
    priority: Literal["None", "Low", "Medium", "High", "Critical"]
    reasons: list[str] = Field(default_factory=list)


class AIInsights(BaseModel):
    executive_summary: ExecutiveSummary
    customer_intent: CustomerIntent
    root_cause: RootCause
    predicted_csat: PredictedCSAT
    business_risk: BusinessRisk
    next_best_action: NextBestAction
    coaching_recommendations: list[CoachingRecommendation] = Field(
        default_factory=list
    )
    audio_text_correlations: list[AudioTextCorrelation] = Field(
        default_factory=list
    )
    contradictions: list[ContradictionFinding] = Field(default_factory=list)
    automation_decisions: list[AutomationDecision] = Field(
        default_factory=list
    )
    automation: SupervisorDecisionFlags

RUBRIC_VERSION = "0.1.0"          # legacy monolithic path (extract.py)
RUBRIC_VERSION_GRAPH = "0.5.0"    # graph path: 0.2.0 added customer_satisfaction,
                                  # 0.3.0 added evidence re-anchor loop +
                                  # escalation investigation node,
                                  # 0.4.0 added acoustic-text fusion (hybrid
                                  # scores) + expected-workflow track,
                                  # 0.4.1 refit fusion thresholds on the
                                  # 22-call batch + arbitrate node on
                                  # text/acoustic disagreement

Speaker = Literal["AGENT", "CUSTOMER"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared evidence object -- makes every judgment provable
# ─────────────────────────────────────────────────────────────────────────────
class Evidence(BaseModel):
    quote: str = Field(description="Verbatim text copied from the transcript (no paraphrasing).")
    speaker: Speaker
    timestamp: Optional[str] = Field(default=None, description="mm:ss of the turn, if available.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPLIANCE  (all TEXT-derivable; binary pass/fail with N/A)
#    passed = True | False | None(=not applicable, e.g. recording disclosure
#    "when needed", or transfer steps when there was no transfer)
# ─────────────────────────────────────────────────────────────────────────────
class ComplianceItem(BaseModel):
    passed: Optional[bool] = Field(description="True=met, False=violated, None=not applicable to this call.")
    evidence: Optional[Evidence] = Field(default=None, description="Required when passed is True or False.")
    note: Optional[str] = None


class ComplianceChecklist(BaseModel):
    # "Announce their name"
    name_announced: ComplianceItem
    # "Announce company name"
    company_announced: ComplianceItem
    # "State 'this call may be recorded' (when needed)"  -> None if not needed
    recording_disclosure: ComplianceItem
    # "Confirm customer identity before discussing account" (last 4 / acct # / customer ID / DOB)
    identity_verified: ComplianceItem
    identity_method: Optional[str] = Field(
        default=None, description="e.g. 'last 4 digits', 'account number', 'date of birth', 'customer ID'.")
    # "Provide resolution / next steps before closing"
    resolution_provided: ComplianceItem
    # "If call transferred, provide information of next steps" -> None if no transfer
    transfer_next_steps: ComplianceItem


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUALITY  (1-5 each). Each dimension lists the concrete signals the rubric
#    names, split into text vs audio. The text LLM fills text signals + a score;
#    `requires_audio` flags dimensions whose FULL judgment also needs WP3.
# ─────────────────────────────────────────────────────────────────────────────
class HybridScore(BaseModel):
    """Provenance for a fused score (v0.4.0). method='text_only' means the
    acoustic model contributed nothing (no data / dimension is text-defined)."""
    text_score: int = Field(ge=1, le=5, description="LLM score from text signals alone.")
    acoustic_score: Optional[float] = Field(default=None, description="Acoustic subscore mapped to 1-5.")
    text_weight: float
    acoustic_weight: float
    coverage: Optional[float] = Field(
        default=None, description="Fraction of channel sentences the audio model processed.")
    channel: Optional[str] = Field(default=None, description="Speaker channel the acoustic signal came from.")
    method: Literal["weighted_mean", "text_only"] = "text_only"


class QualityDimension(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor, 5=excellent. Text LLM score; "
                       "overwritten with the fused score when acoustic data exists (v0.4.0).")
    signals_present: List[str] = Field(default_factory=list, description="Rubric behaviours observed.")
    signals_absent: List[str] = Field(default_factory=list, description="Expected behaviours not observed.")
    evidence: List[Evidence] = Field(default_factory=list)
    requires_audio: bool = Field(
        default=False, description="True if a complete judgment also needs acoustic signals (WP3).")
    hybrid: Optional[HybridScore] = Field(
        default=None, description="How this score was computed (v0.4.0+). Null in older evaluations.")


class QualityDimensions(BaseModel):
    # a. Efficiency: FCR, appropriate duration, unnecessary holds/transfers (text-derivable;
    #    duration is metadata)
    efficiency: QualityDimension
    # c. Problem Resolution: options A/B/C, pros/cons, explains WHY, respects choice, issue solved
    problem_resolution: QualityDimension
    # b. Clarity: open-ended questions, "does that make sense?", confirms understanding, recap
    clarity: QualityDimension
    # d. Professionalism: grammar/complete sentences/no slang (TEXT) + pace/confident voice (AUDIO)
    professionalism: QualityDimension   # requires_audio likely True
    # e. Empathy: validation phrases + name use (TEXT) + flat/robotic tone, matches emotion (AUDIO)
    empathy: QualityDimension           # requires_audio likely True
    # f. Customer Satisfaction: problem addressed, clear resolution, followed through (TEXT)
    #    + "sounds satisfied", not frustrated at end (AUDIO). Added in v0.2.0.
    customer_satisfaction: Optional[QualityDimension] = Field(
        default=None,
        description="Customer outcome signals. Added in v0.2.0; null in v0.1.0 evaluations."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESCALATION RISK
#    Red flags are mostly TEXT (phrases). Customer emotion is text-inferred here
#    but should be CONFIRMED by the audio model before it drives an action.
# ─────────────────────────────────────────────────────────────────────────────
RedFlag = Literal[
    "manager_requested",        # "I want to speak to a manager"
    "competitor_switch",        # "I'm switching to [competitor]"
    "repeat_attempts",          # "I've called a couple of times"
    "issue_too_complex",        # agent struggling / repeated holds
    "explicit_dissatisfaction", # other strong negative statements
]


class EscalationFusion(BaseModel):
    """Provenance for the fused risk level.
    v0.4.0 fused via ordinal max (more severe tier wins, "ordinal_max").
    v0.4.1: tiers that AGREE merge deterministically ("agreement"); tiers
    that DISAGREE are resolved by an arbitration LLM node that weighs both
    signals in context ("llm_arbitration") -- disagreement is exactly where
    mechanical merging has nothing to stand on (batch evidence: of 10 calls
    where either channel flagged risk, both flagged on only 1)."""
    text_risk: Literal["none", "review", "escalate"]
    acoustic_risk: Optional[Literal["none", "review", "escalate"]] = None
    late_mean_escalation: Optional[float] = Field(
        default=None, description="Mean customer escalation_score over the final third of the call.")
    peak_escalation: Optional[float] = Field(
        default=None, description="Max customer escalation_score anywhere in the call. Context only "
                                  "since v0.4.1 -- not a tier trigger (batch median 0.56 gave it no "
                                  "discriminating power).")
    method: Literal["agreement", "llm_arbitration", "ordinal_max", "text_only"] = "text_only"
    arbitration_rationale: Optional[str] = Field(
        default=None, description="Arbitrator's reasoning when method='llm_arbitration'.")


class EscalationRisk(BaseModel):
    red_flags: List[RedFlag] = Field(default_factory=list)
    customer_emotion_text: Literal["calm", "mild_frustration", "frustrated", "angry", "distressed"] = Field(
        description="Emotion inferred from TEXT only; audio model confirms before action.")
    risk_level: Literal["none", "review", "escalate"] = Field(
        description="none / review (manager may review) / escalate (immediate). "
                    "Fused with the acoustic tier when audio data exists (v0.4.0).")
    evidence: List[Evidence] = Field(default_factory=list)
    requires_audio: bool = Field(
        default=True, description="Emotion intensity needs acoustic confirmation.")
    hybrid: Optional[EscalationFusion] = Field(
        default=None, description="How risk_level was computed (v0.4.0+). Null in older evaluations.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. INVESTIGATION (v0.3.0) -- produced ONLY when risk_level != "none".
#    A conditional deep-dive that turns escalation red flags into a
#    manager-ready incident report.
# ─────────────────────────────────────────────────────────────────────────────
class InvestigationReport(BaseModel):
    summary: str = Field(description="2-3 sentence incident summary for a manager.")
    contributing_factors: List[str] = Field(
        default_factory=list, description="What led to the escalation risk (agent + customer side).")
    recommended_action: str = Field(description="Concrete next step for the reviewing manager.")
    priority: Literal["low", "medium", "high"] = Field(
        description="Review urgency. escalate risk -> high, review risk -> low/medium.")
    evidence: List[Evidence] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPECTED WORKFLOW (v0.4.0) -- three isolated LLM phases:
#    a. subject: one sentence stating what the customer contacted about
#       (the REQUEST only, never the outcome).
#    b. expected steps: generated from domain + subject ONLY -- the model has
#       not seen the transcript, so expectations cannot be contaminated by
#       what actually happened on the call.
#    c. check: audits the transcript against that independent checklist.
# ─────────────────────────────────────────────────────────────────────────────
class WorkflowStep(BaseModel):
    step: str = Field(description="Short imperative phrase, e.g. 'Verify customer identity'.")
    rationale: Optional[str] = Field(default=None, description="Why this step is expected for this call type.")
    met: Optional[bool] = Field(
        default=None, description="True=performed, False=missed, None=not determinable/applicable. "
                                  "Filled by the check phase; null before checking.")
    evidence: Optional[Evidence] = None


class CallWorkflow(BaseModel):
    subject: str = Field(description="One-sentence gist of what the customer wanted (request, not outcome).")
    expected_steps: List[WorkflowStep] = Field(
        description="4-8 boxes a competent agent should check for a call with this subject.")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level evaluation payload
# ─────────────────────────────────────────────────────────────────────────────
class CallMetadata(BaseModel):
    call_id: str
    domain: str                       # injected as "Call Type" lens
    accent: Optional[str] = None
    duration_seconds: Optional[float] = None
    transcript_model: Optional[str] = None


class CallEvaluation(BaseModel):
    rubric_version: str = RUBRIC_VERSION
    metadata: CallMetadata
    compliance: ComplianceChecklist
    quality: QualityDimensions
    escalation: EscalationRisk

    investigation: Optional[InvestigationReport] = Field(
        default=None,
        description=(
            "Deep-dive incident report. Only populated when "
            "risk_level != 'none' (v0.3.0+)."
        ),
    )

    workflow: Optional[CallWorkflow] = Field(
        default=None,
        description=(
            "Subject-derived expected workflow and audit results "
            "(v0.4.0+)."
        ),
    )

    overall_summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence plain-language summary for the dashboard.",
    )

    ai_insights: Optional[AIInsights] = Field(
        default=None,
        description=(
            "Integrated AI supervisor insights generated from the transcript, "
            "QA results, workflow audit, and available acoustic sentiment."
        ),
    )

# Documentation map: which rubric areas are text vs audio vs metadata.
# Used by the prompt builder (Phase 1) and the report.
MODALITY_MAP = {
    "compliance.name_announced":        "text",
    "compliance.company_announced":     "text",
    "compliance.recording_disclosure":  "text",
    "compliance.identity_verified":     "text",
    "compliance.resolution_provided":   "text",
    "compliance.transfer_next_steps":   "text",
    "quality.efficiency":               "text+metadata",  # duration from metadata
    "quality.problem_resolution":       "text",
    "quality.clarity":                  "text",
    "quality.professionalism":          "text+audio",     # grammar=text, tone/pace=audio
    "quality.empathy":                  "text+audio",     # phrases=text, warmth/tone=audio
    "escalation.red_flags":             "text",
    "escalation.customer_emotion":      "text+audio",      # words=text, intensity=audio
    "quality.customer_satisfaction":    "text+audio",      # outcome=text, "sounds satisfied"=audio
}


if __name__ == "__main__":
    # emit the JSON schema so we can eyeball the contract
    import json
    print(f"RUBRIC_VERSION = {RUBRIC_VERSION}\n")
    print(json.dumps(CallEvaluation.model_json_schema(), indent=2)[:1500], "...")
