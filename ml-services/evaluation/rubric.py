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
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

RUBRIC_VERSION = "0.1.0"

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
class QualityDimension(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor, 5=excellent, judged from TEXT signals only.")
    signals_present: List[str] = Field(default_factory=list, description="Rubric behaviours observed.")
    signals_absent: List[str] = Field(default_factory=list, description="Expected behaviours not observed.")
    evidence: List[Evidence] = Field(default_factory=list)
    requires_audio: bool = Field(
        default=False, description="True if a complete judgment also needs acoustic signals (WP3).")


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


class EscalationRisk(BaseModel):
    red_flags: List[RedFlag] = Field(default_factory=list)
    customer_emotion_text: Literal["calm", "mild_frustration", "frustrated", "angry", "distressed"] = Field(
        description="Emotion inferred from TEXT only; audio model confirms before action.")
    risk_level: Literal["none", "review", "escalate"] = Field(
        description="none / review (manager may review) / escalate (immediate).")
    evidence: List[Evidence] = Field(default_factory=list)
    requires_audio: bool = Field(
        default=True, description="Emotion intensity needs acoustic confirmation.")


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
    overall_summary: Optional[str] = Field(
        default=None, description="2-3 sentence plain-language summary for the dashboard.")


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
}


if __name__ == "__main__":
    # emit the JSON schema so we can eyeball the contract
    import json
    print(f"RUBRIC_VERSION = {RUBRIC_VERSION}\n")
    print(json.dumps(CallEvaluation.model_json_schema(), indent=2)[:1500], "...")
