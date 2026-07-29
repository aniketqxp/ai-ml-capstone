import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .database import Base


class Agent(Base):
    __tablename__ = "agents"

    agent_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name       = Column(String(100), nullable=False)
    team       = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    calls      = relationship("Call", back_populates="agent")

class Call(Base):
    __tablename__ = "calls"

    call_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id         = Column(UUID(as_uuid=True), ForeignKey("agents.agent_id"), nullable=False)
    audio_path       = Column(String(500))
    call_date        = Column(DateTime, nullable=False)
    duration_seconds = Column(Integer)
    call_metadata    = Column(JSONB)
    created_at       = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="calls")
    jobs  = relationship("Job", back_populates="call")

class Job(Base):
    __tablename__ = "jobs"

    job_id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id    = Column(UUID(as_uuid=True), ForeignKey("calls.call_id"), nullable=False)
    status     = Column(String(20), default="queued")
    stage      = Column(String(50))
    error      = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    call = relationship("Call", back_populates="jobs")

class Transcript(Base):
    __tablename__ = "transcripts"

    transcript_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id        = Column(UUID(as_uuid=True), nullable=True)
    source_call_id = Column(String(100))
    turn_id        = Column(Integer, nullable=False)
    speaker        = Column(String(20))
    start_time     = Column(Float)
    end_time       = Column(Float)
    text           = Column(Text)
    avg_confidence = Column(Float)
    low_confidence = Column(Boolean, default=False)

class Evaluation(Base):
    __tablename__ = "evaluations"

    evaluation_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id                = Column(UUID(as_uuid=True), nullable=True)
    agent_id               = Column(UUID(as_uuid=True), ForeignKey("agents.agent_id"), nullable=False)
    overall_grade          = Column(String(1))
    weighted_score         = Column(Numeric(5, 2))
    scorecard              = Column(JSONB)
    compliance_flags       = Column(JSONB)
    escalation_risk        = Column(SmallInteger)
    escalation_flag        = Column(Boolean, default=False)
    coaching_required      = Column(Boolean, default=False)
    manual_review_required = Column(Boolean, default=False)
    llm_scored             = Column(Boolean, default=True)
    created_at             = Column(DateTime, default=datetime.utcnow)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    evaluation_run_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.job_id"))
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.call_id"))
    public_call_id = Column(String(100), nullable=False, index=True)
    runtime_run_id = Column(String(200), nullable=False)
    evaluator_version = Column(String(100), nullable=False)
    mode = Column(String(20), nullable=False)
    status = Column(String(50), nullable=False)
    decision_sha256 = Column(String(64))
    attention_required = Column(Boolean)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationFeedback(Base):
    __tablename__ = "evaluation_feedback"

    feedback_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    evaluation_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.evaluation_run_id"),
    )
    public_call_id = Column(String(100), nullable=False, index=True)
    decision_sha256 = Column(String(64), nullable=False)
    feedback_type = Column(String(40), nullable=False)
    finding_id = Column(String(200))
    action_type = Column(String(80))
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class SentimentSegment(Base):
    __tablename__ = "sentiment_segments"

    id                     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id                = Column(String(100), nullable=False)
    segment_index          = Column(Integer, nullable=False)
    segment_key            = Column(String(200))
    seq_id                 = Column(Integer)
    speaker                = Column(String(20))
    start_time             = Column(Float)
    end_time               = Column(Float)
    text                   = Column(Text)
    sentiment              = Column(String(20))
    dominant_emotion       = Column(String(50))
    escalation_score       = Column(Float)
    processing_status      = Column(String(30))
    audio_features         = Column(JSONB)
    explainability_flags   = Column(JSONB)
    escalation_explanation = Column(JSONB)
    has_audio_features     = Column(Boolean, default=False)
    audio_feature_version  = Column(String(100))
    domain                 = Column(String(50))
    model_version          = Column(String(100))
    created_at             = Column(DateTime, default=datetime.utcnow)

class CallAudioSummary(Base):
    __tablename__ = "call_audio_summaries"

    id                             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id                        = Column(String(100), nullable=False)
    domain                         = Column(String(50))
    model_version                  = Column(String(100))
    has_audio_features             = Column(Boolean, default=False)
    audio_feature_version          = Column(String(100))
    audio_feature_match_summary    = Column(JSONB)
    dashboard_audio_feature_series = Column(JSONB)
    call_summary                   = Column(JSONB)
    created_at                     = Column(DateTime, default=datetime.utcnow)
