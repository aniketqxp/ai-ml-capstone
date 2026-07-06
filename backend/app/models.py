import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, SmallInteger, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB
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

    agent       = relationship("Agent", back_populates="calls")
    jobs        = relationship("Job", back_populates="call")
    evaluations = relationship("Evaluation", back_populates="call")

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
    call_id                = Column(UUID(as_uuid=True), ForeignKey("calls.call_id"), nullable=False)
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

    call  = relationship("Call", back_populates="evaluations")

class SentimentSegment(Base):
    __tablename__ = "sentiment_segments"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id           = Column(String(100), nullable=False)
    segment_index     = Column(Integer, nullable=False)
    segment_key       = Column(String(200))
    seq_id            = Column(Integer)
    sentiment         = Column(String(20))
    dominant_emotion  = Column(String(50))
    escalation_score  = Column(Float)
    processing_status = Column(String(30))
    domain            = Column(String(50))
    model_version     = Column(String(100))
    created_at        = Column(DateTime, default=datetime.utcnow)
    