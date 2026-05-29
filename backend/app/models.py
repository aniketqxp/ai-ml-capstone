from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class CallRecord(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    transcripts = relationship("TranscriptRecord", back_populates="call")
    evidence = relationship("EvidenceRecord", back_populates="call")

class TranscriptRecord(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    speaker = Column(String)
    text = Column(String)
    start_time = Column(Integer) # offset in milliseconds
    end_time = Column(Integer)

    call = relationship("CallRecord", back_populates="transcripts")

class EvidenceRecord(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    category = Column(String)
    content = Column(JSON) # Structured compliance indicators

    call = relationship("CallRecord", back_populates="evidence")
