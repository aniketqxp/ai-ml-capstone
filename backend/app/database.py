import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# On Hugging Face Spaces DATABASE_URL MUST be the Supabase SESSION-POOLER
# (IPv4) URI -- the direct db host is IPv6-only and silently times out.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/capstone")

# pool_pre_ping: the Space sleeps after ~48h idle; on wake every pooled
# connection is long dead. pre_ping issues SELECT 1 before checkout and
# reconnects transparently instead of 500-ing the first request.
# pool_recycle keeps connections under the pooler's server-side idle timeout.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Create tables if the database is reachable.

    Import-time table creation would hard-crash the whole app when the DB is
    unreachable (e.g. a local smoke import with no Postgres). Callers invoke
    this at startup and may tolerate failure in dev.
    """
    Base.metadata.create_all(bind=engine)
    # create_all intentionally does not alter existing tables. Keep this small,
    # additive compatibility migration here so long-lived Docker/Supabase
    # databases created by an older branch gain the runtime-ingest lookup key.
    with engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE transcripts "
            "ADD COLUMN IF NOT EXISTS source_call_id VARCHAR(100)"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_transcripts_source_call_id "
            "ON transcripts (source_call_id)"
        ))
        for definition in (
            "progress_percent INTEGER DEFAULT 0",
            "progress_current INTEGER",
            "progress_total INTEGER",
            "progress_message VARCHAR(200)",
            "estimated_seconds_remaining INTEGER",
            "cancel_requested BOOLEAN DEFAULT FALSE",
            "started_at TIMESTAMP WITHOUT TIME ZONE",
        ):
            connection.execute(text(
                f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {definition}"
            ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
