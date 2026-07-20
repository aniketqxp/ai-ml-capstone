import os
from sqlalchemy import create_engine
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
