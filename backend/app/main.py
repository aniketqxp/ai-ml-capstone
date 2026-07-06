from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine
from app.models import Base
from app.routers import calls
from app.routers import sentiment

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI/ML Capstone API",
    description="Backend API orchestration layer and agent router",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calls.router)
app.include_router(sentiment.router)

@app.get("/")
def read_root():
    return {"status": "online", "service": "backend-orchestration"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "capstone_api"}