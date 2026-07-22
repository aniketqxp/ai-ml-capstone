import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Agent

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/", status_code=201)
async def create_agent(
    payload: dict,
    db: Session = Depends(get_db)
):
    name = payload.get("name")
    team = payload.get("team")

    if not name:
        return {"error": "name is required"}

    agent = Agent(
        name=name,
        team=team,
        created_at=datetime.utcnow()
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    return {
        "agent_id": str(agent.agent_id),
        "name": agent.name,
        "team": agent.team,
        "created_at": agent.created_at.isoformat()
    }

@router.get("/")
def get_all_agents(db: Session = Depends(get_db)):
    agents = db.query(Agent).all()
    return [
        {
            "agent_id": str(a.agent_id),
            "name": a.name,
            "team": a.team,
            "created_at": a.created_at.isoformat() if a.created_at else None
        }
        for a in agents
    ]

@router.get("/{agent_id}")
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(
        Agent.agent_id == uuid.UUID(agent_id)
    ).first()

    if not agent:
        return {"error": "agent not found"}

    return {
        "agent_id": str(agent.agent_id),
        "name": agent.name,
        "team": agent.team,
        "created_at": agent.created_at.isoformat() if agent.created_at else None
    }