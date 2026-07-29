# phase2_week4/day5_fastapi_agent.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/agent_db"
engine = create_engine(DATABASE_URL)

class Base(DeclarativeBase):
    pass

class Message(Base):
    __tablename__ = "fastapi_agent_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(engine)

# ── Memory helper ─────────────────────────────────────────────────────────
def save_message(session_id: str, role: str, content: str):
    with Session(engine) as session:
        session.add(Message(
            session_id=session_id,
            role=role,
            content=content
        ))
        session.commit()

def get_history(session_id: str, limit: int = 20) -> list:
    with Session(engine) as session:
        rows = session.query(Message)\
            .filter(Message.session_id == session_id)\
            .order_by(Message.created_at.asc())\
            .limit(limit).all()
    result = []
    for row in rows:
        if row.role == "user":
            result.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            result.append(AIMessage(content=row.content))
    return result

# ── LLM and tools ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

SYSTEM_PROMPT = """You are a helpful AI assistant for TechCorp.
Answer questions about shipping, returns, payments, and discounts.
Be concise and friendly. Remember the conversation history."""

@tool
def get_shipping_info(destination: str) -> str:
    """Get shipping time and cost for a destination.

    Args:
        destination: Destination region e.g. 'Europe' or 'international'
    """
    if "europe" in destination.lower():
        return "3-5 business days, costs 4.99 euros. Free on orders over 50 euros."
    elif "express" in destination.lower():
        return "1-2 business days, costs 9.99 euros."
    else:
        return "7-14 business days for international shipping."

@tool
def get_discount_info(discount_type: str) -> str:
    """Get information about available discounts.

    Args:
        discount_type: Type of discount e.g. 'student', 'bulk', 'newsletter'
    """
    discounts = {
        "student": "20% off with valid student ID",
        "bulk": "15% off on orders of 10 or more items",
        "newsletter": "10% off your first order when you subscribe",
    }
    key = discount_type.lower()
    return discounts.get(key, f"No specific discount found for: {discount_type}")

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_shipping_info, get_discount_info, get_current_time]
tools_dict = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ── Agent function ────────────────────────────────────────────────────────
def run_agent(message: str, session_id: str) -> str:
    """Run the agent with PostgreSQL memory."""
    save_message(session_id, "user", message)

    system = SystemMessage(content=SYSTEM_PROMPT)
    history = get_history(session_id)
    messages = [system] + history

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in tools_dict:
                result = tools_dict[tc["name"]].invoke(tc["args"])
                logger.info(f"Tool {tc['name']} → {result}")
                history.append(response)
                from langchain_core.messages import ToolMessage
                history.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))
        final = llm.invoke([system] + history)
        reply = final.content
    else:
        reply = response.content

    save_message(session_id, "assistant", reply)
    return reply

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="TechCorp AI Agent API",
    description="AI Agent with PostgreSQL memory, exposed via FastAPI",
    version="2.0.0"
)

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    message: str
    response: str
    timestamp: datetime
    status: str = "success"

class HistoryItem(BaseModel):
    role: str
    content: str
    created_at: datetime

class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryItem]
    count: int

@app.get("/")
async def health():
    return {"status": "running", "agent": "TechCorp AI v2.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the AI agent."""
    logger.info(f"Chat: session={request.session_id}, msg={request.message[:50]}")
    try:
        response = run_agent(request.message, request.session_id)
        return ChatResponse(
            session_id=request.session_id,
            message=request.message,
            response=response,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def history(session_id: str):
    """Get conversation history for a session."""
    with Session(engine) as session:
        rows = session.query(Message)\
            .filter(Message.session_id == session_id)\
            .order_by(Message.created_at.asc()).all()

    items = [HistoryItem(
        role=r.role,
        content=r.content,
        created_at=r.created_at
    ) for r in rows]

    return HistoryResponse(
        session_id=session_id,
        messages=items,
        count=len(items)
    )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear all messages for a session."""
    with Session(engine) as session:
        count = session.query(Message)\
            .filter(Message.session_id == session_id).delete()
        session.commit()
    return {"deleted": count, "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)