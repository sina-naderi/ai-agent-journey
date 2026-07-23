# phase2_week4/day2_fastapi_postgres.py
import os
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(
    title="AI Agent API",
    description="REST API for the TechCorp AI Agent with PostgreSQL storage",
    version="1.0.0"
)

# ── Database ──────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "agent_db",
    "user": "postgres",
    "password": "postgres"
}

def get_db():
    """Get a database connection."""
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """Initialize database tables on startup."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            user_message TEXT NOT NULL,
            assistant_message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database initialized")

# ── Pydantic Models ───────────────────────────────────────────────────────
# Pydantic validates request/response data automatically
# If required field is missing → FastAPI returns 422 automatically

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(..., min_length=1, max_length=100,
                            description="Unique session identifier")
    message: str = Field(..., min_length=1,
                         description="User message")
    language: str = Field(default="en",
                          description="Response language: 'en' or 'fa'")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    session_id: str
    user_message: str
    assistant_message: str
    timestamp: datetime
    status: str = "success"

class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    user_message: str
    assistant_message: str
    created_at: datetime

class HistoryResponse(BaseModel):
    """Response model for history endpoint."""
    session_id: str
    turns: list[ConversationTurn]
    total_count: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    timestamp: datetime

# ── Startup event ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Initialize database when the server starts."""
    init_db()
    logger.info("API server started")

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check — verify API and database are running."""
    try:
        conn = get_db()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return HealthResponse(
        status="running",
        database=db_status,
        timestamp=datetime.now()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get an AI response.
    Conversation is saved to PostgreSQL automatically.
    """
    logger.info(f"Chat request: session={request.session_id}")

    # Simple mock response — in Day 6 we'll connect the real agent
    if "shipping" in request.message.lower():
        response = "Standard shipping takes 3-5 business days within Europe."
    elif "return" in request.message.lower():
        response = "You can return products within 30 days with original receipt."
    elif "discount" in request.message.lower():
        response = "We offer a 20% student discount with valid student ID."
    else:
        response = "I can help with shipping, returns, and discounts. What would you like to know?"

    # Save to PostgreSQL
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO conversations
               (session_id, user_message, assistant_message)
               VALUES (%s, %s, %s)
               RETURNING created_at""",
            (request.session_id, request.message, response)
        )
        created_at = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return ChatResponse(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=response,
        timestamp=created_at
    )

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str, limit: int = 20):
    """Get conversation history for a session."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT user_message, assistant_message, created_at
               FROM conversations
               WHERE session_id = %s
               ORDER BY created_at ASC
               LIMIT %s""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    turns = [ConversationTurn(**row) for row in rows]

    return HistoryResponse(
        session_id=session_id,
        turns=turns,
        total_count=len(turns)
    )

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear all conversation history for a session."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM conversations WHERE session_id = %s",
            (session_id,)
        )
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Deleted {deleted} messages for session {session_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)