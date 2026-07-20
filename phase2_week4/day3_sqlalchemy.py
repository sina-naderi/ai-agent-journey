# phase2_week4/day3_sqlalchemy.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Database setup ────────────────────────────────────────────────────────
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/agent_db"
engine = create_engine(DATABASE_URL, echo=False)

# ── Define models (tables as Python classes) ──────────────────────────────
class Base(DeclarativeBase):
    pass

class Conversation(Base):
    """Maps to conversations table in PostgreSQL."""
    __tablename__ = "conversations_orm"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Conversation(id={self.id}, session={self.session_id})>"

class AgentAction(Base):
    """Maps to agent_actions table — logs every tool call."""
    __tablename__ = "agent_actions_orm"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    tool_name = Column(String(100), nullable=False)
    tool_input = Column(Text)
    tool_output = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

# Create all tables automatically
Base.metadata.create_all(engine)
logger.info("Tables created via SQLAlchemy ORM")

# ── Repository pattern — encapsulates database operations ─────────────────
# This pattern separates database logic from business logic
# Makes testing easier — you can mock the repository

class ConversationRepository:
    """Handles all database operations for conversations."""

    def __init__(self, engine):
        self.engine = engine

    def save(self, session_id: str, user_msg: str, assistant_msg: str) -> Conversation:
        """Save a new conversation turn."""
        with Session(self.engine) as session:
            conv = Conversation(
                session_id=session_id,
                user_message=user_msg,
                assistant_message=assistant_msg
            )
            session.add(conv)
            session.commit()
            session.refresh(conv)
            logger.info(f"Saved conversation id={conv.id}")
            return conv

    def get_history(self, session_id: str, limit: int = 10) -> list[Conversation]:
        """Get recent conversation history for a session."""
        with Session(self.engine) as session:
            return session.query(Conversation)\
                .filter(Conversation.session_id == session_id)\
                .order_by(Conversation.created_at.asc())\
                .limit(limit)\
                .all()

    def count_messages(self, session_id: str) -> int:
        """Count total messages in a session."""
        with Session(self.engine) as session:
            return session.query(Conversation)\
                .filter(Conversation.session_id == session_id)\
                .count()

    def delete_session(self, session_id: str) -> int:
        """Delete all conversations for a session. Returns deleted count."""
        with Session(self.engine) as session:
            count = session.query(Conversation)\
                .filter(Conversation.session_id == session_id)\
                .delete()
            session.commit()
            return count

class ActionRepository:
    """Handles all database operations for agent actions."""

    def __init__(self, engine):
        self.engine = engine

    def log(self, session_id: str, tool_name: str,
            tool_input: str, tool_output: str) -> AgentAction:
        """Log a tool call."""
        with Session(self.engine) as session:
            action = AgentAction(
                session_id=session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output
            )
            session.add(action)
            session.commit()
            session.refresh(action)
            return action

    def get_tool_stats(self, session_id: str) -> dict:
        """Get tool usage stats for a session."""
        with Session(self.engine) as session:
            actions = session.query(AgentAction)\
                .filter(AgentAction.session_id == session_id)\
                .all()
            stats = {}
            for action in actions:
                stats[action.tool_name] = stats.get(action.tool_name, 0) + 1
            return stats

# ── Tests ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    conv_repo = ConversationRepository(engine)
    action_repo = ActionRepository(engine)

    session_id = "orm_test_001"

    # Save conversations
    logger.info("Saving test conversations...")
    conv_repo.save(session_id, "What is the return policy?",
                   "You can return products within 30 days.")
    conv_repo.save(session_id, "Is there free shipping?",
                   "Free shipping on orders over 50 euros.")
    conv_repo.save(session_id, "Student discount?",
                   "Yes, 20% off with valid student ID.")

    # Log tool actions
    action_repo.log(session_id, "search_knowledge_base",
                    "return policy", "30 days with receipt")
    action_repo.log(session_id, "search_knowledge_base",
                    "free shipping", "orders over 50 euros")
    action_repo.log(session_id, "get_current_time", "", "2026-07-14 10:00:00")

    # Retrieve and display
    print("\n── Conversation History (via ORM) ──")
    history = conv_repo.get_history(session_id)
    for turn in history:
        print(f"[{turn.created_at}] User: {turn.user_message}")
        print(f"              Bot:  {turn.assistant_message}\n")

    print(f"Total messages: {conv_repo.count_messages(session_id)}")

    print("\n── Tool Usage Stats ──")
    stats = action_repo.get_tool_stats(session_id)
    for tool, count in stats.items():
        print(f"  {tool}: {count} calls")