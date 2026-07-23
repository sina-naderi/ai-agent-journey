# phase2_week4/day4_agent_with_postgres.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

class Base(DeclarativeBase):
    pass

class Message(Base):
    """Stores all messages for persistent agent memory."""
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    tool_name = Column(String(100))  # filled only for tool messages
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(engine)
logger.info("Agent messages table ready")

# ── PostgreSQL Memory Manager ─────────────────────────────────────────────
class PostgresMemory:
    """
    Manages conversation history stored in PostgreSQL.
    Replaces LangGraph's MemorySaver with persistent storage.
    """

    def __init__(self, engine, session_id: str):
        self.engine = engine
        self.session_id = session_id

    def add_user_message(self, content: str):
        """Save a user message to the database."""
        self._save_message("user", content)

    def add_ai_message(self, content: str):
        """Save an AI response to the database."""
        self._save_message("assistant", content)

    def add_tool_result(self, tool_name: str, content: str):
        """Save a tool result to the database."""
        self._save_message("tool", content, tool_name=tool_name)

    def _save_message(self, role: str, content: str, tool_name: str = None):
        """Internal method to save any message type."""
        with Session(self.engine) as session:
            msg = Message(
                session_id=self.session_id,
                role=role,
                content=content,
                tool_name=tool_name
            )
            session.add(msg)
            session.commit()

    def get_messages(self, limit: int = 20) -> list:
        """
        Retrieve recent messages and convert to LangChain format.
        This is what gets sent to the LLM on every turn.
        """
        with Session(self.engine) as session:
            messages = session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .order_by(Message.created_at.asc())\
                .limit(limit)\
                .all()

        # Convert database rows to LangChain message objects
        result = []
        for msg in messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))
        return result

    def clear(self):
        """Delete all messages for this session."""
        with Session(self.engine) as session:
            session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .delete()
            session.commit()
        logger.info(f"Cleared memory for session: {self.session_id}")

    def count(self) -> int:
        """Count total messages in this session."""
        with Session(self.engine) as session:
            return session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .count()

# ── LLM and tools setup ───────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

SYSTEM_PROMPT = """You are a helpful AI assistant.
Remember everything the user tells you in this conversation.
Be concise and friendly."""

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression e.g. '2 + 2' or '15 * 4'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

tools = [get_current_time, calculate]
tools_dict = {t.name: t for t in tools}

# ── Agent with PostgreSQL memory ──────────────────────────────────────────
def chat_with_postgres_memory(message: str, session_id: str) -> str:
    """
    Chat function that uses PostgreSQL for persistent memory.
    Unlike MemorySaver, conversations survive server restarts.
    """
    memory = PostgresMemory(engine, session_id)

    # Save user message to database
    memory.add_user_message(message)

    # Build message list: system prompt + history + new message
    system = SystemMessage(content=SYSTEM_PROMPT)
    history = memory.get_messages()
    messages = [system] + history

    # Call LLM with tools
    response = llm.bind_tools(tools).invoke(messages)

    # Handle tool calls if any
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tools_dict:
                result = tools_dict[tool_name].invoke(tool_args)
                memory.add_tool_result(tool_name, str(result))
                logger.info(f"Tool {tool_name} → {result}")

        # Call LLM again with tool results for final answer
        updated_history = memory.get_messages()
        final_response = llm.invoke([system] + updated_history + [response])
        reply = final_response.content
    else:
        reply = response.content

    # Save AI response to database
    memory.add_ai_message(reply)
    logger.info(f"Session {session_id} now has {memory.count()} messages in DB")

    return reply

# ── Tests ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    session = "postgres_memory_test"

    print("=" * 55)
    print("Test 1: Memory persists in PostgreSQL")
    print(f"Bot: {chat_with_postgres_memory('My name is Sina.', session)}")
    print(f"Bot: {chat_with_postgres_memory('What is my name?', session)}")

    print("\n" + "=" * 55)
    print("Test 2: Tools work with PostgreSQL memory")
    print(f"Bot: {chat_with_postgres_memory('What time is it?', session)}")
    print(f"Bot: {chat_with_postgres_memory('What time did you just tell me?', session)}")

    print("\n" + "=" * 55)
    print("Test 3: Check database directly")
    memory = PostgresMemory(engine, session)
    print(f"Total messages in PostgreSQL: {memory.count()}")
    print("Messages in DB:")
    for msg in memory.get_messages():
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}]: {msg.content[:60]}...")