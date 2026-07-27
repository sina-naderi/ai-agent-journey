# phase2_week4/day4_agent_with_postgres.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/agent_db"
engine = create_engine(DATABASE_URL, echo=False)

class Base(DeclarativeBase):
    pass

class Message(Base):
    """Stores all messages for persistent agent memory."""
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tool_name = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(engine)
logger.info("Agent messages table ready")

# ── PostgreSQL Memory Manager ─────────────────────────────────────────────
class PostgresMemory:
    """Manages conversation history stored in PostgreSQL."""

    def __init__(self, engine, session_id: str):
        self.engine = engine
        self.session_id = session_id

    def add_user_message(self, content: str):
        self._save("user", content)

    def add_ai_message(self, content: str):
        self._save("assistant", content)

    def add_tool_result(self, tool_name: str, content: str):
        self._save("tool", content, tool_name=tool_name)

    def _save(self, role: str, content: str, tool_name: str = None):
        with Session(self.engine) as session:
            session.add(Message(
                session_id=self.session_id,
                role=role,
                content=content,
                tool_name=tool_name
            ))
            session.commit()

    def get_messages(self, limit: int = 20) -> list:
        """Get recent messages as LangChain message objects."""
        with Session(self.engine) as session:
            rows = session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .order_by(Message.created_at.asc())\
                .limit(limit).all()

        result = []
        for msg in rows:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))
            # tool messages are NOT included here
            # they're only used in the same turn they're created
        return result

    def clear(self):
        with Session(self.engine) as session:
            session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .delete()
            session.commit()

    def count(self) -> int:
        with Session(self.engine) as session:
            return session.query(Message)\
                .filter(Message.session_id == self.session_id)\
                .count()

# ── LLM and tools ─────────────────────────────────────────────────────────
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

# ── Agent با PostgreSQL memory ────────────────────────────────────────────
def chat_with_postgres_memory(message: str, session_id: str) -> str:
    """
    Chat with agent using PostgreSQL for persistent memory.

    KEY CHANGE from original:
    When tool calls happen, we pass tool results directly in the same
    request instead of reloading from DB. This is required by the API —
    an assistant message with tool_calls MUST be immediately followed
    by tool messages with matching tool_call_ids.
    """
    memory = PostgresMemory(engine, session_id)

    # Save user message
    memory.add_user_message(message)

    # Build messages: system + history from DB
    system = SystemMessage(content=SYSTEM_PROMPT)
    history = memory.get_messages()
    messages = [system] + history

    # First LLM call
    response = llm.bind_tools(tools).invoke(messages)

    if response.tool_calls:
        # ── CHANGED: build tool results in memory, not from DB ──────────
        # The API requires this exact sequence:
        # [...messages, assistant_with_tool_calls, tool_result_1, tool_result_2]
        tool_messages = []

        for tool_call in response.tool_calls:
            try:
                if tool_call["name"] in tools_dict:
                    result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
                    logger.info(f"Tool {tool_call['name']} → {result}")
                else:
                    result = f"Unknown tool: {tool_call['name']}"
            except Exception as e:
                result = f"Tool error: {e}"
                logger.error(f"Tool {tool_call['name']} failed: {e}")

            # Save to DB for logging
            memory.add_tool_result(tool_call["name"], str(result))

            # Build ToolMessage for the API
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]  # must match the tool_call_id!
            ))

        # Second LLM call with correct sequence:
        # system + history + assistant_response + tool_results
        # ── NOT: system + updated_history (this breaks the sequence!) ──
        final_response = llm.invoke(
            [system] + history + [response] + tool_messages
        )
        reply = final_response.content

    else:
        reply = response.content

    # Save AI reply to DB
    memory.add_ai_message(reply)
    logger.info(f"Session {session_id}: {memory.count()} messages in DB")
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