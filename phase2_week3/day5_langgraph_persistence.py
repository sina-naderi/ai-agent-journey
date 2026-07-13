# phase2_week3/day5_langgraph_persistence.py
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

# ── Define tools ─────────────────────────────────────────────────────────

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression to evaluate, e.g. '2 + 2'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

# ── Setup LangGraph agent with built-in memory ───────────────────────────
# MemorySaver stores conversation state in memory (RAM)
# In production, replace with SqliteSaver or PostgresSaver
checkpointer = MemorySaver()

# System prompt defines agent behavior
system_prompt = SystemMessage(content=(
    "You are a helpful AI assistant with access to tools. "
    "Remember everything the user tells you during this conversation. "
    "Be concise."
))

# create_react_agent with checkpointer = agent with persistent memory
# Each thread_id is an isolated conversation session
agent = create_react_agent(
    llm,
    tools=[get_current_time, calculate],
    checkpointer=checkpointer,
    prompt=system_prompt
)

# ── Helper function ───────────────────────────────────────────────────────
def chat(message: str, thread_id: str = "default") -> str:
    """
    Send a message to the agent with a specific thread (session).
    thread_id isolates different conversations from each other.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    return result["messages"][-1].content

# ── Test 1: Memory persists across turns ──────────────────────────────────
print("=" * 55)
print("Test 1: Memory across turns (thread: default)")
print(f"Bot: {chat('My name is Sina and I am learning AI agents.')}")
print(f"Bot: {chat('What is my name?')}")
print(f"Bot: {chat('What am I learning about?')}")

# ── Test 2: Tools work alongside memory ──────────────────────────────────
print("\n" + "=" * 55)
print("Test 2: Tools work within a memory session")
print(f"Bot: {chat('What time is it?')}")
print(f"Bot: {chat('What time did you just tell me?')}")  # Memory test

# ── Test 3: Thread isolation ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("Test 3: Thread isolation (different sessions)")
print(f"Thread A: {chat('My favorite framework is LangChain.', thread_id='user_A')}")
print(f"Thread B: {chat('What is my favorite framework?', thread_id='user_B')}")
# Thread B should NOT know about LangChain — different session
print(f"Thread A: {chat('What framework did I mention?', thread_id='user_A')}")
# Thread A should still remember LangChain

# ── Test 4: Calculator with memory context ────────────────────────────────
print("\n" + "=" * 55)
print("Test 4: Tool + memory together")
print(f"Bot: {chat('Calculate 123 * 456 for me.')}")
print(f"Bot: {chat('What was the result of that calculation?')}")  # Memory test