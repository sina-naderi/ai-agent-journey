import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

# ── Session store — maps session_id to chat history ──────────────────────
# In production, replace InMemoryChatMessageHistory with
# PostgresChatMessageHistory or RedisChatMessageHistory
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ── Build chain with memory slot ─────────────────────────────────────────
# MessagesPlaceholder inserts the full history at this position in the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# Wrap chain so it automatically reads/writes history per session
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ── Chat helper ───────────────────────────────────────────────────────────
def chat(message: str, session_id: str = "default") -> str:
    """Send a message and maintain conversation history per session."""
    response = chain_with_memory.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

# ── Test 1: Memory persists within a session ──────────────────────────────
print("=" * 50)
print("Test 1: Memory within same session")
print(f"Bot: {chat('My name is Sina and I am learning AI agents.')}")
print(f"Bot: {chat('What is my name?')}")
print(f"Bot: {chat('What am I learning about?')}")

# ── Test 2: Sessions are isolated from each other ─────────────────────────
print("\n" + "=" * 50)
print("Test 2: Session isolation")
print(f"Session A: {chat('My favorite city is Amsterdam.', session_id='user_A')}")
# Session B has never been told about Amsterdam — should not know
print(f"Session B: {chat('What is my favorite city?', session_id='user_B')}")

# ── Test 3: Session A still has its own memory ────────────────────────────
print("\n" + "=" * 50)
print("Test 3: Session A still remembers")
print(f"Session A: {chat('What city did I mention?', session_id='user_A')}")

# ── Print message history for inspection ──────────────────────────────────
print("\n" + "=" * 50)
print("Message history for 'default' session:")
history = get_session_history("default")
for msg in history.messages:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{role}]: {msg.content[:70]}...")