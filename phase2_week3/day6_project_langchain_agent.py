# phase2_week3/day6_project_langchain_agent.py
import os
import logging
import chromadb
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Initialize models ─────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Models loaded.")

# ── Setup ChromaDB knowledge base ─────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(PARENT_DIR, "phase2_week2", "chroma_db_qa_bot")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# chroma_client = chromadb.PersistentClient(path="./phase2_week2/chroma_db_qa_bot")

try:
    collection = chroma_client.get_collection("qa_bot_docs")
    logger.info(f"Knowledge base loaded: {collection.count()} documents")
except Exception:
    logger.info("Creating new knowledge base...")
    collection = chroma_client.get_or_create_collection("qa_bot_docs")
    documents = [
        "Standard shipping takes 3-5 business days within Europe and costs 4.99 euros",
        "Express shipping takes 1-2 business days and costs 9.99 euros",
        "Free shipping is available for orders over 50 euros",
        "Return policy allows returns within 30 days with original receipt",
        "Products must be in original packaging for returns",
        "Refunds are processed within 5-7 business days",
        "We accept Visa, Mastercard, PayPal, Apple Pay, and Google Pay",
        "All products come with a standard 1-year warranty",
        "Warranty covers manufacturing defects but not physical damage",
        "Support team available Monday to Friday 9am to 6pm CET",
        "Email support at support@techcorp.com",
        "Phone support at +49-30-123-4567",
        "Bulk orders of 10 or more items receive a 15% discount automatically",
        "Student discount of 20% available with valid student ID",
        "Newsletter subscribers get 10% off their first order",
    ]
    embeddings = embedding_model.encode(documents).tolist()
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    logger.info(f"Knowledge base created with {len(documents)} documents")

# ── Define tools ──────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information about
    shipping, returns, payment, warranty, support, or discounts.

    Use this when the user asks about company policies, services, or products.

    Args:
        query: The search query to find relevant information
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    docs = results["documents"][0]
    if not docs:
        return "No relevant information found in the knowledge base."
    context = "\n".join([f"- {doc}" for doc in docs])
    logger.info(f"Found {len(docs)} relevant documents for: {query}")
    return context

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression to evaluate, e.g. '2 + 2' or '15 * 4 / 2'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# ── Build agent with LangGraph memory ────────────────────────────────────
checkpointer = MemorySaver()

system_prompt = SystemMessage(content=(
    "You are a helpful AI assistant for TechCorp customers and developers.\n\n"
    "You have access to these tools:\n"
    "- search_knowledge_base: use for ANY question about shipping, returns, "
    "payment, warranty, discounts, or support — always search before answering\n"
    "- get_current_time: use when asked about the current time or date\n"
    "- calculate: use for any math calculation\n\n"
    "Rules:\n"
    "- Always search the knowledge base before answering company-related questions\n"
    "- If information is not in the knowledge base, say so honestly\n"
    "- Remember everything the user tells you in this conversation\n"
    "- Answer in the same language the user writes in\n"
    "- Be concise and friendly"
))

agent = create_react_agent(
    llm,
    tools=[search_knowledge_base, get_current_time, calculate],
    checkpointer=checkpointer,
    prompt=system_prompt
)

# ── Chat function ─────────────────────────────────────────────────────────
def chat(message: str, thread_id: str = "main") -> str:
    """Send a message and get a response with persistent memory."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )
    return result["messages"][-1].content

# ── CLI Interface ─────────────────────────────────────────────────────────
def main():
    print("╔═══════════════════════════════════════════════╗")
    print("║   TechCorp AI Assistant 🤖                    ║")
    print("║   Powered by LangChain + RAG + LangGraph      ║")
    print("╠═══════════════════════════════════════════════╣")
    print("║  Tools: knowledge base, calculator, time       ║")
    print("║  Memory: remembers your conversation           ║")
    print("║  Commands: 'quit' to exit                      ║")
    print("╚═══════════════════════════════════════════════╝\n")

    thread_id = f"session_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"Session started: {thread_id}")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response = chat(user_input, thread_id=thread_id)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()