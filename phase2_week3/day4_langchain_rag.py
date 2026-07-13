import os
import logging
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Initialize models ────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.3  # Lower temperature for factual Q&A
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Load or create ChromaDB collection ───────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
CHROMA_PATH = os.path.join(PARENT_DIR, "phase2_week2", "chroma_db_qa_bot")

# chroma_client = chromadb.PersistentClient(path="./phase2_week2/chroma_db_qa_bot")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = chroma_client.get_collection("qa_bot_docs")
    logger.info(f"Loaded existing collection with {collection.count()} docs")
except Exception:
    logger.info("Collection not found — creating new one...")
    collection = chroma_client.get_or_create_collection("qa_bot_docs")
    documents = [
        "Standard shipping takes 3-5 business days within Europe and costs 4.99 euros",
        "Express shipping takes 1-2 business days and costs 9.99 euros",
        "Free shipping is available for orders over 50 euros",
        "Return policy allows returns within 30 days with original receipt",
        "We accept Visa, Mastercard, PayPal, Apple Pay, and Google Pay",
        "All products come with a standard 1-year warranty covering manufacturing defects",
        "Warranty does not cover physical damage, water damage, or modifications",
        "Support team available Monday to Friday 9am to 6pm CET",
        "Email support at support@techcorp.com",
        "Bulk orders of 10 or more items receive a 15% discount automatically",
        "Student discount of 20% available with valid student ID",
    ]
    embeddings = embedding_model.encode(documents).tolist()
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    logger.info(f"Created collection with {len(documents)} documents")

# ── Define RAG as a LangChain tool ───────────────────────────────────────
# The agent decides WHEN to use this tool based on the docstring
# This is more efficient than always searching the knowledge base

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information about
    shipping, returns, payment, warranty, support, or discounts.

    Use this tool whenever the user asks about company policies or services.

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
    return "\n".join([f"- {doc}" for doc in docs])

@tool
def get_contact_info() -> str:
    """Get TechCorp contact information for direct customer support."""
    return (
        "Email: support@techcorp.com | "
        "Phone: +49-30-123-4567 | "
        "Hours: Monday-Friday 9am-6pm CET"
    )

# ── Create agent with RAG tools ───────────────────────────────────────────
tools = [search_knowledge_base, get_contact_info]
agent = create_agent(llm, tools)

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for TechCorp. "
    "Use search_knowledge_base to find accurate information before answering. "
    "If the answer is not in the knowledge base, say so honestly. "
    "Be concise and friendly."
)

def ask(question: str) -> str:
    """Ask the agent a customer support question."""
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    })
    return result["messages"][-1].content

# ── Tests ─────────────────────────────────────────────────────────────────
questions = [
    "How long does shipping take to Germany?",
    "Can I return a product after 3 weeks?",
    "Is there a student discount?",
    "How do I contact support?",
    "What payment methods do you accept?",
    "What is the weather in Amsterdam?",  # Not in KB — should say so
]

for q in questions:
    print("=" * 50)
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print()