import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)

# ── Setup Chroma with categorized documents ──────────────────────────────

chroma_client = chromadb.PersistentClient(path="./chroma_db_v3")
collection = chroma_client.get_or_create_collection(name="categorized_docs")

# Documents with metadata categories
documents = [
    # Shipping category
    ("Standard shipping takes 3-5 business days within Europe", {"category": "shipping", "priority": "high"}),
    ("Express shipping takes 1-2 business days and costs €9.99", {"category": "shipping", "priority": "high"}),
    ("Free shipping is available for orders over €50", {"category": "shipping", "priority": "medium"}),
    ("International shipping takes 7-14 business days", {"category": "shipping", "priority": "medium"}),

    # Returns category
    ("Return policy allows returns within 30 days of purchase", {"category": "returns", "priority": "high"}),
    ("Product must be in original packaging for returns", {"category": "returns", "priority": "high"}),
    ("Refunds are processed within 5-7 business days", {"category": "returns", "priority": "medium"}),

    # Payment category
    ("We accept Visa, Mastercard, PayPal, and Apple Pay", {"category": "payment", "priority": "high"}),
    ("Bank transfers available for orders over €200", {"category": "payment", "priority": "low"}),
    ("All payments use SSL encryption for security", {"category": "payment", "priority": "medium"}),

    # Support category
    ("Support team available Monday-Friday 9am-6pm CET", {"category": "support", "priority": "high"}),
    ("Email support at support@techcorp.com", {"category": "support", "priority": "high"}),
    ("Phone support at +49-30-123-4567", {"category": "support", "priority": "medium"}),
    ("Average email response time is 24 hours", {"category": "support", "priority": "low"}),
]

# Clear and repopulate
if collection.count() > 0:
    existing = collection.get()
    collection.delete(ids=existing["ids"])

texts = [doc[0] for doc in documents]
metadatas = [doc[1] for doc in documents]
embeddings = embedding_model.encode(texts).tolist()

collection.upsert(
    documents=texts,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=metadatas
)
logger.info(f"Stored {len(documents)} categorized documents")


# ── Feature 1: Basic Search (no filter) ─────────────────────────────────

def basic_search(question: str, top_k: int = 3) -> list:
    """Standard semantic search — no filtering."""
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]


# ── Feature 2: Metadata Filtering ───────────────────────────────────────

def filtered_search(question: str, category: str, top_k: int = 3) -> list:
    """
    Search only within a specific category.
    Useful when you know which domain the question belongs to.
    """
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"category": category}   # ← metadata filter
    )
    return results["documents"][0]


def priority_search(question: str, min_priority: str = "high") -> list:
    """Search only high-priority documents."""
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        where={"priority": min_priority}
    )
    return results["documents"][0]


# ── Feature 3: Auto-detect category ─────────────────────────────────────

def detect_category(question: str) -> str:
    """
    Use LLM to detect which category a question belongs to.
    Then use filtered search for better results.
    """
    response = llm_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{
            "role": "user",
            "content": f"""Classify this question into ONE category.
Categories: shipping, returns, payment, support, other

Question: {question}
Answer with only the category name, nothing else."""
        }],
        temperature=0
    )
    category = response.choices[0].message.content.strip().lower()
    logger.info(f"Detected category: {category}")
    return category if category in ["shipping", "returns", "payment", "support"] else None


def smart_rag(question: str) -> dict:
    """
    Smart RAG pipeline:
    1. Detect category
    2. Use filtered search if category found
    3. Fall back to basic search if not
    4. Answer with LLM
    """
    # Step 1: Detect category
    category = detect_category(question)

    # Step 2: Search (filtered or basic)
    if category:
        docs = filtered_search(question, category)
        search_type = f"filtered ({category})"
    else:
        docs = basic_search(question)
        search_type = "basic"

    logger.info(f"Search type: {search_type}, found {len(docs)} docs")

    # Step 3: Answer with LLM
    context = "\n".join([f"- {doc}" for doc in docs])
    response = llm_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{
            "role": "user",
            "content": f"""Answer using ONLY the context below.
If not in context, say "I don't have that information."

Context:
{context}

Question: {question}"""
        }],
        temperature=0.3
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "category": category or "unknown",
        "search_type": search_type,
        "sources": docs
    }


# ── Tests ─────────────────────────────────────────────────────────────────

test_questions = [
    "How long does shipping take?",
    "Can I return a damaged product?",
    "What payment methods do you accept?",
    "How can I contact support?",
    "Is there free shipping?",
]

print("=" * 60)
print("Smart RAG with Auto-Category Detection")
print("=" * 60)

for question in test_questions:
    result = smart_rag(question)
    print(f"\nQ: {result['question']}")
    print(f"Category: {result['category']} | Search: {result['search_type']}")
    print(f"A: {result['answer']}")

# ── Compare basic vs smart RAG ───────────────────────────────────────────

print("\n" + "=" * 60)
print("Basic vs Smart RAG comparison:")
print("=" * 60)

q = "How do I return something I bought?"
print(f"\nQuestion: {q}")

basic_docs = basic_search(q)
smart_result = smart_rag(q)

print(f"\nBasic search found: {len(basic_docs)} docs")
for doc in basic_docs:
    print(f"  • {doc[:60]}...")

print(f"\nSmart search (category: {smart_result['category']}) found: {len(smart_result['sources'])} docs")
for doc in smart_result['sources']:
    print(f"  • {doc[:60]}...")