import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Setup ────────────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)
chroma_client = chromadb.PersistentClient(path="./chroma_db_qa_bot")

# ── Knowledge Base ───────────────────────────────────────────────────────
DOCUMENTS = [
    # Shipping
    ("Standard shipping takes 3-5 business days within Europe and costs €4.99",
     {"category": "shipping", "priority": "high"}),
    ("Express shipping takes 1-2 business days and costs €9.99",
     {"category": "shipping", "priority": "high"}),
    ("Free shipping is available for orders over €50",
     {"category": "shipping", "priority": "medium"}),
    ("International shipping to non-EU countries takes 7-14 business days",
     {"category": "shipping", "priority": "medium"}),
    ("You can track your order using the tracking number sent to your email",
     {"category": "shipping", "priority": "low"}),
    # Returns
    ("Return policy allows returns within 30 days of purchase with original receipt",
     {"category": "returns", "priority": "high"}),
    ("Product must be in original packaging with all accessories for returns",
     {"category": "returns", "priority": "high"}),
    ("Refunds are processed within 5-7 business days after receiving the item",
     {"category": "returns", "priority": "medium"}),
    # Payment
    ("We accept Visa, Mastercard, PayPal, Apple Pay, and Google Pay",
     {"category": "payment", "priority": "high"}),
    ("Bank transfers are available for orders over €200",
     {"category": "payment", "priority": "low"}),
    ("All payments are processed securely using SSL encryption",
     {"category": "payment", "priority": "medium"}),
    ("We do not store any credit card information on our servers",
     {"category": "payment", "priority": "medium"}),
    # Warranty
    ("All products come with a standard 1-year warranty",
     {"category": "warranty", "priority": "high"}),
    ("Warranty covers manufacturing defects and hardware failures",
     {"category": "warranty", "priority": "high"}),
    ("Warranty does not cover physical damage, water damage, or unauthorized modifications",
     {"category": "warranty", "priority": "high"}),
    ("Extended warranty plans are available for purchase at checkout",
     {"category": "warranty", "priority": "medium"}),
    # Support
    ("Support team is available Monday to Friday from 9am to 6pm CET",
     {"category": "support", "priority": "high"}),
    ("Email support at support@techcorp.com — average response time 24 hours",
     {"category": "support", "priority": "high"}),
    ("Phone support available at +49-30-123-4567 during business hours",
     {"category": "support", "priority": "medium"}),
    ("Live chat is available on our website during business hours",
     {"category": "support", "priority": "low"}),
    # Discounts
    ("Bulk orders of 10+ items get a 15% discount automatically",
     {"category": "discounts", "priority": "high"}),
    ("Newsletter subscribers get 10% off their first order",
     {"category": "discounts", "priority": "medium"}),
    ("Student discount of 20% available with valid student ID",
     {"category": "discounts", "priority": "medium"}),
]

# ── Initialize collection ────────────────────────────────────────────────
collection = chroma_client.get_or_create_collection(name="qa_bot_docs")

if collection.count() == 0:
    logger.info("Initializing knowledge base...")
    texts = [doc[0] for doc in DOCUMENTS]
    metadatas = [doc[1] for doc in DOCUMENTS]
    embeddings = embedding_model.encode(texts).tolist()
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
        metadatas=metadatas
    )
    logger.info(f"Loaded {len(DOCUMENTS)} documents into knowledge base")
else:
    logger.info(f"Knowledge base ready with {collection.count()} documents")

# ── Q&A Bot Class ────────────────────────────────────────────────────────

class QABot:
    """
    A customer support Q&A bot using RAG:
    - Detects question category
    - Searches relevant documents
    - Answers with LLM grounded in context
    - Maintains conversation history
    """

    SYSTEM_PROMPT = """You are a helpful customer support agent for TechCorp.
Answer questions using ONLY the provided context.
If the answer is not in the context, say: "I don't have that information. Please contact support@techcorp.com"
Be concise and friendly."""

    CATEGORIES = ["shipping", "returns", "payment", "warranty", "support", "discounts"]

    def __init__(self):
        self.history = []

    def detect_category(self, question: str) -> str | None:
        """Detect question category using LLM."""
        response = llm_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{
                "role": "user",
                "content": f"""Classify into ONE category: {', '.join(self.CATEGORIES)}, other
Question: {question}
Reply with only the category name."""
            }],
            temperature=0
        )
        cat = response.choices[0].message.content.strip().lower()
        return cat if cat in self.CATEGORIES else None

    def search(self, question: str, category: str | None, top_k: int = 3) -> list:
        """Search with optional category filter."""
        query_embedding = embedding_model.encode([question]).tolist()
        kwargs = {"query_embeddings": query_embedding, "n_results": top_k}
        if category:
            kwargs["where"] = {"category": category}
        results = collection.query(**kwargs)
        return results["documents"][0]

    def answer(self, question: str) -> str:
        """Full RAG pipeline: detect → search → answer."""
        # Detect category
        category = self.detect_category(question)
        logger.info(f"Category: {category or 'unknown'}")

        # Search
        docs = self.search(question, category)
        context = "\n".join([f"- {doc}" for doc in docs])

        # Build messages with history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(self.history[-4:])  # last 2 turns
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })

        # Get answer
        response = llm_client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            temperature=0.3
        )
        reply = response.choices[0].message.content

        # Save to history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": reply})

        return reply

# ── CLI Interface ─────────────────────────────────────────────────────────

def main():
    bot = QABot()
    print("╔══════════════════════════════════════════╗")
    print("║   TechCorp Customer Support Bot 🤖       ║")
    print("║   Powered by RAG + GapGPT                ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Ask about: shipping, returns, payment,  ║")
    print("║  warranty, support, discounts            ║")
    print("║  Type 'quit' to exit                     ║")
    print("╚══════════════════════════════════════════╝\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            answer = bot.answer(question)
            print(f"\nBot: {answer}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()