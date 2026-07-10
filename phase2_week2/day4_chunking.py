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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)

# ── Chunking Functions ───────────────────────────────────────────────────

def chunk_by_size(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of roughly chunk_size words.
    overlap = how many words to repeat between chunks (keeps context).
    """
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # overlap keeps context between chunks

    return chunks


def chunk_by_paragraph(text: str) -> list[str]:
    """Split text by paragraphs (double newline)."""
    paragraphs = text.split("\n\n")
    # Remove empty paragraphs and strip whitespace
    return [p.strip() for p in paragraphs if p.strip()]


# ── Sample long document ─────────────────────────────────────────────────

COMPANY_MANUAL = """
Welcome to TechCorp Customer Service Manual.

Our return policy allows customers to return any product within 30 days of purchase.
The product must be in its original packaging and include all accessories.
Customers need to provide the original receipt or order confirmation email.
Refunds are processed within 5-7 business days after we receive the returned item.

Shipping Policy:
We offer three shipping options for all orders.
Standard shipping takes 3-5 business days within Europe and costs €4.99.
Express shipping takes 1-2 business days and costs €9.99.
Free shipping is available for orders over €50.
International shipping to non-EU countries takes 7-14 business days.

Payment Methods:
TechCorp accepts all major credit cards including Visa and Mastercard.
We also accept PayPal, Apple Pay, and Google Pay.
Bank transfers are available for orders over €200.
All payments are processed securely using SSL encryption.
We do not store any credit card information on our servers.

Warranty Information:
All TechCorp products come with a standard 1-year warranty.
The warranty covers manufacturing defects and hardware failures.
It does not cover physical damage, water damage, or unauthorized modifications.
Extended warranty plans are available for purchase at checkout.
To claim warranty, contact support with your order number and description of the issue.

Customer Support:
Our support team is available Monday to Friday from 9am to 6pm CET.
You can reach us by email at support@techcorp.com.
Phone support is available at +49-30-123-4567.
Live chat is available on our website during business hours.
Average response time for email support is 24 hours.
"""

# ── Ingest document into Chroma ──────────────────────────────────────────

def ingest_document(text: str, collection_name: str, chunk_method: str = "paragraph"):
    """
    Chunk a document and store it in Chroma with metadata.
    chunk_method: 'paragraph' or 'size'
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Clear existing data
    if collection.count() > 0:
        existing = collection.get()
        collection.delete(ids=existing["ids"])

    # Chunk the document
    if chunk_method == "paragraph":
        chunks = chunk_by_paragraph(text)
    else:
        chunks = chunk_by_size(text, chunk_size=100, overlap=20)

    logger.info(f"Created {len(chunks)} chunks using '{chunk_method}' method")

    # Create embeddings
    embeddings = embedding_model.encode(chunks).tolist()

    # Store in Chroma with metadata
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{
            "chunk_index": i,
            "chunk_method": chunk_method,
            "source": "company_manual"
        } for i in range(len(chunks))]
    )

    logger.info(f"Stored {len(chunks)} chunks in '{collection_name}'")
    return collection


def search_and_answer(question: str, collection) -> str:
    """Search collection and answer with LLM."""
    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    context = "\n".join([f"- {doc}" for doc in results["documents"][0]])

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
    return response.choices[0].message.content


# ── Compare chunking methods ─────────────────────────────────────────────

print("=" * 50)
print("Method 1: Paragraph chunking")
collection_para = ingest_document(COMPANY_MANUAL, "manual_paragraph", "paragraph")
print(f"Chunks created: {collection_para.count()}")

print("\n" + "=" * 50)
print("Method 2: Size-based chunking")
collection_size = ingest_document(COMPANY_MANUAL, "manual_size", "size")
print(f"Chunks created: {collection_size.count()}")

# ── Test both methods ────────────────────────────────────────────────────

questions = [
    "How long does express shipping take?",
    "What is the warranty period?",
    "Can I pay with Apple Pay?",
    "How do I contact customer support?",
]

print("\n" + "=" * 50)
print("Comparing answers from both methods:\n")

for question in questions:
    print(f"Q: {question}")
    answer_para = search_and_answer(question, collection_para)
    answer_size = search_and_answer(question, collection_size)
    print(f"Paragraph: {answer_para}")
    print(f"Size-based: {answer_size}")
    print()