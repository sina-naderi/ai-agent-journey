import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Load embedding model ─────────────────────────────────────────────────
logger.info("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Model loaded!")

# ── Setup Chroma ─────────────────────────────────────────────────────────
# PersistentClient = داده‌ها روی دیسک ذخیره می‌شن (بعد از بستن برنامه هم هستن)
client = chromadb.PersistentClient(path="./chroma_db")

# ساخت یه collection (مثل یه جدول در دیتابیس)
collection = client.get_or_create_collection(
    name="company_docs",
    metadata={"description": "Company FAQ and policies"}
)
logger.info("Chroma collection ready!")

# ── Add documents ────────────────────────────────────────────────────────
documents = [
    "The return policy is 30 days for all products with original receipt",
    "Shipping takes 3-5 business days to Europe and 7-10 days worldwide",
    "Contact support at support@company.com or call +1-800-123-4567",
    "Our office is open Monday to Friday, 9am to 5pm CET",
    "We accept Visa, Mastercard, PayPal, and bank transfers",
    "All products come with a 1-year warranty against manufacturing defects",
    "You can track your order using the tracking number sent to your email",
    "Bulk orders of 10+ items get a 15% discount automatically",
]

# Embedding ها رو بساز
logger.info("Creating embeddings and storing in Chroma...")
embeddings = model.encode(documents).tolist()

# توی Chroma ذخیره کن
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": "company_faq", "index": i} for i in range(len(documents))]
)
logger.info(f"Stored {len(documents)} documents in Chroma!")

# ── Semantic Search ──────────────────────────────────────────────────────
def search(query: str, top_k: int = 2) -> list:
    """Search for most relevant documents."""
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return list(zip(
        results["documents"][0],
        results["distances"][0]
    ))

# ── Tests ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Test 1: 'Can I return a product after 2 weeks?'")
for doc, distance in search("Can I return a product after 2 weeks?"):
    print(f"  [score: {1 - distance:.3f}] {doc}")

print("\n" + "=" * 50)
print("Test 2: 'How long does delivery take?'")
for doc, distance in search("How long does delivery take?"):
    print(f"  [score: {1 - distance:.3f}] {doc}")

print("\n" + "=" * 50)
print("Test 3: 'Is there any discount for large orders?'")
for doc, distance in search("Is there any discount for large orders?"):
    print(f"  [score: {1 - distance:.3f}] {doc}")

print("\n" + "=" * 50)
print("Test 4: 'What warranty do products have?'")
for doc, distance in search("What warranty do products have?"):
    print(f"  [score: {1 - distance:.3f}] {doc}")

# ── Check persistence ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"Total documents in collection: {collection.count()}")
print("Data saved to ./chroma_db/ folder — persists after restart!")