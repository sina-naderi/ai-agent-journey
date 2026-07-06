import os
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Load embedding model ─────────────────────────────────────────────────
# This runs locally — no API key needed
logger.info("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Model loaded!")

# ── Create embeddings ────────────────────────────────────────────────────
sentences = [
    "The return policy is 30 days for all products",
    "Shipping takes 3-5 business days to Europe",
    "Contact support at support@company.com",
    "Our office is open Monday to Friday, 9am to 5pm",
    "We accept Visa, Mastercard, and PayPal",
]

logger.info("Creating embeddings...")
embeddings = model.encode(sentences)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]} numbers (dimensions)")
print(f"\nFirst embedding (first 5 numbers): {embeddings[0][:5]}")

# ── Semantic search ──────────────────────────────────────────────────────
import numpy as np

def cosine_similarity(a, b):
    """Calculate similarity between two vectors (1 = identical, 0 = unrelated)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query: str, top_k: int = 2) -> list:
    """Find most similar sentences to the query."""
    query_embedding = model.encode([query])[0]
    
    similarities = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarities.append((score, sentences[i]))
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    return similarities[:top_k]

# ── Tests ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Test 1: 'How can I return a product?'")
results = semantic_search("How can I return a product?")
for score, text in results:
    print(f"  [{score:.3f}] {text}")

print("\n" + "=" * 50)
print("Test 2: 'What payment methods do you accept?'")
results = semantic_search("What payment methods do you accept?")
for score, text in results:
    print(f"  [{score:.3f}] {text}")

print("\n" + "=" * 50)
print("Test 3: 'When does the office close?'")
results = semantic_search("When does the office close?")
for score, text in results:
    print(f"  [{score:.3f}] {text}")