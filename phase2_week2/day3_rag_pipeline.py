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
logger.info("Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

llm_client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)

# ── Setup Chroma ─────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

# Add documents if collection is empty
if collection.count() == 0:
    logger.info("Collection empty — adding documents...")
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
    embeddings = embedding_model.encode(documents).tolist()
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )
    logger.info(f"Added {len(documents)} documents!")
else:
    logger.info(f"Collection has {collection.count()} documents — skipping insert")

# ── RAG Function ─────────────────────────────────────────────────────────

def rag_answer(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Search Chroma for relevant documents
    3. Send question + context to LLM
    4. Return answer
    """
    logger.info(f"Question: {question}")

    # Step 1: Embed the question
    question_embedding = embedding_model.encode([question]).tolist()

    # Step 2: Search for relevant documents
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    relevant_docs = results["documents"][0]
    logger.info(f"Found {len(relevant_docs)} relevant documents")

    # Step 3: Build context from retrieved documents
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    # Step 4: Send to LLM with context
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    response = llm_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3   # low temperature = more factual answers
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": relevant_docs
    }

# ── Tests ─────────────────────────────────────────────────────────────────

questions = [
    "Can I return a product after 3 weeks?",
    "How long does shipping take to Germany?",
    "Do you offer any discounts?",
    "What payment methods are accepted?",
    "What is the weather in Tehran?",   # out of context — should say "I don't have that info"
]

for question in questions:
    print("\n" + "=" * 50)
    result = rag_answer(question)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources used:")
    for src in result['sources']:
        print(f"  • {src[:60]}...")