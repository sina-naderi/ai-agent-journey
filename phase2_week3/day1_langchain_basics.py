# phase2_week3/day1_langchain_basics.py
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Setup LLM ────────────────────────────────────────────────────────────
# Initialize LLM using OpenAI-compatible interface (works with GapGPT)
llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

# ── Test 1: Simple invocation ────────────────────────────────────────────
# Our old method:
# response = client.chat.completions.create(...)
# reply = response.choices[0].message.content

# The LangChain Approach — Much Simpler:
logger.info("Test 1: Simple message...")
response = llm.invoke("Say hello in 10 words or less.")
print(f"Test 1: {response.content}\n")

# ── Test 2: With system and human messages ───────────────────────────────
logger.info("Test 2: With system message...")
messages = [
    SystemMessage(content="You are a helpful assistant. Be very concise."),
    HumanMessage(content="What is RAG in AI?")
]
response = llm.invoke(messages)
print(f"Test 2: {response.content}\n")

# ── Test 3: Prompt Template ──────────────────────────────────────────────
# Prompt templates allow reusable prompts with variable substitution
# Similar to f-strings but more powerful and composable
logger.info("Test 3: Prompt template...")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Be concise."),
    ("human", "{question}")
])

# Chain operator (|) pipes output from prompt into llm
chain = prompt | llm

result = chain.invoke({
    "domain": "Python programming",
    "question": "What is asyncio?"
})
print(f"Test 3: {result.content}\n")

# ── Test 4: Reuse chain with multiple inputs ──────────────────────────────
logger.info("Test 4: Multiple invocations with same chain...")

topics = ["RAG", "Vector Database", "LangChain"]
for topic in topics:
    result = chain.invoke({
        "domain": "AI engineering",
        "question": f"What is {topic}? One sentence only."
    })
    print(f"{topic}: {result.content}")