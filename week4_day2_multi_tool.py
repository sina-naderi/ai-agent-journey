import os
import json
import logging
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)

# ── Tools ────────────────────────────────────────────────────────────────

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

def search_knowledge(query: str) -> str:
    """Search a simple knowledge base for technical info."""
    kb = {
        "python": "Python is a high-level programming language known for simplicity.",
        "ai agent": "An AI agent perceives its environment and takes actions to achieve goals.",
        "langchain": "LangChain is a framework for building LLM-powered applications.",
        "rag": "RAG combines search with LLM generation for grounded responses.",
        "fastapi": "FastAPI is a modern Python web framework for building APIs.",
    }
    for key, val in kb.items():
        if key in query.lower():
            return val
    return f"No info found for: {query}"

def get_weather(city: str) -> str:
    mock_weather = {
        "Tehran": "28°C, Sunny",
        "Amsterdam": "18°C, Cloudy",
        "Berlin": "22°C, Partly cloudy",
        "London": "15°C, Rainy",
    }
    return mock_weather.get(city, f"No data for {city}")

# ── Tool definitions ──────────────────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '2 + 2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search knowledge base for technical questions about Python, AI, or frameworks",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
    "get_weather": get_weather,
}

# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant for a developer learning AI agents.

You have access to tools — use them when needed:
- get_current_time: when asked about time or date
- calculate: for any math calculation
- search_knowledge: for technical questions about Python, AI, or frameworks
- get_weather: for weather questions

Rules:
- Be concise: keep answers under 3 sentences unless explaining code
- If unsure, say so — never make up technical details
- Always respond in the same language the user writes in
- Use tools proactively when they would help
"""

# ── Agent Loop ────────────────────────────────────────────────────────────

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """Run the agent with system prompt and multiple tools."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.info(f"Iteration {iterations}...")

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = available_functions[tc.function.name](**args)
                except Exception as e:
                    result = f"Tool error: {e}"

                logger.info(f"Tool: {tc.function.name} → {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

    return "Max iterations reached."


# ── Tests ─────────────────────────────────────────────────────────────────

print("=" * 50)
print("Test 1: Time + Calculation")
print(run_agent("What time is it? Also, what is 999 * 123?"))

print("\n" + "=" * 50)
print("Test 2: Knowledge Base")
print(run_agent("What is RAG and how does it relate to AI agents?"))

print("\n" + "=" * 50)
print("Test 3: Multi-tool in one question")
print(run_agent("What's the weather in Berlin and Amsterdam, and what time is it there?"))