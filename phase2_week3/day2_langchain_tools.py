# phase2_week3/day2_langchain_tools.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY"),
    model="gpt-5-nano",
    temperature=0.7
)

# ── Define tools using @tool decorator ───────────────────────────────────
# The docstring becomes the tool description — the LLM uses it to decide
# when and why to call each tool. Write clear, specific descriptions!

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression to evaluate, e.g. '2 + 2' or '15 * 4'
    """
    try:
        # Only allow safe mathematical characters
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name, e.g. 'Amsterdam' or 'Berlin'
    """
    # Mock weather data — replace with real API in production
    mock_weather = {
        "Amsterdam": "18°C, Cloudy",
        "Berlin": "22°C, Sunny",
        "London": "15°C, Rainy",
        "Vienna": "24°C, Clear",
        "Tehran": "32°C, Sunny",
    }
    return mock_weather.get(city, f"No weather data available for {city}")

# ── Create agent — replaces our entire manual ReAct loop from Phase 1 ────
tools = [get_current_time, calculate, get_weather]
agent = create_agent(llm, tools)

def run(question: str) -> str:
    """Run the agent and return the final answer."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    # Last message in the list is always the final answer
    return result["messages"][-1].content

# ── Tests ─────────────────────────────────────────────────────────────────
print("=" * 50)
print("Test 1: Time tool")
print(run("What time is it right now?"))

print("\n" + "=" * 50)
print("Test 2: Calculator tool")
print(run("What is 1234 * 5678?"))

print("\n" + "=" * 50)
print("Test 3: Weather tool — two cities")
print(run("What's the weather in Amsterdam and Berlin?"))

print("\n" + "=" * 50)
print("Test 4: Multiple tools in one question")
print(run("What time is it and what is 42 * 7?"))