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

# ── Step 1: Define real functions ────────────────────────────────────────

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe mathematical operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Get mock weather for a city (replace with real API later)."""
    mock_weather = {
        "Tehran": "28°C, Sunny",
        "Amsterdam": "18°C, Cloudy",
        "Berlin": "22°C, Partly cloudy",
        "London": "15°C, Rainy",
    }
    return mock_weather.get(city, f"Weather data not available for {city}")

# ── Step 2: Define tools for the LLM ────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
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
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or '15 * 4'"
                    }
                },
                "required": ["expression"]
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
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tehran' or 'Amsterdam'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ── Step 3: Map function names to actual functions ───────────────────────

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}

# ── Step 4: Run the agent loop ───────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Run the agent loop:
    1. Send message + tools to LLM
    2. If LLM wants to use a tool, call it
    3. Send tool result back to LLM
    4. Repeat until LLM gives final answer
    """
    logger.info(f"User: {user_message}")
    messages = [{"role": "user", "content": user_message}]

    while True:
        # Send to LLM with available tools
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # LLM decides when to use tools
        )

        choice = response.choices[0]

        # If LLM is done, return final answer
        if choice.finish_reason == "stop":
            logger.info("Agent finished.")
            return choice.message.content

        # If LLM wants to use tools
        if choice.finish_reason == "tool_calls":
            # Add LLM's response to message history
            messages.append(choice.message)

            # Execute each tool the LLM requested
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                logger.info(f"Tool called: {func_name}({func_args})")

                # Call the real function
                func = available_functions[func_name]
                result = func(**func_args)

                logger.info(f"Tool result: {result}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })


# ── Test the agent ───────────────────────────────────────────────────────

print("=" * 50)
result = run_agent("What time is it right now?")
print(f"Answer: {result}\n")

print("=" * 50)
result = run_agent("What is 1234 * 5678?")
print(f"Answer: {result}\n")

print("=" * 50)
result = run_agent("What's the weather in Amsterdam and Tehran?")
print(f"Answer: {result}\n")