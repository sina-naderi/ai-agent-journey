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
            return "Error: invalid characters in expression"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

def broken_tool(query: str) -> str:
    """A tool that always fails — for testing error handling."""
    raise Exception("This tool is intentionally broken!")

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
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "broken_tool",
            "description": "Search for information (this one is broken)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "broken_tool": broken_tool,
}

SYSTEM_PROMPT = "You are a helpful assistant. Use tools when needed."

# ── Safe Agent with full error handling ──────────────────────────────────

def run_safe_agent(user_message: str, max_iterations: int = 10) -> str:
    """
    Agent with comprehensive error handling:
    - Tool failures don't crash the agent
    - Unknown tools are handled gracefully
    - Max iterations prevent infinite loops
    - API errors are caught and reported
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.info(f"Iteration {iterations}/{max_iterations}")

        try:
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=30
            )
            choice = response.choices[0]

            # Agent is done
            if choice.finish_reason == "stop":
                logger.info("Agent completed successfully")
                return choice.message.content

            # Agent wants to use tools
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    func_name = tc.function.name
                    logger.info(f"Tool requested: {func_name}")

                    # Safety check — unknown tool
                    if func_name not in available_functions:
                        result = f"Error: unknown tool '{func_name}'"
                        logger.warning(result)
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = available_functions[func_name](**args)
                            logger.info(f"Tool result: {result}")
                        except json.JSONDecodeError:
                            result = "Error: invalid tool arguments"
                            logger.error(result)
                        except Exception as e:
                            # Tool failed — don't crash, report error to LLM
                            result = f"Tool error: {str(e)}"
                            logger.error(f"Tool {func_name} failed: {e}")

                    # Always add tool result (even if error)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

        except Exception as e:
            logger.error(f"API error on iteration {iterations}: {e}")
            return f"Agent stopped due to API error: {e}"

    # Max iterations reached
    logger.warning(f"Max iterations ({max_iterations}) reached")
    return "Agent stopped: max iterations reached to prevent infinite loop."


# ── Tests ─────────────────────────────────────────────────────────────────

print("=" * 50)
print("Test 1: Normal operation")
print(run_safe_agent("What time is it and what is 42 * 7?"))

print("\n" + "=" * 50)
print("Test 2: Broken tool — agent should handle gracefully")
print(run_safe_agent("Search for information about Python using the search tool"))

print("\n" + "=" * 50)
print("Test 3: Max iterations (set very low to test)")
print(run_safe_agent("Keep calculating: what is 1+1, then 2+2, then 3+3, then 4+4, then 5+5, then 6+6, then 7+7, then 8+8, then 9+9, then 10+10", max_iterations=2))