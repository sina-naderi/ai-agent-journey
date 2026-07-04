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

# ── Tools (same as before) ───────────────────────────────────────────────

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
    }
]

available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

SYSTEM_PROMPT = """You are a helpful AI assistant.
Remember everything the user tells you during this conversation.
Be concise. Answer in the same language as the user."""

# ── Agent with Memory ────────────────────────────────────────────────────

class AgentWithMemory:
    """An agent that remembers the full conversation history."""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.gapgpt.app/v1",
            api_key=os.getenv("GAP_API_KEY")
        )
        # Memory = message history
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def chat(self, user_input: str, max_iter: int = 10) -> str:
        """Send a message and maintain conversation history."""
        self.messages.append({"role": "user", "content": user_input})
        iterations = 0

        while iterations < max_iter:
            iterations += 1
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=self.messages,
                tools=tools,
                tool_choice="auto"
            )
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                reply = choice.message.content
                # Save assistant reply to memory
                self.messages.append({
                    "role": "assistant",
                    "content": reply
                })
                return reply

            if choice.finish_reason == "tool_calls":
                self.messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = available_functions[tc.function.name](**args)
                    except Exception as e:
                        result = f"Tool error: {e}"

                    logger.info(f"Tool: {tc.function.name} → {result}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

        return "Max iterations reached."

    def clear_memory(self):
        """Reset conversation — forget everything."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("Memory cleared.")

    def show_history(self):
        """Print conversation history (useful for debugging)."""
        print("\n── Conversation History ──")
        for msg in self.messages:
            if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
                print(f"[{msg['role'].upper()}] {msg['content'][:80]}...")
        print("─────────────────────────\n")


# ── Tests ─────────────────────────────────────────────────────────────────

agent = AgentWithMemory()

print("=== Test: Memory across messages ===\n")

# Message 1 — introduce yourself
response = agent.chat("Hi! My name is Sina and I'm learning to build AI agents.")
print(f"Agent: {response}\n")

# Message 2 — does it remember?
response = agent.chat("What is my name?")
print(f"Agent: {response}\n")

# Message 3 — add more context
response = agent.chat("I want to migrate to Germany and work as an AI engineer.")
print(f"Agent: {response}\n")

# Message 4 — does it remember everything?
response = agent.chat("Summarize what you know about me in one sentence.")
print(f"Agent: {response}\n")

# Show history
agent.show_history()

# Test clear memory
print("=== Test: After clearing memory ===\n")
agent.clear_memory()
response = agent.chat("What is my name?")
print(f"Agent: {response}")