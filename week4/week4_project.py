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
    kb = {
        "python": "Python is a high-level programming language known for simplicity and readability.",
        "ai agent": "An AI agent perceives its environment and takes actions to achieve defined goals.",
        "langchain": "LangChain is a framework for building applications powered by language models.",
        "rag": "RAG (Retrieval Augmented Generation) combines search with LLM generation.",
        "fastapi": "FastAPI is a modern, fast Python web framework for building APIs.",
        "docker": "Docker is a platform for developing and running applications in containers.",
        "postgresql": "PostgreSQL is a powerful open-source relational database system.",
        "n8n": "n8n is a workflow automation tool that connects different services and APIs.",
        "vector database": "A vector database stores embeddings for semantic search — used in RAG systems.",
    }
    query_lower = query.lower()
    for key, val in kb.items():
        if key in query_lower:
            return val
    return f"No information found for: {query}. Try: python, ai agent, langchain, rag, fastapi, docker, postgresql, n8n"

def get_weather(city: str) -> str:
    mock_weather = {
        "Tehran": "32°C, Sunny",
        "Amsterdam": "18°C, Cloudy",
        "Berlin": "22°C, Partly cloudy",
        "London": "15°C, Rainy",
        "Vienna": "24°C, Sunny",
        "Zurich": "20°C, Clear",
        "Brussels": "17°C, Overcast",
    }
    return mock_weather.get(city, f"Weather data not available for {city}")

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
            "description": "Evaluate a mathematical expression. Use for any math calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression e.g. '2 + 2' or '15 * 4 / 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search knowledge base for technical topics: python, ai agent, langchain, rag, fastapi, docker, postgresql, n8n, vector database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Technical topic to search for"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Supports: Tehran, Amsterdam, Berlin, London, Vienna, Zurich, Brussels",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
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

# ── AI Assistant Class ────────────────────────────────────────────────────

class AIAssistant:
    """
    A complete AI assistant with:
    - Multiple tools (time, calculator, knowledge base, weather)
    - Conversation memory
    - Error handling and max iteration guard
    - Proper system prompt
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant for a developer learning AI agent development.

You have access to these tools — use them when needed:
- get_current_time: when asked about time or date
- calculate: for any math calculation  
- search_knowledge: for technical questions about Python, AI, frameworks
- get_weather: for weather in specific cities

Rules:
- Be concise: 1-3 sentences for simple questions, more for complex ones
- Use tools proactively when they would help give accurate answers
- If you don't know something, say so honestly
- Answer in the same language the user writes in
- Remember everything from this conversation"""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.gapgpt.app/v1",
            api_key=os.getenv("GAP_API_KEY")
        )
        self.messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self.turn_count = 0

    def chat(self, user_input: str, max_iter: int = 10) -> str:
        """Process a user message and return the assistant's response."""
        self.turn_count += 1
        self.messages.append({"role": "user", "content": user_input})
        iterations = 0

        while iterations < max_iter:
            iterations += 1
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto"
                )
                choice = response.choices[0]

                if choice.finish_reason == "stop":
                    reply = choice.message.content
                    self.messages.append({
                        "role": "assistant",
                        "content": reply
                    })
                    return reply

                if choice.finish_reason == "tool_calls":
                    self.messages.append(choice.message)
                    for tc in choice.message.tool_calls:
                        try:
                            if tc.function.name not in available_functions:
                                result = f"Error: unknown tool '{tc.function.name}'"
                            else:
                                args = json.loads(tc.function.arguments)
                                result = available_functions[tc.function.name](**args)
                                logger.info(f"Tool {tc.function.name} → {result}")
                        except Exception as e:
                            result = f"Tool error: {e}"
                            logger.error(f"Tool {tc.function.name} failed: {e}")

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })

            except Exception as e:
                logger.error(f"API error: {e}")
                return f"Sorry, I encountered an error: {e}"

        return "I reached my processing limit for this request."

    def clear(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.turn_count = 0
        print("Memory cleared.\n")


# ── CLI Interface ─────────────────────────────────────────────────────────

def main():
    assistant = AIAssistant()
    print("╔══════════════════════════════════════╗")
    print("║     AI Assistant — Week 4 Project    ║")
    print("╠══════════════════════════════════════╣")
    print("║  Tools: time, calculator, knowledge  ║")
    print("║         base, weather                ║")
    print("║  Commands: 'clear', 'quit'           ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                assistant.clear()
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()