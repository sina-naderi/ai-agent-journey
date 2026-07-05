import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def call_llm(message: str) -> str:
    """
    Send a message to Gap API and return the response.
    
    Args:
        message (str): The user message to send to the LLM
        
    Returns:
        str: The model's response content
    """
    # Initialize OpenAI client with Gap's base URL
    # Gap API is compatible with OpenAI's client library
    client = OpenAI(
        base_url="https://api.gapgpt.app/v1",
        api_key=os.getenv("GAP_API_KEY")  # API key stored in .env file
    )

    # Send chat completion request to the model
    response = client.chat.completions.create(
        model="gpt-5-nano",  # Google's Gemma model hosted on Gap
        messages=[{"role": "user", "content": message}],
        temperature=0.7          # Balanced creativity vs determinism
    )
    
    # Extract and return the assistant's response
    return response.choices[0].message.content


# Test 1 — simple message
result = call_llm("Say hello in 10 words or less.")
print(f"Test 1: {result}")

# Test 2 — with system prompt style
result = call_llm("What is an AI agent? Answer in one sentence.")
print(f"Test 2: {result}")

# Test 3 — see full response structure (using client instead of requests)
# Create a new client instance for the test
client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)

# Make a simple request to inspect the full response object
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hi!"}]
)

print(f"\nFull response structure:")
# Convert the response object to a dictionary for JSON pretty-printing
print(json.dumps({
    "id": response.id,                      # Unique request identifier
    "model": response.model,                # Model used for completion
    "choices": [
        {
            "index": choice.index,          # Choice index (0 for single response)
            "message": {
                "role": choice.message.role,    # Role: "assistant"
                "content": choice.message.content  # The actual response text
            },
            "finish_reason": choice.finish_reason  # Why generation stopped
        }
        for choice in response.choices
    ],
    "usage": {
        "prompt_tokens": response.usage.prompt_tokens,      # Input token count
        "completion_tokens": response.usage.completion_tokens,  # Output token count
        "total_tokens": response.usage.total_tokens          # Total tokens used
    }
}, indent=2))