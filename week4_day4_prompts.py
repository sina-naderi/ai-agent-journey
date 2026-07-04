import os
import logging
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


def ask(system_prompt: str, user_message: str) -> str:
    """Send a message with a specific system prompt."""
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        
    )
    return response.choices[0].message.content


# ── Test 1: Bad vs Good prompt ───────────────────────────────────────────

BAD_PROMPT = "You are a helpful assistant."

GOOD_PROMPT = """You are a customer support agent for a software company.

Your role:
- Answer questions about Python, APIs, and AI tools only
- Be concise: maximum 2 sentences per answer
- If the question is outside your domain, say: "That's outside my area — please contact our general support"

Never:
- Give medical, legal, or financial advice
- Make up features that don't exist
- Use technical jargon without explanation

Format: plain text, no bullet points unless listing steps."""

question = "How do I connect my Python script to a REST API?"

print("=" * 50)
print("BAD PROMPT response:")
print(ask(BAD_PROMPT, question))

print("\n" + "=" * 50)
print("GOOD PROMPT response:")
print(ask(GOOD_PROMPT, question))


# ── Test 2: Out-of-domain question ───────────────────────────────────────

out_of_domain = "What's the best diet for losing weight?"

print("\n" + "=" * 50)
print("Out-of-domain question with GOOD PROMPT:")
print(ask(GOOD_PROMPT, out_of_domain))


# ── Test 3: Structured output ────────────────────────────────────────────

STRUCTURED_PROMPT = """You are a data extraction assistant.
Extract information from text and return it as JSON only.
No explanation, no markdown, just valid JSON.

Format:
{
  "name": "...",
  "goal": "...",
  "skills": ["...", "..."],
  "location": "..."
}"""

bio_text = """
My name is Sina. I'm a programmer from Iran learning AI agent development.
I know Python, APIs, and n8n. My goal is to migrate to Germany and work
as an AI engineer at a European company.
"""

print("\n" + "=" * 50)
print("Structured output test:")
result = ask(STRUCTURED_PROMPT, bio_text)
print(result)

# Try to parse it as JSON
import json
import re

def extract_json(text: str) -> str:
    """Remove markdown code blocks if present."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

# در Test 3:
result = ask(STRUCTURED_PROMPT, bio_text)
print(result)

try:
    clean = extract_json(result)
    data = json.loads(clean)
    print(f"\nParsed successfully!")
    print(f"Name: {data.get('name')}")
    print(f"Goal: {data.get('goal')}")
except json.JSONDecodeError as e:
    print(f"\nStill couldn't parse: {e}")
    print(f"Raw output: {result}")
    
try:
    data = json.loads(result)
    print(f"\nParsed successfully!")
    print(f"Name: {data.get('name')}")
    print(f"Goal: {data.get('goal')}")
except json.JSONDecodeError:
    print("\nCouldn't parse as JSON — prompt needs adjustment")