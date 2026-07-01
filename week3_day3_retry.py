import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


def call_llm_with_retry(message: str, max_retries: int = 3) -> str | None:
    """
    Call GapGPT API with automatic retry on failure.

    Args:
        message: The user message to send
        max_retries: Maximum number of retry attempts

    Returns:
        The model response, or None if all retries failed
    """
    client = OpenAI(
        base_url="https://api.gapgpt.app/v1",
        api_key=os.getenv("GAP_API_KEY")
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}...")

            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=500
            )

            
            # Check what's actually coming back
            print(f"Content: '{response.choices[0].message.content}'")
            print(f"Finish reason: {response.choices[0].finish_reason}")
            print(f"Tokens: {response.usage}")

            logger.info("Success!")
            return response.choices[0].message.content

        except Exception as e:
            # Exponential backoff: wait 1s, 2s, 4s between retries
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)

    logger.error("All retries exhausted")
    return None


def call_llm_with_fallback(message: str) -> str:
    """
    Try multiple providers until one works.

    This is the pattern you actually used this week —
    when one service failed, you tried another.
    """
    providers = [
        {"name": "GapGPT-gpt-5-nano", "func": call_llm_with_retry},
        # Add more providers here as fallbacks
    ]

    for provider in providers:
        logger.info(f"Trying provider: {provider['name']}...")
        result = provider["func"](message)

        if result:
            logger.info(f"Success with {provider['name']}")
            return result

        logger.warning(f"{provider['name']} failed, trying next...")

    return "All providers failed. Check your API keys and quotas."


# Test retry logic
print("=== Test 1: Normal call ===")
result = call_llm_with_retry("Say hello in 10 words or less.")
print(f"Result: {result}")

print("\n=== Test 2: Fallback pattern ===")
result = call_llm_with_fallback("What is an AI agent? Answer in one sentence.")
print(f"Result: {result}")