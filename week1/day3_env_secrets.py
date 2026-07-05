from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read the values
api_key = os.getenv("ANTHROPIC_API_KEY")
name = os.getenv("MY_NAME", "default_value")  # default value can also be set

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file!")

print(f"Hello {name}, API key loaded successfully.")
print(f"Key starts with: {api_key[:10]}...")  # only first few characters, never print the full key