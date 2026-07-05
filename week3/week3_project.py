import os
import time
import logging
import requests
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="AI Agent - Week 3 Project")

# ── LLM Client with retry and fallback ──────────────────────────────────

def call_gapgpt(message: str, max_retries: int = 3) -> str | None:
    """Call GapGPT API with automatic retry on failure."""
    client = OpenAI(
        base_url="https://api.gapgpt.app/v1",
        api_key=os.getenv("GAP_API_KEY")
    )

    for attempt in range(max_retries):
        try:
            logger.info(f"GapGPT attempt {attempt + 1}/{max_retries}...")
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": message}],
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)

    return None


def call_llm(message: str) -> str:
    """
    Try multiple providers until one works.
    Currently uses GapGPT — add more providers as fallbacks.
    """
    providers = [call_gapgpt]

    for provider in providers:
        logger.info(f"Trying provider: {provider.__name__}...")
        result = provider(message)
        if result:
            return result
        logger.warning(f"{provider.__name__} failed, trying next...")

    return "All providers failed. Check your API keys."


# ── FastAPI Webhook ──────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "running", "version": "week3"}


@app.post("/chat")
async def chat(request: Request):
    """
    Main chat endpoint.
    Receives a message, calls LLM, returns response.

    Expected body: {"user_id": "123", "message": "Hello"}
    """
    try:
        data = await request.json()
        user_id = data.get("user_id", "anonymous")
        message = data.get("message", "")

        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "message field is required"}
            )

        logger.info(f"User {user_id}: {message[:50]}...")
        response = call_llm(message)

        return {
            "user_id": user_id,
            "message": message,
            "response": response,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/summarize")
async def summarize(request: Request):
    """
    Summarize any text using LLM.
    Expected body: {"text": "Long text here..."}
    """
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "text field is required"}
            )

        prompt = f"Summarize the following text in 2-3 sentences:\n\n{text}"
        summary = call_llm(prompt)

        return {
            "original_length": len(text),
            "summary": summary,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)