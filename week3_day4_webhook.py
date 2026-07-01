import os
import json
import logging
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

app = FastAPI(title="AI Agent Webhook")

client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=os.getenv("GAP_API_KEY")
)


def get_ai_response(message: str) -> str:
    """Get response from LLM for a given message."""
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": message}],
        temperature=0.7
    )
    return response.choices[0].message.content


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "service": "AI Agent Webhook"}


@app.post("/webhook/message")
async def handle_message(request: Request):
    """
    Receive a message via webhook and respond with AI.

    Expected body: {"user_id": "123", "text": "Hello"}
    """
    try:
        data = await request.json()
        logger.info(f"Received webhook: {json.dumps(data)}")

        user_id = data.get("user_id", "unknown")
        text = data.get("text", "")

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "text field is required"}
            )

        # Get AI response
        ai_response = get_ai_response(text)
        logger.info(f"AI response for user {user_id}: {ai_response[:50]}...")

        return {
            "user_id": user_id,
            "response": ai_response,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)