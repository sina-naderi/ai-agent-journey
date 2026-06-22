import asyncio
import aiohttp
import logging
import json
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


async def fetch_post(session: aiohttp.ClientSession, post_id: int) -> dict:
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Post {post_id}: {data['title'][:35]}...")
            return {"type": "post", "id": post_id, "title": data["title"]}
    except Exception as e:
        logger.error(f"Failed post {post_id}: {e}")
        return {"type": "post", "id": post_id, "error": str(e)}


async def fetch_user(session: aiohttp.ClientSession, user_id: int) -> dict:
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            logger.info(f"User {user_id}: {data['name']}")
            return {
                "type": "user",
                "id": user_id,
                "name": data["name"],
                "email": data["email"]
            }
    except Exception as e:
        logger.error(f"Failed user {user_id}: {e}")
        return {"type": "user", "id": user_id, "error": str(e)}


async def main():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            fetch_post(session, 1),
            fetch_post(session, 2),
            fetch_user(session, 1),
            fetch_post(session, 99999),  # Test error — should fail gracefully
        )

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())