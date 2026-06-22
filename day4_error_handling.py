import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def fetch_post(post_id: int) -> dict | None:
    """
    Fetch a post from the API.
    Returns None on failure.
    """
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"

    try:
        logger.info(f"Fetching post {post_id}...")
        response = requests.get(url, timeout=10)   # timeout is important!
        response.raise_for_status()                # raise exception if 4xx/5xx

        data = response.json()
        logger.info(f"Successfully fetched: {data['title'][:40]}...")
        return data

    except requests.exceptions.Timeout:
        logger.error("Request timed out after 10 seconds")
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} — {e}")
        return None

    except requests.exceptions.ConnectionError:
        logger.error("Network connection failed — check your internet")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


# Test with a valid ID
post = fetch_post(1)
if post:
    print(f"Title: {post['title']}")

# Test with an invalid ID — should see error log, not a crash
fetch_post(99999)