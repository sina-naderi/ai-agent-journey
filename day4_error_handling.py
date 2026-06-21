import requests
import logging
from dotenv import load_dotenv

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def fetch_post(post_id: int) -> dict | None:
    """
    یک post را از API می‌گیرد.
    Returns None on failure.
    """
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"

    try:
        logger.info(f"Fetching post {post_id}...")
        response = requests.get(url, timeout=10)   # timeout مهم است!
        response.raise_for_status()                # اگر 4xx/5xx بود، exception بیندازد

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


# تست با ID درست
post = fetch_post(1)
if post:
    print(f"Title: {post['title']}")

# تست با ID اشتباه — باید لاگ خطا ببینی، نه crash
fetch_post(99999)