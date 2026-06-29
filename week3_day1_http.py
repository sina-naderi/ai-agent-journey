import requests
from dotenv import load_dotenv

load_dotenv()

# ── Section 1: Headers ──────────────────────────────────────────────────────

# Common headers used in most APIs
headers = {
    "Content-Type": "application/json",   # type of data we're sending
    "Authorization": "Bearer YOUR_TOKEN", # authentication
    "User-Agent": "MyAIAgent/1.0",        # identify our app
    "Accept": "application/json"          # type of data we want back
}

response = requests.get(
    "https://jsonplaceholder.typicode.com/posts",
    headers=headers
)

# Print response headers from server
print(response.headers["Content-Type"])


# ── Section 2: Query Parameters ─────────────────────────────────────────────

# Without params — returns all posts (100 items)
response = requests.get("https://jsonplaceholder.typicode.com/posts")
print(len(response.json()))  # 100

# With params — filtered results
params = {
    "userId": 1,   # only posts from user #1
    "_limit": 3    # only 3 results
}
response = requests.get(
    "https://jsonplaceholder.typicode.com/posts",
    params=params
)
print(len(response.json()))  # 3
print(response.url)          # see how params are added to the URL


# ── Section 3: Pagination ────────────────────────────────────────────────────

def fetch_all_posts() -> list:
    """Fetch all posts using pagination."""
    all_posts = []
    page = 1

    while True:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/posts",
            params={"_page": page, "_limit": 10}
        )
        data = response.json()

        if not data:  # if page is empty, we're done
            break

        all_posts.extend(data)
        print(f"Page {page}: got {len(data)} posts")
        page += 1

    return all_posts


posts = fetch_all_posts()
print(f"Total posts: {len(posts)}")