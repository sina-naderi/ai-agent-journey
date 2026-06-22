import asyncio
import aiohttp
import time

# ─── روش sync (کند) ───────────────────────
import requests

def fetch_sync(post_id):
    response = requests.get(f"https://jsonplaceholder.typicode.com/posts/{post_id}")
    return response.json()["title"]

def run_sync():
    start = time.time()
    results = [fetch_sync(i) for i in range(1, 4)]   # یکی‌یکی صبر می‌کنه
    print(f"Sync: {time.time() - start:.2f}s")
    return results

# ─── روش async (سریع) ───────────────────────
async def fetch_async(session, post_id):
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    async with session.get(url) as response:
        data = await response.json()
        return data["title"]

async def run_async():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_async(session, i) for i in range(1, 4)]
        results = await asyncio.gather(*tasks)    # همه با هم اجرا می‌شن
    print(f"Async: {time.time() - start:.2f}s")
    return results

# ─── اجرا ───────────────────────
run_sync()
asyncio.run(run_async())