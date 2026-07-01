# test_week3_project.py
import requests

base = "http://localhost:8000"

# Test 1 — health check
print(requests.get(f"{base}/").json())

# Test 2 — chat
print(requests.post(f"{base}/chat", json={
    "user_id": "sina",
    "message": "What is an AI agent in one sentence?"
}).json())

# Test 3 — summarize
print(requests.post(f"{base}/summarize", json={
    "text": "Python is a high-level programming language. It was created by Guido van Rossum in 1991. Python is known for its simple syntax and readability. It is widely used in web development, data science, AI, and automation."
}).json())
