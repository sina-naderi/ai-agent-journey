import requests

response = requests.post(
    "http://localhost:8000/webhook/message",
    json={"user_id": "123", "text": "What is Python?"}
)
print(response.json())