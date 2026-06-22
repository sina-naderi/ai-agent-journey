import requests

# Simple GET request
url = "https://jsonplaceholder.typicode.com/posts/1"
response = requests.get(url)

# Check status code
print(response.status_code)   # 200 = success

# Convert JSON to Python dictionary
data = response.json()
print(data["title"])
print(":")
print(data["body"])

# Send data with POST
new_post = {
    "title": "My First AI Agent Post",
    "body": "This is the content",
    "userId": 1
}
response = requests.post(
    "https://jsonplaceholder.typicode.com/posts",
    json=new_post
)
print(response.status_code)   # 201 = created
print(response.json())