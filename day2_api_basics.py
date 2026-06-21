import requests

# یک درخواست ساده — Simple GET request
url = "https://jsonplaceholder.typicode.com/posts/1"
response = requests.get(url)

# بررسی status code
print(response.status_code)   # 200 = موفق

# تبدیل JSON به dictionary پایتون
data = response.json()
print(data["title"])
print(":")
print(data["body"])

# ارسال داده با POST — Send data with POST
new_post = {
    "title": "My First AI Agent Post",
    "body": "This is the content",
    "userId": 1
}
response = requests.post(
    "https://jsonplaceholder.typicode.com/posts",
    json=new_post
)
print(response.status_code)   # 201 = ساخته شد
print(response.json())