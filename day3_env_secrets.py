from dotenv import load_dotenv
import os

# بارگذاری متغیرها از فایل .env
load_dotenv()

# خواندن مقادیر
api_key = os.getenv("ANTHROPIC_API_KEY")
name = os.getenv("MY_NAME", "default_value")  # مقدار پیش‌فرض هم می‌شه داد

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file!")

print(f"Hello {name}, API key loaded successfully.")
print(f"Key starts with: {api_key[:10]}...")  # فقط چند کاراکتر اول، هرگز کل کلید رو print نکن