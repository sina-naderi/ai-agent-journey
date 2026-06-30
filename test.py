
from ollamafreeapi import OllamaFreeAPI

client = OllamaFreeAPI()

# List available models
models = client.list_models()
for m in models:
    print(m)