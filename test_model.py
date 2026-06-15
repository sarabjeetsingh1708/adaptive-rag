from dotenv import load_dotenv
load_dotenv()

import os
from google import genai

key = os.getenv("GEMINI_API_KEY")

print("Key:", key[:10] + "...")

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="hello"
)

print(response.text)