import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.5-flash")

response = model.generate_content("как реализовать безопасное хранение API-ключа (.env или переменные окружения)?")

print(response.text)