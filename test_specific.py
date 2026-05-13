import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
django.setup()

from chatbot.views import get_ai_response

# Test the exact questions from the screenshot
questions = [
    "Which crops are good in November?",
    "pesticides for sugarcane"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {get_ai_response(q)}")
    print("-" * 80)