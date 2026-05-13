import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
django.setup()

from chatbot.views import get_gemini_response

def test_gemini():
    print("Testing Gemini API...")
    
    test_questions = [
        "Best fertilizer for tomatoes?",
        "How to control rice pests?",
        "When to plant wheat?",
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = get_gemini_response(question)
        if response:
            print(f"A: {response[:150]}...")
        else:
            print("A: No response from Gemini API")

if __name__ == "__main__":
    test_gemini()