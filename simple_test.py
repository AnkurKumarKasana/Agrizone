import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
django.setup()

from chatbot.views import get_ai_response

def test_responses():
    test_questions = [
        "Best fertilizer for tomatoes?",
        "How to control pests in rice?",
        "When to plant crops in June?",
        "Tomato disease management",
        "Soil pH improvement",
        "Organic farming methods"
    ]
    
    print("Testing AgriGPT AI Assistant")
    print("=" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Q: {question}")
        print("-" * 30)
        response = get_ai_response(question)
        print(f"A: {response}")
    
    print("\n" + "=" * 40)
    print("Testing completed!")

if __name__ == "__main__":
    test_responses()