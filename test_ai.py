#!/usr/bin/env python3
"""
Test script to verify OpenAI API integration
"""
import os
import sys
import django
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
django.setup()

from chatbot.views import get_ai_response

def test_ai_responses():
    """Test various agricultural questions"""
    test_questions = [
        "What's the best fertilizer for tomatoes?",
        "How do I control pests in my rice field?",
        "When should I plant corn?",
        "What causes yellowing leaves in plants?",
        "How to improve soil health naturally?",
        "What's the weather like today?",  # Non-agricultural question
        "Tell me about sustainable farming practices"
    ]
    
    print("🌱 Testing AgriGPT AI Assistant")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 30)
        
        try:
            response = get_ai_response(question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")

if __name__ == "__main__":
    test_ai_responses()