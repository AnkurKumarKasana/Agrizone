"""
Instructions to get FREE Google Gemini API Key:

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Replace 'your-gemini-api-key-here' in .env file with your actual key

The Gemini API is FREE with generous limits:
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day

This is perfect for your agricultural chatbot!
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_key():
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your-gemini-api-key-here':
        print("❌ No Gemini API key found!")
        print("\n📝 To get a FREE Gemini API key:")
        print("1. Visit: https://makersuite.google.com/app/apikey")
        print("2. Sign in with Google account")
        print("3. Click 'Create API Key'")
        print("4. Copy the key and update .env file")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content("What's the best fertilizer for tomatoes?")
        
        if response and response.text:
            print("✅ Gemini API working perfectly!")
            print(f"Sample response: {response.text[:100]}...")
            return True
        else:
            print("❌ No response from Gemini")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_key()