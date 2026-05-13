import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_openai():
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "What's the best fertilizer for tomatoes?"}],
                max_tokens=100
            )
            print("✅ OpenAI API working!")
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
    else:
        print("❌ No API key found")

if __name__ == "__main__":
    test_openai()