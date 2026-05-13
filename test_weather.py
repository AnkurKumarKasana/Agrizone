#!/usr/bin/env python
import os
import sys
import django
import json
import requests

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrizone.settings')
django.setup()

def test_weather_api():
    """Test the weather API endpoint"""
    url = 'http://127.0.0.1:8000/weather/get-weather/'
    data = {'city': 'Delhi'}
    
    try:
        response = requests.post(url, json=data, timeout=10)
        result = response.json()
        
        print("Weather API Test Results:")
        print("=" * 40)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get('success'):
            print("\n✅ Weather API is working!")
            if result.get('demo'):
                print("📝 Using demo data (API key not active)")
            else:
                print("🌐 Using real weather data")
        else:
            print(f"\n❌ Weather API error: {result.get('error')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Django server is not running. Start with: python manage.py runserver")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == '__main__':
    test_weather_api()