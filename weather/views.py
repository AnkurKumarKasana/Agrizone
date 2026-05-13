from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import requests
import json

def weather_home(request):
    return render(request, 'weather/weather.html')

@csrf_exempt
def get_weather(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            city = data.get('city', '').strip()
            
            if not city:
                return JsonResponse({'success': False, 'error': 'City name is required'})
            
            # Try WeatherAPI first
            api_key = settings.OPENWEATHER_API_KEY
            if api_key and api_key != 'your-openweather-api-key':
                try:
                    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        weather_data = response.json()
                        
                        result = {
                            'success': True,
                            'city': weather_data['location']['name'],
                            'country': weather_data['location']['country'],
                            'temperature': round(weather_data['current']['temp_c'], 1),
                            'feels_like': round(weather_data['current']['feelslike_c'], 1),
                            'humidity': weather_data['current']['humidity'],
                            'pressure': round(weather_data['current']['pressure_mb']),
                            'description': weather_data['current']['condition']['text'],
                            'icon': '01d',
                            'wind_speed': round(weather_data['current']['wind_kph'] / 3.6, 1),
                            'wind_direction': weather_data['current']['wind_dir'],
                            'visibility': round(weather_data['current']['vis_km'], 1),
                            'uv_index': weather_data['current']['uv'],
                            'cloud_cover': weather_data['current']['cloud'],
                            'precipitation': weather_data['current']['precip_mm'],
                            'dewpoint': round(weather_data['current']['dewpoint_c'], 1),
                        }
                        return JsonResponse(result)
                    elif response.status_code == 401:
                        # Invalid API key - fall through to demo data
                        pass
                    else:
                        # Other API errors - fall through to demo data
                        pass
                except requests.exceptions.RequestException:
                    # Network errors - fall through to demo data
                    pass
                except (KeyError, ValueError, TypeError):
                    # JSON parsing or data structure errors - fall through to demo data
                    pass
            
            # Fallback to demo weather data
            import random
            demo_weather = {
                'success': True,
                'city': city.title(),
                'country': 'DEMO',
                'temperature': round(random.uniform(15, 35), 1),
                'humidity': random.randint(40, 80),
                'pressure': random.randint(1000, 1020),
                'description': random.choice(['Clear Sky', 'Few Clouds', 'Scattered Clouds', 'Partly Cloudy']),
                'icon': random.choice(['01d', '02d', '03d', '04d']),
                'wind_speed': round(random.uniform(2, 15), 1),
                'visibility': round(random.uniform(8, 15), 1),
                'demo': True
            }
            
            return JsonResponse(demo_weather)
                
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid request format'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Service error: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})