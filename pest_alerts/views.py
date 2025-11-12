from django.shortcuts import render
from django.http import JsonResponse
from .models import PestAlert
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def alerts_home(request):
    return render(request, 'pest_alerts/alerts.html')

def get_alerts(request):
    crop = request.GET.get('crop', '').lower()
    location = request.GET.get('location', '').strip()
    
    if not crop or not location:
        return JsonResponse({'success': False, 'error': 'Please enter both crop and location'})
    
    # Get weather data
    weather_data = get_weather_data(location)
    
    # Use fallback weather data if API fails
    if not weather_data:
        weather_data = {
            'temp': 28.0,
            'humidity': 75.0,
            'description': 'typical conditions (API unavailable)'
        }
    
    # Find matching alerts
    alerts = find_matching_alerts(crop, weather_data)
    
    return JsonResponse({
        'success': True,
        'alerts': alerts,
        'weather': weather_data
    })

def get_weather_data(location):
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return None
            
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['current']['temp_c'],
                'humidity': data['current']['humidity'],
                'description': data['current']['condition']['text']
            }
    except Exception as e:
        print(f"Weather API Error: {e}")
    return None

def find_matching_alerts(crop, weather):
    temp = weather['temp']
    humidity = weather['humidity']
    
    alerts = PestAlert.objects.filter(crop__iexact=crop)
    matching_alerts = []
    
    for alert in alerts:
        match = True
        
        if alert.min_temp and temp < alert.min_temp:
            match = False
        if alert.max_temp and temp > alert.max_temp:
            match = False
        if alert.min_humidity and humidity < alert.min_humidity:
            match = False
        if alert.max_humidity and humidity > alert.max_humidity:
            match = False
            
        if match:
            matching_alerts.append({
                'pest_name': alert.pest_name,
                'disease_name': alert.disease_name,
                'symptoms': alert.symptoms,
                'prevention': alert.prevention,
                'treatment': alert.treatment,
                'severity': alert.severity
            })
    
    return matching_alerts
