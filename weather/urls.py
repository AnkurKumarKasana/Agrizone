from django.urls import path
from . import views

app_name = 'weather'

urlpatterns = [
    path('', views.weather_home, name='weather_home'),
    path('get-weather/', views.get_weather, name='get_weather'),
]