from django.urls import path
from . import views

app_name = 'crop_recommendation'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('recommend/', views.recommend, name='recommend'),
    path('predict/', views.predict, name='predict'),
    path('send-contact/', views.send_contact, name='send_contact'),
]