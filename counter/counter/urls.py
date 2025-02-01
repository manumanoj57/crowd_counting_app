from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    
    
    path('crowd_detection/', include('crowd_detection.urls')),
]
