from django.urls import path
from .views import process_audio

urlpatterns = [
    path('process-audio/', process_audio, name='process-audio'),
]
