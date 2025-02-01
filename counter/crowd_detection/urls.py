from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import index,video_feed, get_people_count, process_image

urlpatterns = [
    path('', index, name='index'),  # Load HTML page
    path('video_feed/', video_feed, name='video_feed'),
    path('get_people_count/', get_people_count, name='get_people_count'),
    path('process_image/', process_image, name='process_image'),
]

# Serve media files
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
