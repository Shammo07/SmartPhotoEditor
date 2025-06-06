from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  
    path('process/<int:photo_id>/<str:action>/', views.process_image, name='process_image'),
    path('preview-image/', views.preview_image, name='preview_image'),
]