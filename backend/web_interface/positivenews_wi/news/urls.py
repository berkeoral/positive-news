from django.urls import path

from . import views

urlpatterns = [
    path('fetch_news', views.fetch_news, name='index'),
]

