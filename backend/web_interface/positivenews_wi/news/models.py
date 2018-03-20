from django.db import models

class News(models.Model):
    news_url = models.TextField()
    news_date = models.DateTimeField()
    news_title = models.CharField(max_length=100)
    news_text = models.TextField()
    news_summary = models.TextField()
