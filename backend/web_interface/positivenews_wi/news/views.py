from json import JSONDecodeError

from django.shortcuts import render
from django.http import HttpResponse
from django.core import serializers
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from news.models import News

import json


@csrf_exempt
def fetch_news(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            sss = News.objects.all()
            response = HttpResponse(
                serializers.serialize("json", News.objects.filter(news_date__gte=body["news_date"])))
        except JSONDecodeError:
            response = HttpResponse("Json Decode Error, badly formatted body")
        except Exception as e:
            response = HttpResponse("Jsadsay")
            print(e)

    else:
        response = HttpResponse("POST methods expected")
    return response
