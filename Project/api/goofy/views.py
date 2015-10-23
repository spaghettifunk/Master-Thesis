from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View

# Create your views here.
def index(request):
    return HttpResponse('resulteeeeee')

class MyView(View):
    def get(self, request):
        # <view logic>
        return HttpResponse('result')
