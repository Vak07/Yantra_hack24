from django.shortcuts import render
from .calculation_items import *
from django.http import HttpResponse,JsonResponse
from .calculation_items import *
# Create your views here.
def check(request):
   if request.method == 'GET':
    return HttpResponse('Hi')
# def respond(request):
def give_insights(request):
  if request.method=='POST':
    r=total_transactions()
    total_trans=r[0]
    merchant=r[1]
    agent=r[1]
    susp=suspicion_data()
    return JsonResponse({"suspicious data":susp,"merchants":merchant,"customers":merchant,"total_transactions":total_trans})

