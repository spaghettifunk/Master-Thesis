import sys
import json

from django.http import HttpResponse
from rest_framework.decorators import api_view
from .models import User, GeneralScore

# login process
@api_view(['POST'])
def login(request):
    try:
        if request.method == 'POST':
            json_data = json.loads(request.body)
            data = dict(json_data)

            all_users = User.objects.all()
            for u in all_users:
                if data['Username'] == u.username:
                    score = get_score(u)
                    response = json.dumps(score)
                    return HttpResponse(response)

            return HttpResponse("FAILED")
    except:
        print "Error: ", sys.exc_info()
        raise

def get_score(user):
    try:
        all_scores = GeneralScore.objects.all()
        for sc in all_scores:
            if user.id == sc.id:
                return sc.total_score
        return -1
    except:
        print "Error: ", sys.exc_info()
        raise

# Registration process
@api_view(['POST'])
def register(request):
    try:
        if request.method == 'POST':
            json_data = json.loads(request.body)
            data = dict(json_data)

            username = data['Username']
            password = data['Password']
            gender = data['Gender']
            nationality = data['Nationality']
            occupation = data['Occupation']

            new_user = User(username=username, password=password, gender=gender, nationality=nationality, occupation=occupation)
            new_user.save()

            return HttpResponse("SUCCESS")
    except:
        print "Error: ", sys.exc_info()
        raise

@api_view(['POST'])
def test_pronunciation(request):
    try:
        return HttpResponse("SUCCESS")
    except:
        print "Error: ", sys.exc_info()
        return HttpResponse("FAILED")

