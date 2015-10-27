#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The MIT License

Copyright (c) 2015 University of Rochester, Uppsala University
Authors: Davide Berdin, Philip J. Guo, Olle Galmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import json
import random

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

            response = {}
            all_users = User.objects.all()
            for u in all_users:
                if data['Username'] == u.username:
                    response["Response"] = "SUCCESS"
                    response["Username"] = u.username
                    response["Password"] = u.password
                    response["Gender"] = u.gender
                    response["Nationality"] = u.nationality
                    response["Occupation"] = u.occupation
                    response["Score"] = get_score(u)

                    sentence, phonetic = get_sentence()
                    response["Sentence"] = sentence
                    response["Phonetic"] = phonetic

                    return HttpResponse(json.dumps(response))

            response = {}
            response["Response"] = "FAILED"
            response["Reason"] = "Something went wrong during the authentication\n Try later"
            return HttpResponse(json.dumps(response))
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

def get_sentence():
    sentences = ["A piece of cake", "Blow a fuse", "Catch some zs", "Down to the wire", "Eager beaver", "Fair and square", "Get cold feet", "Mellow out", "Pulling your legs", "Thinking out loud"]
    phonetic = ["ɐ pˈiːs ʌv kˈeɪk", "blˈoʊ ɐ fjˈuːz", "kˈætʃ sˌʌm zˌiːˈɛs", "dˌaʊn tə ðə wˈaɪɚ", "ˈiːɡɚ bˈiːvɚ", "fˈɛɹ ænd skwˈɛɹ", "ɡɛt kˈoʊld fˈiːt", "mˈɛloʊ ˈaʊt", "pˈʊlɪŋ jʊɹ lˈɛɡz", "θˈɪŋkɪŋ ˈaʊt lˈaʊd"]
    index = random.randrange(start=0, stop=len(sentences))

    return sentences[index], phonetic[index]

# Registration process
@api_view(['POST'])
def register(request):
    try:
        if request.method == 'POST':
            json_data = json.loads(request.body)
            data = dict(json_data)

            response = {}
            all_users = User.objects.all();
            username = data['Username']

            for u in all_users:
                if u.username == username:
                    response["Response"] = "FAILED"
                    response["Reason"] = "Username already registered"
                    return HttpResponse(json.dumps(response))

            password = data['Password']
            gender = data['Gender']
            nationality = data['Nationality']
            occupation = data['Occupation']

            new_user = User(username=username, password=password, gender=gender, nationality=nationality, occupation=occupation)
            new_user.save()

            sentence, phonetic = get_sentence()

            response["Response"] = "SUCCESS"
            response["Sentence"] = sentence
            response["Phonetic"] = phonetic

            return HttpResponse(json.dumps(response))
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
