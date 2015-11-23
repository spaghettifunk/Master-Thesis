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

import json
import random
import glob
import datetime

from django.http import HttpResponse
from rest_framework.decorators import api_view
from .models import User, UserHistory
from machine_learning.prepare_data import *
from machine_learning.GMM_system import GMM_prototype

data_directory = "data/"


# Login prcess
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


def get_sentence():
    sentences = ["A piece of cake", "Blow a fuse", "Catch some zs", "Down to the wire", "Eager beaver",
                 "Fair and square", "Get cold feet", "Mellow out", "Pulling your legs", "Thinking out loud"]
    phonetic = ["ɐ pˈiːs ʌv kˈeɪk", "blˈoʊ ɐ fjˈuːz", "kˈætʃ sˌʌm zˌiːˈɛs", "dˌaʊn tə ðə wˈaɪɚ", "ˈiːɡɚ bˈiːvɚ",
                "fˈɛɹ ænd skwˈɛɹ", "ɡɛt kˈoʊld fˈiːt", "mˈɛloʊ ˈaʊt", "pˈʊlɪŋ jʊɹ lˈɛɡz", "θˈɪŋkɪŋ ˈaʊt lˈaʊd"]
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
            all_users = User.objects.all()
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

            new_user = User(username=username, password=password, gender=gender, nationality=nationality,
                            occupation=occupation)
            new_user.save()

            sentence, phonetic = get_sentence()

            response["Response"] = "SUCCESS"
            response["Sentence"] = sentence
            response["Phonetic"] = phonetic

            return HttpResponse(json.dumps(response))
    except:
        print "Error: ", sys.exc_info()
        raise


# noinspection PyTypeChecker
@api_view(['POST'])
def test_pronunciation(request):
    try:
        response = {}
        if request.method != 'POST':
            raise

        path = os.path.dirname(os.path.abspath(__file__))
        audio_path = path + "/audio/"

        json_data = json.loads(request.body)
        data = dict(json_data)

        audiofile_string = data['FileAudio']
        phonemes = data['PredictedPhonemes']
        sentence = data['Sentence']

        audiofile_byte = base64.b64decode(audiofile_string)
        random_hash = random.getrandbits(128)

        generate_filename = sentence.replace(' ', '_')
        temp_audiofile = audio_path + generate_filename + "_" + str(random_hash) + ".wav"

        with open(temp_audiofile, 'wb') as output:
            output.write(audiofile_byte)

        user_data = data['User']
        user = json.loads(user_data)

        # need to load the correct model - M or F
        gender = user['gender']
        response['Response'] = 'SUCCESS'

        # TODO: remove the following line related to sentence becuase I'm still testing the same file
        # sentence = "A piece of cake"

        phonemes, vowel_stress, result_wer, pitch_chart, vowel_chart = classify_user_audio(temp_audiofile, phonemes, sentence, gender)

        # save data here on model
        user_model = User.objects.filter(username=user['username'])
        user_history = UserHistory(username=user['username'], sentence=sentence, chart_id=str(random_hash), date=datetime.date.today(), vowels=vowel_chart)
        user_history.save()

        response['Phonemes'] = phonemes
        response['VowelStress'] = vowel_stress
        response['WER'] = result_wer
        response['PitchChart'] = pitch_chart
        response['VowelChart'] = vowel_chart

        # clean up everything
        cleaned_name = temp_audiofile.replace('.wav', '*')

        filelist = glob.glob(cleaned_name)
        for filename in filelist:
            os.remove(filename)

        data_path = path + '/machine_learning/data/'
        (dirName, fileName) = os.path.split(temp_audiofile)
        cleaned_name = data_path + fileName.replace('.wav', '*')

        filelist = glob.glob(cleaned_name)
        for filename in filelist:
            os.remove(filename)

        return HttpResponse(json.dumps(response))

    except User.DoesNotExist:
        print "Error: ", sys.exc_info()
        response['Response'] = 'FAILED'
        response['Reason'] = "User not existing in model"

        return HttpResponse(json.dumps(response))
    except:
        print "Error: ", sys.exc_info()

        response['Response'] = 'FAILED'
        response['Reason'] = sys.exc_info()
        return HttpResponse(json.dumps(response))


def classify_user_audio(audiofile, phonemes, sentence, gender):

    # Force Alignment
    force_alignment(audiofile, sentence)

    # FAVE-exctract
    if gender == 'm':
        extract_data(audiofile, True)
        pitch_binary = get_pitch_contour(audiofile, sentence, True)
    else:
        extract_data(audiofile, False)
        pitch_binary = get_pitch_contour(audiofile, sentence, False)

    # Exctract pronounced phonemes and vowel stress
    phonemes, vowel_stress, result_wer = extract_phonemes(audiofile, sentence, phonemes)

    # Create GMM testing set
    (dirName, fileName) = os.path.split(audiofile)
    test_data = create_test_data(fileName)

    # Test on GMM and get prediction
    X_test, Y_test = create_test_set(test_data)
    plot_filename = audiofile.replace('.wav', '_chart.png')

    gmm_obj = GMM_prototype()
    if gender == 'm':
        vowel_binary = gmm_obj.test_gmm(X_test, Y_test, plot_filename, sentence, False)
    else:
        vowel_binary = gmm_obj.test_gmm(X_test, Y_test, plot_filename, sentence, True)
    return phonemes, vowel_stress, result_wer, pitch_binary, vowel_binary


# History process
@api_view(['POST'])
def fetch_history_data(request):
    try:
        if request.method == 'POST':
            json_data = json.loads(request.body)
            data = dict(json_data)

            response = {}
            username = data['Username']
            sentence = data['Sentence']

            chart_ids = []
            vowels_date = []
            vowels_images = []
            history_data = UserHistory.objects.all()
            for history in history_data:
                if history.username == username and history.sentence == sentence:

                    chart_ids.append(history.chart_id)

                    decoded_str = str(history.vowels).decode('utf-8')
                    json_image = json.dumps(decoded_str, ensure_ascii=False).decode('utf8')
                    vowels_images.append(json_image)
                    vowels_date.append(str(history.date))

            response["Response"] = "SUCCESS"
            response["ChartId"] = chart_ids
            response["VowelsDate"] = vowels_date
            response["VowelsImages"] = vowels_images
            return HttpResponse(json.dumps(response))

    except:
        print "Error: ", sys.exc_info()
        raise
