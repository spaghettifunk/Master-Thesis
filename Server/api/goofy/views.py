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

import matplotlib.dates as dates

from django.http import HttpResponse
from rest_framework.decorators import api_view
from machine_learning.prepare_data import *
from machine_learning.GMM_system import GMM_prototype

from .models import User, UserHistory, UserSentenceVowelsTrend, UserReport

data_directory = "data/"


# Login process
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

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
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

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
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

        phonemes, vowel_stress, result_wer, pitch_chart, vowel_chart = classify_user_audio(temp_audiofile,
                                                                                           phonemes,
                                                                                           sentence,
                                                                                           user['username'],
                                                                                           gender)
        # save data here on model
        user_history = UserHistory(username=user['username'], sentence=sentence, chart_id=str(random_hash),
                                   date=datetime.date.today(), vowels=vowel_chart)
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


def classify_user_audio(audiofile, phonemes, sentence, username, gender):
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
        vowel_binary, trend_data = gmm_obj.test_gmm(X_test, Y_test, plot_filename, sentence, False)
    else:
        vowel_binary, trend_data = gmm_obj.test_gmm(X_test, Y_test, plot_filename, sentence, True)

    # build trend chart
    build_trend_chart(username, sentence, trend_data)

    return phonemes, vowel_stress, result_wer, pitch_binary, vowel_binary


def build_trend_chart(username, sentence, trend_data):
    try:

        for pred_vowel, actual_vowel, distance, date in trend_data:
            chart_from_db = UserSentenceVowelsTrend.objects.filter(username=username, sentence=sentence,
                                                                   vowel=actual_vowel)

            if len(chart_from_db) == 0:
                # insert new data
                value_tuple = [(pred_vowel, actual_vowel, distance, date)]
                trend_values_json = json.dumps(value_tuple)
                new_data_db = UserSentenceVowelsTrend(username=username,
                                                      sentence=sentence,
                                                      vowel=actual_vowel,
                                                      trend_values=trend_values_json,
                                                      trend=None)
                new_data_db.save()
            else:
                # update values
                old_trend_data_json = chart_from_db.get().trend_values
                old_trend_data = json.loads(old_trend_data_json)
                value_tuple = [pred_vowel, actual_vowel, distance, date]
                old_trend_data.append(value_tuple)

                chart_from_db.update(trend_values=json.dumps(old_trend_data))

                for item in chart_from_db:
                    item.save()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise


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
            vowels = data['Vowels']

            # region History
            history_chart_ids = []
            history_vowels_date = []
            history_vowels_images = []
            history_data = UserHistory.objects.all()
            for history in history_data:
                if history.username == username and history.sentence == sentence:
                    history_chart_ids.append(history.chart_id)

                    decoded_str = str(history.vowels).decode('utf-8')
                    json_image = json.dumps(decoded_str, ensure_ascii=False).decode('utf8')
                    history_vowels_images.append(json_image)
                    history_vowels_date.append(str(history.date))
            # endregion

            path = os.path.dirname(os.path.abspath(__file__))
            audio_path = path + "/audio/"

            # region Trend
            trend_images = []
            trend_data = UserSentenceVowelsTrend.objects.filter(username=username, sentence=sentence)
            trend_plot_filename = audio_path + sentence.replace(' ', '_') + "_" + str(random.getrandbits(128)) + ".png"
            vowels = vowels.replace(' ', '')
            native_vowels = vowels.split(',')

            for trend in trend_data:
                if trend.vowel in native_vowels:
                    # build the graph here
                    index = 0

                    plt.figure()
                    x_axis = []
                    y_axis = []
                    x_values_str = []
                    for data in json.loads(trend.trend_values):
                        date_obj = datetime.datetime.strptime(str(data[3]), '%m-%d-%Y %H:%M')
                        x_axis.append(dates.date2num(date_obj))
                        y_axis.append(str(data[2]))
                        x_values_str.append(str(data[3]))

                    plt.xticks(x_axis, x_values_str, rotation=45)
                    plt.plot_date(x_axis, y_axis, tz=None, xdate=True, ydate=False, linestyle='-', marker='D',
                                  color='g')
                    plt.title("Vowel: " + trend.vowel)

                    plt.savefig(trend_plot_filename, bbox_inches='tight', transparent=True)
                    with open(trend_plot_filename, "rb") as imageFile:
                        trend_pic = base64.b64encode(imageFile.read())

                        json_image = json.dumps(trend_pic)
                        trend_images.append(json_image)

                    index += 1
            # endregion

            response["Response"] = "SUCCESS"
            response["ChartId"] = history_chart_ids
            response["VowelsDate"] = history_vowels_date
            response["VowelsImages"] = history_vowels_images

            response["TrendImages"] = trend_images

            # clean things up
            os.remove(trend_plot_filename)

            return HttpResponse(json.dumps(response))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise


# Report process
@api_view(['POST'])
def save_report(request):
    try:
        if request.method == 'POST':
            json_data = json.loads(request.body)
            data = dict(json_data)

            username = data['Username']
            report = data['Report']

            user_report = UserReport.objects.filter(username=username)
            if len(user_report) == 0:
                u = UserReport(username=username, report_values=report)
                u.save()
            else:
                user_report.update(report_values=json.dumps(report))
                for item in user_report:
                    item.save()

            response = {"Response": "SUCCESS", "Result": "OK"}

            return HttpResponse(json.dumps(response))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise
