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

import datetime
import glob
import random
import shutil
import base64
import hashlib
import uuid
import matplotlib.dates as dates
import numpy as np
from rest_framework.decorators import api_view
from machine_learning.GMM_system import GmmPrototype
from machine_learning.prepare_data import *
from .models import User, UserHistory, UserSentenceVowelsTrend, UserReport

data_directory = "data/"
collected_directory = "collected/"


# Login process
@api_view(['POST'])
def login(request):
    try:
        if request.method == 'POST':

            print>> sys.stderr, "*** LOGIN ***"

            json_data = json.loads(request.body)
            data = dict(json_data)

            response = {}
            all_users = User.objects.all()
            for u in all_users:
                if data['Username'] == u.username:

                    # retrieve salt and calculate hashed password
                    salt = u.salt
                    hashed_password = hashlib.sha512(data["Password"] + salt).hexdigest()

                    if hashed_password == u.password:
                        response["Response"] = "SUCCESS"
                        response["Username"] = u.username
                        response["Password"] = u.password
                        response["Gender"] = u.gender
                        response["Nationality"] = u.nationality
                        response["Occupation"] = u.occupation

                        sentence, phonetic = get_sentence()
                        response["Sentence"] = sentence
                        response["Phonetic"] = phonetic

                        log_message = "*** LOGIN SUCCEEDED: " + u.username + " ***"
                        print>> sys.stderr, log_message

                        return HttpResponse(json.dumps(response))

            response = {}
            response["Response"] = "FAILED"
            response["Reason"] = "Something went wrong during the authentication\n Try later"

            log_message = "*** LOGIN FAILED: " + data['Username'] + " ***"
            print>> sys.stderr, log_message

            return HttpResponse(json.dumps(response))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        l = Logger()
        l.log_error("Exception in Login-request", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "Exception in login-process"

        return HttpResponse(json.dumps(response))


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

            print>> sys.stderr, "*** REGISTRATION ***"

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

            # let's protect the password
            salt = uuid.uuid4().hex
            hashed_password = hashlib.sha512(password + salt).hexdigest()

            gender = data['Gender']
            nationality = data['Nationality']
            occupation = data['Occupation']

            new_user = User(username=username, password=hashed_password, salt=salt, gender=gender, nationality=nationality,
                            occupation=occupation)
            new_user.save()

            sentence, phonetic = get_sentence()

            response["Response"] = "SUCCESS"
            response["Sentence"] = sentence
            response["Phonetic"] = phonetic

            log_message = "*** REGISTRATION SUCCEEDED: " + username + " ***"
            print>> sys.stderr, log_message

            return HttpResponse(json.dumps(response))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        l = Logger()
        l.log_error("Exception in Register-request", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "Exception in registration process"

        return HttpResponse(json.dumps(response))


# noinspection PyTypeChecker
@api_view(['POST'])
def test_pronunciation(request):
    try:
        response = {}
        if request.method != 'POST':
            raise

        print>> sys.stderr, "*** TEST PRONUNCIATION ***"

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

        print>> sys.stderr, "*** FILE CONVERTED ***"

        phonemes, vowel_stress, result_wer, normalized_native, normalized_user, vowel_chart = classify_user_audio(
                                                                                                    temp_audiofile,
                                                                                                    phonemes,
                                                                                                    sentence,
                                                                                                    user['username'])
        # save data here on model
        user_history = UserHistory(username=user['username'], sentence=sentence, chart_id=str(random_hash),
                                   date=datetime.date.today(), vowels=vowel_chart)
        user_history.save()

        response['Phonemes'] = phonemes
        response['VowelStress'] = vowel_stress
        response['WER'] = result_wer
        response['YValuesNative'] = normalized_native
        response['YValuesUser'] = normalized_user
        response['VowelChart'] = vowel_chart

        data_path = path + '/machine_learning/data/'
        (dir_name, file_name) = os.path.split(temp_audiofile)
        cleaned_name = data_path + file_name.replace('.wav', '*')

        filelist = glob.glob(cleaned_name)
        for filename in filelist:
            f = os.path.basename(filename)
            shutil.move(filename, data_path + collected_directory + f)

        return HttpResponse(json.dumps(response))

    except User.DoesNotExist:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        l = Logger()
        l.log_error("Exception in build-trend-chart", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "User not existing in model"

        return HttpResponse(json.dumps(response))
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        l = Logger()
        l.log_error("Exception in build-trend-chart", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "Exception in test-pronunciation process"
        return HttpResponse(json.dumps(response))


def classify_user_audio(audiofile, phonemes, sentence, username):
    print >> sys.stderr, "*** START FORCE ALIGNMENT ***"

    # Force Alignment
    force_alignment(audiofile, sentence)

    # FAVE-extract
    print >> sys.stderr, "*** START EXTRACT DATA ***"
    extract_data(audiofile)

    print >> sys.stderr, "*** START PITCH CONTOUR ***"
    normalized_native, normalized_user = get_pitch_contour(audiofile, sentence)

    print >> sys.stderr, "*** START EXTRACT PHONEMES ***"
    # Extract pronounced phonemes and vowel stress
    phonemes, vowel_stress, result_wer = extract_phonemes(audiofile, sentence, phonemes)

    print >> sys.stderr, "*** START CREATION GMM TEST-SET ***"
    # Create GMM testing set
    (dir_name, file_name) = os.path.split(audiofile)
    test_data = create_test_data(file_name)

    # Test on GMM and get prediction
    X_test, Y_test = create_test_set(test_data)
    plot_filename = audiofile.replace('.wav', '_chart.png')

    print >> sys.stderr, "*** START GMM TEST ***"
    gmm_obj = GmmPrototype()
    vowel_binary, trend_data = gmm_obj.test_gmm(X_test, Y_test, plot_filename, sentence)

    print >> sys.stderr, "*** START BUILD TREND CHART ***"
    # build trend chart
    build_trend_chart(username, sentence, trend_data)

    return phonemes, vowel_stress, result_wer, normalized_native, normalized_user, vowel_binary


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

        l = Logger()
        l.log_error("Exception in build-trend-chart", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response = {'Response': 'FAILED', 'Reason': "Exception in build-chart-process"}
        return HttpResponse(json.dumps(response))


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

            # region Trend
            trend_images_title = []
            trend_images = []
            trend_images_time = []
            trend_data = UserSentenceVowelsTrend.objects.filter(username=username, sentence=sentence)
            vowels = vowels.replace(' ', '')
            native_vowels = vowels.split(',')

            for trend in trend_data:
                if trend.vowel in native_vowels:

                    x_axis = []
                    y_axis = []
                    x_values_str = []
                    x_index = 0
                    for data in json.loads(trend.trend_values):
                        x_axis.append(x_index)
                        y_axis.append(str(data[2]))
                        x_index += 1
                        x_values_str.append(str(data[3]))

                    print>> sys.stderr, "*** DUMPING Y_AXIS ***"

                    title = "Vowel: " + trend.vowel
                    trend_images_title.append(title)
                    trend_images.append(json.dumps(y_axis))
                    trend_images_time.append(json.dumps(x_values_str))

                    print>> sys.stderr, "*** DUMPED ***"

            # endregion

            response["Response"] = "SUCCESS"
            response["ChartId"] = history_chart_ids
            response["VowelsDate"] = history_vowels_date
            response["VowelsImages"] = history_vowels_images

            response["TrendImagesTitle"] = trend_images_title
            response["TrendImages"] = trend_images
            response["TrendImagesTime"] = trend_images_time

            print>> sys.stderr, "*** DUMPING RESPONSE ***"
            res_json = json.dumps(response)
            print>> sys.stderr, "*** DUMPED RESPONSE ***"

            return HttpResponse(res_json)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        l = Logger()
        l.log_error("Exception in fetch-history-request", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "Exception in fetch-history process"

        return HttpResponse(json.dumps(response))


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

        l = Logger()
        l.log_error("Exception in save-report-request", str(traceback.print_exc()) + "\n\n" + fname + " " + str(exc_tb.tb_lineno))

        response['Response'] = 'FAILED'
        response['Reason'] = "Exception in save-report process"

        return HttpResponse(json.dumps(response))
