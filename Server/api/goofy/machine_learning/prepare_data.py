__author__ = 'davideberdin'

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

import os
import csv
import sys
import numpy as np

from subprocess import Popen
from copy import deepcopy


class GMM_structure:
    stress = []
    words = []
    norm_F1 = []
    norm_F2 = []

    def __init__(self):
        self.stress = []
        self.words = []
        self.norm_F1 = []
        self.norm_F2 = []

    def set_object(self, n, val):
        if n == 0:
            self.stress.append(val)
        if n == 1:
            self.words.append(val)
        if n == 2:
            self.norm_F1.append(val)
        if n == 3:
            self.norm_F2.append(val)

    def concat_object(self, n, val):
        if n == 0:
            self.norm_F1 += val
        if n == 1:
            self.norm_F2 += val

    def get_object(self, n):
        if n == 0:
            return self.stress
        if n == 1:
            return self.words
        if n == 2:
            return self.norm_F1
        if n == 3:
            return self.norm_F2


def force_alignment(audio_file, sentence):
    path = os.path.dirname(os.path.abspath(__file__))
    results_directory = path + "/data"
    path_fa = path + "/libraries/force_alignment/"
    path_fa_sentences = path_fa + "sentences/"

    # sentence: A piece of cake -> a_piece_of_cake
    tmp_sentence = sentence.lower()
    phonemes_filename = tmp_sentence.replace(' ', '_')

    # directory containing the txt files with each sentence
    get_sentences_directory = os.path.join(path_fa_sentences, phonemes_filename + '.txt')

    # result of p2fa
    try:
        (dirName, fileName) = os.path.split(audio_file)
        output_filename = os.path.join(results_directory, fileName.replace('.wav', '.TextGrid'))

        # call the file
        command = "python " + path_fa + "align.py " + audio_file + " " + get_sentences_directory + " " + output_filename

        # run command
        proc = Popen(['/usr/local/bin/zsh', '-c', command])
        proc.wait()
    except:
        print "Error: ", sys.exc_info()
        raise


def extract_data(audio_file, female=False):
    # need to change speakerfile for the female gender
    path = os.path.dirname(os.path.abspath(__file__))
    path_fave = path + "/libraries/FAVE_extract/"

    if female:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker " + path_fave + "/speakerinfo_female.speakerfile"
    else:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker " + path_fave + "/speakerinfo_male.speakerfile"

    textgrid_file_directory = path + "/data/"
    output_file_directory = path + "/data/"

    wav_file = audio_file
    wav_file_cleaned = wav_file.replace('.wav', '.TextGrid')

    (dirName, fileName) = os.path.split(wav_file_cleaned)

    textgrid_file = os.path.join(textgrid_file_directory, fileName)
    output_file = os.path.join(output_file_directory, fileName.replace('.TextGrid', '.txt'))

    # debug print
    command = "python " + path_fave + "bin/extractFormants.py " + config_file + " " + audio_file + " " + textgrid_file + " " + output_file
    print command

    try:
        # run command
        proc = Popen(['/usr/local/bin/zsh', '-c', command])
        proc.wait()
    except:
        print "Error: ", sys.exc_info()
        raise


def create_test_data(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    path_data = path + "/data/"

    txt_file = path_data + filename.replace('.wav', '_norm.txt')
    csv_file = path_data + filename.replace('.wav', '.csv')

    # use 'with' if the program isn't going to immediately terminate
    # so you don't leave files open
    # the 'b' is necessary on Windows
    # it prevents \x1a, Ctrl-z, from ending the stream prematurely
    # and also stops Python converting to / from different line terminators
    # On other platforms, it has no effect

    with open(txt_file, "rb") as opened_txt:
        in_txt = csv.reader(opened_txt, delimiter='\t')

        with open(csv_file, 'wb') as opened_csv:
            out_csv = csv.writer(opened_csv)
            out_csv.writerows(in_txt)

    all_data = dict()
    with open(csv_file, 'r') as tabbed_file:
        reader = csv.reader(tabbed_file, delimiter="\t")
        all_lines = list(reader)

        not_included = 0
        for line in all_lines:
            if not_included <= 2:
                not_included += 1
                continue

            l = line[0].split(',')

            data = GMM_structure()
            data.set_object(0, l[1])
            data.set_object(1, l[2])
            try:
                if l[3] == '':
                    f1_val = 0.0
                else:
                    f1_val = float(l[3])

                if l[4] == '':
                    f2_val = 0.0
                else:
                    f2_val = float(l[4])

                data.set_object(2, f1_val)
                data.set_object(3, f2_val)
            except:
                print "Error: ", sys.exc_info()

            if l[0] in all_data:
                # append the new number to the existing array at this slot
                obj = all_data[l[0]]

                # we use it only for phoneme prediction
                obj.concat_object(0, data.norm_F1)
                obj.concat_object(1, data.norm_F2)

                all_data[l[0]] = obj
            else:
                # create a new array in this slot
                all_data[l[0]] = data
    return all_data


def create_test_set(test_data):

    try:
        X_test = test_data.values()
        Y_test = test_data.keys()

        return X_test, Y_test
    except:
        print "Error: ", sys.exc_info()
        raise