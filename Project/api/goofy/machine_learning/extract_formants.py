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
import sys
from subprocess import Popen


def extract_data(audio_file, female=False):
    # need to change speakerfile for the female gender
    path = os.path.dirname(os.path.abspath(__file__))
    path_fave = path + "/libraries/FAVE_extract/"

    if female:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker speakerinfo_female.speakerfile"
    else:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker speakerinfo_male.speakerfile"

    textgrid_file_directory = path + "/data/"
    output_file_directory = path + "/data/"

    wav_file = audio_file.name
    wav_file_cleaned = wav_file.replace('.wav', '.TextGrid')

    (dirName, fileName) = os.path.split(wav_file_cleaned)

    textgrid_file = os.path.join(textgrid_file_directory, fileName)
    output_file = os.path.join(output_file_directory, fileName.replace('.TextGrid', '.txt'))

    # debug print
    command = "python " + path_fave + "bin/extractFormants.py " + config_file + " " + textgrid_file + " " + output_file
    print command

    try:
        # run command
        Popen(['/usr/local/bin/zsh', '-c', command])
    except:
        print "Error: ", sys.exc_info()
        raise
