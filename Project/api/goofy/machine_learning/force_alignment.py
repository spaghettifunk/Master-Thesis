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

def force_alignment(audio_file, sentence):

    path = os.path.dirname(os.path.abspath(__file__))
    results_directory = path + "/data/"
    path_fa = path + "/libraries/force_alignment/"
    path_fa_sentences = path_fa + "sentences/"

    # sentence: A piece of cake -> a_piece_of_cake
    tmp_sentence = sentence.lower()
    phonemes_filename = tmp_sentence.replace(' ', '_')

    # directory containing the txt files with each sentence
    get_sentences_directory = os.path.join(path_fa_sentences, phonemes_filename + '.txt')

    # result of p2fa
    try:
        (dirName, fileName) = os.path.split(audio_file.name)
        output_filename = os.path.join(results_directory, fileName.replace('.wav', '.TextGrid'))

        # call the file
        command = "python " + path_fa + "align.py " + audio_file.name + " " + get_sentences_directory + " " + output_filename

        # run command
        Popen(['/usr/local/bin/zsh', '-c', command])
    except:
        print "Error: ", sys.exc_info()
        raise

