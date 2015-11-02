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

from copy import deepcopy


class GMM_structure:
    filename = ""
    vowels = []
    stress = []
    words = []
    norm_F1 = []
    norm_F2 = []

    def __init__(self, fn):
        self.filename = fn
        self.vowels = []
        self.stress = []
        self.words = []
        self.norm_F1 = []
        self.norm_F2 = []

    def set_object(self, n, val):
        if n == 0:
            self.vowels.append(val)
        if n == 1:
            self.stress.append(val)
        if n == 2:
            self.words.append(val)
        if n == 3:
            self.norm_F1.append(val)
        if n == 4:
            self.norm_F2.append(val)

    def get_object(self, n):
        if n == 0:
            return self.vowels
        if n == 1:
            return self.stress
        if n == 2:
            return self.words
        if n == 3:
            return self.norm_F1
        if n == 4:
            return self.norm_F2


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

    all_data = []
    with open(csv_file, 'r') as tabbed_file:
        reader = csv.reader(tabbed_file, delimiter="\t")
        all_lines = list(reader)

        data = GMM_structure(file)
        not_included = 0
        for l in all_lines:
            if not_included <= 2:
                not_included += 1
                continue

            data_split = l[0].split(",")

            data.set_object(0, data_split[0])
            data.set_object(1, data_split[1])
            data.set_object(2, data_split[2])
            try:
                if data_split[3] == '':
                    f1_val = 0.0
                else:
                    f1_val = float(data_split[3])

                if data_split[4] == '':
                    f2_val = 0.0
                else:
                    f2_val = float(data_split[4])

                data.set_object(3, f1_val)
                data.set_object(4, f2_val)
            except:
                print "Error: ", sys.exc_info()

        all_data.append(deepcopy(data))
    return all_data
