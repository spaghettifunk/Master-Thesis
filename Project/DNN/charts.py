'''
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
'''

import csv
import matplotlib.pyplot as plt
import sys
import os

class Features:
    time = []
    pitch = []
    intensity = []
    f1 = []
    f2 = []
    f3 = []

    def __init__(self):
        self.time = []
        self.pitch = []
        self.intensity = []
        self.f1 = []
        self.f2 = []
        self.f3 = []

    def getObject(self, n):
        if n == 0:
            return self.time
        if n == 1:
            return self.pitch
        if n == 2:
            return self.intensity
        if n == 3:
            return self.f1
        if n == 4:
            return self.f2
        if n == 5:
            return self.f3

    def setObject(self, n, val):
        if n == 0:
            self.time.append(val)
        if n == 1:
            self.pitch.append(val)
        if n == 2:
            self.intensity.append(val)
        if n == 3:
            self.f1.append(val)
        if n == 4:
            self.f2.append(val)
        if n == 5:
            self.f3.append(val)

class Charts:

    def test(self):
        dir = "train-audio-data/results/"

        SELECTOR = 5

        for i in range(10):
            FILENAME_SELECTOR = i
            features_title = ["Time", "Pitch", "Intensity", "F1", "F2", "F3"]
            filename_tile = ["a_piece_of_cake","blow_a_fuse","catch_some_zs","down_to_the_wire","eager_beaver","fair_and_square","get_cold_feet","mellow_out","pulling_your_legs","thinking_out_loud"]

            small_multiples = plt.figure(i)
            plt.subplots_adjust(hspace=.5)

            counter = 1
            for dir_entry in os.listdir(dir):
                fileName = os.path.join(dir, dir_entry)

                if filename_tile[FILENAME_SELECTOR] not in fileName:
                    continue

                splitFilename = fileName.split("_")
                splitPersonName = splitFilename[0].split("/")
                personName = splitPersonName[2]
                temp_fileName = fileName.replace(personName + "_", "")
                goofy = temp_fileName.split("/")
                temp_fileName = goofy[2]

                with open(fileName, 'rU') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')

                    feat = Features()

                    i = 0
                    try:
                        for row in spamreader:
                            if i == 0:
                                i += 1
                                continue

                            feat.setObject(0, float(row[0]))
                            feat.setObject(1, float(row[1]))
                            feat.setObject(2, float(row[2]))
                            feat.setObject(3, float(row[3]))
                            feat.setObject(4, float(row[4]))
                            feat.setObject(5, float(row[5]))

                    except:
                        print "Error: ", sys.exc_info()
                        raise

                    mini = small_multiples.add_subplot(3, 4, counter)
                    mini.set_xlabel(features_title[0])
                    mini.set_ylabel(features_title[SELECTOR])
                    mini.set_title(personName)

                    #print "Time: ", len(feat.getObject(0))
                    #print "F3: ", len(feat.getObject(SELECTOR))

                    mini.plot(feat.getObject(0), feat.getObject(SELECTOR), linestyle='-', color='r')
                    counter += 1

            figure_filename = "figures/" + features_title[SELECTOR] + "_" + filename_tile[FILENAME_SELECTOR] + ".png"
            plt.savefig(os.path.join(dir, figure_filename), dpi=300)