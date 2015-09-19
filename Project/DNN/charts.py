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


import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Charts:

    def test(self):
        time = []
        pitch = []
        intensity = []
        f1 = []
        f2 = []
        f3 = []

        with open('train-audio-data/test/Jeremy_a_piece_of_cacke_1.csv', 'rU') as csvfile:
            spamreader = csv.reader(csvfile, dialect=csv.excel_tab, delimiter=';', quotechar='^')

            i = 0
            for row in spamreader:
                if i == 0:
                    i += 1
                    continue

                time.append(float(row[0]))
                pitch.append(float(row[1]))
                intensity.append(float(row[2]))
                f1.append(float(row[2]))
                f2.append(float(row[2]))
                f3.append(float(row[2]))

        # plot (x_axis, y_axis, type_of_plot)
        plt.figure(1)
        plt.title("Jeremy_a_piece_of_cacke_1.csv")
        plt.subplot(511)
        plt.xlabel('time')
        plt.ylabel('pitch')
        plt.plot(time, pitch, 'bo')

        plt.subplot(512)
        plt.xlabel('time')
        plt.ylabel('intensity')
        plt.plot(time, intensity, 'yo-')

        plt.subplot(513)
        plt.xlabel('time')
        plt.ylabel('f1')
        plt.plot(time, f1, 'r.-')

        #plt.axis([0, 6, 0, 20])
        plt.subplot(514)
        plt.xlabel('time')
        plt.ylabel('f2')
        plt.plot(time, f2, 'r.-')


        plt.subplot(515)
        plt.xlabel('time')
        plt.ylabel('f3')

        # the histogram of the data
        # pos = np.arange(len(time))
        # width = 1.0     # gives histogram aspect to the bar diagram
        #
        # ax = plt.axes()
        # ax.set_xticks(pos + (width / 2))
        # ax.set_xticklabels(time)
        #
        # plt.bar(pos, f3, width, color='r')

        #y = mlab.normpdf(bins)
        # add a 'best fit' line
        plt.plot(time, f3, 'r--')
        #plt.plot(time, f3, 'r.-')


        plt.show()