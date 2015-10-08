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

import sys
import yaml
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy

# models and trainers
from sklearn.svm import LinearSVC
from pystruct.models import GraphCRF, ChainCRF
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM


class Features:
    features_matrix = np.zeros((161, 4))
    time = []
    intensity = []
    f1 = []
    f2 = []
    f3 = []

    def __init__(self):
        self.time = []
        self.intensity = []
        self.f1 = []
        self.f2 = []
        self.f3 = []

    def getObject(self, n):
        if n == 0:
            return self.time
        if n == 1:
            return self.intensity
        if n == 2:
            return self.f1
        if n == 3:
            return self.f2
        if n == 4:
            return self.f3

    def setObject(self, n, val):
        if n == 0:
            self.time.append(val)
        if n == 1:
            self.intensity.append(val)
        if n == 2:
            self.f1.append(val)
        if n == 3:
            self.f2.append(val)
        if n == 4:
            self.f3.append(val)

    def get_matrix(self):
        for t, val in enumerate(self.time):
            self.features_matrix[t] = (self.intensity[t], self.f1[t], self.f2[t], self.f3[t])

        return self.features_matrix


class CRF_HMM:
    train_dictionary_phonemes_directory = "output-data/train_audio_phonemes_labels.txt"
    test_dictionary_phonemes_directory = "output-data/test_audio_phonemes_labels.txt"
    train_csv_directory = "output-data/train-smoothed-csv-files/"
    test_csv_directory = "output-data/test-smoothed-csv-files/"

    X_train = []  # sample for training
    y_train = []  # training labels
    X_test = []  # samples for testing
    y_test = []  # testing labels

    dictionary_trainset = {}
    dictionary_testset = {}

    def load_test_phonemes_dictionary(self):
        with open(self.test_dictionary_phonemes_directory) as data_file:
            self.dictionary_testset = yaml.load(data_file)

    def load_train_phonemes_dictionary(self):
        with open(self.train_dictionary_phonemes_directory) as data_file:
            self.dictionary_trainset = yaml.load(data_file)

    def load_trainig_set(self, isTest=False):
        if isTest:
            csv_directory = self.test_csv_directory
        else:
            csv_directory = self.train_csv_directory

        counter = 0
        for filename in os.listdir(csv_directory):
            file_directory = os.path.join(csv_directory, filename)

            if ".DS_Store" in filename:
                continue

            with open(file_directory, 'rU') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                feat = Features()

                i = 0
                try:
                    for row in csvreader:
                        if i == 0:
                            i += 1
                            continue

                        feat.setObject(0, float(row[0]))
                        feat.setObject(1, float(row[1]))
                        feat.setObject(2, float(row[2]))
                        feat.setObject(3, float(row[3]))
                        feat.setObject(4, float(row[4]))

                except:
                    print "Error: ", sys.exc_info()
                    raise

                if isTest:
                    phonemes_key = filename.replace('.csv', '.TextGrid')
                    phonemes_values = self.dictionary_testset[phonemes_key]
                    self.X_test.append(deepcopy(feat.get_matrix()))

                    # need to fill the array with 0s in order to have
                    # 161 "labels"
                    initial_arr_length = len(phonemes_values)
                    for i in range(161):
                        if i < initial_arr_length:
                            continue
                        phonemes_values.append(0)

                    self.y_test.append(deepcopy(np.array(phonemes_values)))
                    continue
                else:
                    for i in range(5):
                        if i == 0:
                            continue

                        temp = filename
                        phonemes_key = filename.replace('.csv', '')
                        phonemes_key = phonemes_key + '_' + str(i) + '.TextGrid'
                        phonemes_values = self.dictionary_trainset[phonemes_key]

                        initial_arr_length = len(phonemes_values)
                        for i in range(161):
                            if i < initial_arr_length:
                                continue
                            phonemes_values.append(0)

                        self.X_train.append(deepcopy(feat.get_matrix()))
                        self.y_train.append(deepcopy(np.array(phonemes_values)))

                        counter += 1
                        filename = temp

    def train_model(self):
        try:

            # this model works!
            svm = LinearSVC(dual=True, C=1.0, verbose=0, max_iter=1000)

            start = time()
            svm.fit(np.vstack(self.X_train), np.hstack(self.y_train))
            time_svm = time() - start

            prediction = svm.predict(np.vstack(self.X_test[1]))

            print "Time: %f seconds", time_svm
            print "Score: %f \n" % svm.score(np.vstack(self.X_test), np.hstack(self.y_test))
            # print "Prediction: ", prediction

        except:
            print "Error: ", sys.exc_info()
            raise

    def distance_cost_plot(self, distances):
        im = plt.imshow(distances, interpolation='nearest', cmap='Reds')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()

    def dynamicTimeWarp(self, train, test):

        for t in xrange(4):
            # get sequences for each feature
            x = train[:, t]
            y = test[:, t]

            # plt.plot(x, 'r', label='x')
            # plt.plot(y, 'g', label='y')
            # plt.legend()
            # plt.show()

            # matrix to compute the distances
            distances = np.zeros((len(y), len(x)))

            # euclidean distance
            for i in range(len(y)):
                for j in range(len(x)):
                    distances[i, j] = (x[j] - y[i]) ** 2

            accumulated_cost = np.zeros((len(y), len(x)))
            for i in range(1, len(y)):
                for j in range(1, len(x)):
                    accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                 accumulated_cost[i, j - 1]) + distances[i, j]

            path = [[len(x) - 1, len(y) - 1]]
            i = len(y) - 1
            j = len(x) - 1
            while i > 0 and j > 0:
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    if accumulated_cost[i - 1, j] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                         accumulated_cost[i, j - 1]):
                        i -= 1
                    elif accumulated_cost[i, j - 1] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                           accumulated_cost[i, j - 1]):
                        j -= j - 1
                    else:
                        i -= 1
                        j -= 1
                path.append([j, i])
            path.append([0, 0])

            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]

            self.distance_cost_plot(accumulated_cost)
            plt.plot(path_x, path_y)
            # plt.show()

    def DTW(self):
        # compare each feature
        a_piece_of_cake_train = self.X_train[0]
        a_piece_of_cake_test = self.X_test[7]

        self.dynamicTimeWarp(a_piece_of_cake_train, a_piece_of_cake_test)

    def test(self):
        self.load_train_phonemes_dictionary()
        self.load_test_phonemes_dictionary()
        self.load_trainig_set()
        self.load_trainig_set(True)
        self.train_model()

        print "DTW: \n"
        self.DTW()


if __name__ == "__main__":
    goofy = CRF_HMM()
    goofy.test()
