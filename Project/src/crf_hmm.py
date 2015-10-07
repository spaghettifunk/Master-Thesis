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

    X_train = []    # sample for training
    y_train = []    # training labels
    X_test = []     # samples for testing
    y_test = []     # testing labels

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
            svm = LinearSVC(dual=True, C=1.0, verbose=0, max_iter=1000, random_state=3)

            start = time()
            svm.fit(np.vstack(self.X_train), np.hstack(self.y_train))
            time_svm = time() - start

            prediction = svm.predict(np.vstack(self.X_test[1]))

            print "Time: %f seconds", time_svm
            print "Score: %f" % svm.score(np.vstack(self.X_test), np.hstack(self.y_test))
            print "Prediction: ", prediction

        except:
            print "Error: ", sys.exc_info()
            raise

    def test(self):
        self.load_train_phonemes_dictionary()
        self.load_test_phonemes_dictionary()
        self.load_trainig_set()
        self.load_trainig_set(True)
        self.train_model()


if __name__ == "__main__":
    goofy = CRF_HMM()
    goofy.test()
