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

# models and trainers
from sklearn.svm import LinearSVC
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM


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
    csv_directory = "output-data/smoothed-csv-files/train/"

    X_train = []
    y_train = []

    dictionary_trainset = {}
    dictionary_testset = {}

    def load_test_phonemes_dictionary(self):
        with open(self.test_dictionary_phonemes_directory) as data_file:
            self.dictionary_testset = yaml.load(data_file)

    def load_train_phonemes_dictionary(self):
        with open(self.train_dictionary_phonemes_directory) as data_file:
            self.dictionary_trainset = yaml.load(data_file)

    def load_trainig_set(self):
        counter = 0
        for filename in os.listdir(self.csv_directory):
            file_directory = os.path.join(self.csv_directory, filename)

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

                self.X_train.append(feat.get_matrix())
                self.y_train.append(np.array(phonemes_values))
                counter += 1
                filename = temp

    def train_model(self):
        try:

            svm = LinearSVC(dual=False, C=.1)
            svm.fit(np.vstack(self.X_train), np.hstack(self.y_train))

            print("Test score with chain CRF: %f" % svm.score(np.vstack(self.X_train), np.hstack(self.y_train)))

        except:
            print "Error: ", sys.exc_info()
            raise

    def test(self):
        self.load_train_phonemes_dictionary()
        self.load_test_phonemes_dictionary()
        self.load_trainig_set()
        self.train_model()


if __name__ == "__main__":
    goofy = CRF_HMM()
    goofy.test()
