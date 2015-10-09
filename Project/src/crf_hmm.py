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
import statistics
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# monte
from monte.models.crf import ChainCrfLinear
from monte import train

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
            return self.intesity
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

    def get_samples(self):
        return np.concatenate([self.intensity, self.f1, self.f2, self.f3])


class CRF_HMM:

    #region Global variables
    sentences_directory = "output-data/sentences.txt"
    train_dictionary_phonemes_directory = "output-data/train_audio_phonemes_labels.txt"
    test_dictionary_phonemes_directory = "output-data/test_audio_phonemes_labels.txt"
    train_csv_directory = "output-data/train-smoothed-csv-files/"
    test_csv_directory = "output-data/test-smoothed-csv-files/"

    dictionary_trainset = {}
    dictionary_testset = {}

    # DTW stuff related
    DTW_X_train = []  # sample for training
    DTW_Y_train = []  # training labels
    DTW_X_test = []  # samples for testing
    DTW_Y_test = []  # testing labels

    # Phonemes prediciton
    PHONEMES_X_train = []  # sample for training
    PHONEMES_Y_train = []  # training labels
    PHONEMES_X_test = []  # samples for testing
    PHONEMES_Y_test = []  # testing labels
    train_labels_to_int = {}  # dictionary for mapping the sentence with an integer -> the integer will be used as label for the classifier
    test_labels_to_int = {}  # dictionary for mapping the sentence with an integer -> the integer will be used as label for the classifier
    #endregion

    def load_test_phonemes_dictionary(self):
        with open(self.test_dictionary_phonemes_directory) as data_file:
            self.dictionary_testset = yaml.load(data_file)

    def load_train_phonemes_dictionary(self):
        with open(self.train_dictionary_phonemes_directory) as data_file:
            self.dictionary_trainset = yaml.load(data_file)

    # DTW methods
    def load_DTW_set(self, isTest=False):
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
                    self.DTW_X_test.append(deepcopy(feat.get_matrix()))

                    # need to fill the array with 0s in order to have
                    # n "labels"
                    initial_arr_length = len(phonemes_values)
                    for i in range(161):
                        if i < initial_arr_length:
                            continue
                        phonemes_values.append(0)

                    self.DTW_Y_test.append(deepcopy(np.array(phonemes_values)))
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

                        self.DTW_X_train.append(deepcopy(feat.get_matrix()))
                        self.DTW_Y_train.append(deepcopy(np.array(phonemes_values)))

                        counter += 1
                        filename = temp

    def distance_cost_plot(self, distances):
        im = plt.imshow(distances, interpolation='nearest', cmap='Reds')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()

    def dynamicTimeWarp(self, train, test):

        features_names = ['Intensity', 'F1', 'F2', 'F3']
        for feat in xrange(4):
            # get sequences for each feature
            x = train[:, feat]
            y = test[:, feat]

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

            path = [[len(x)-1, len(y)-1]]
            i = len(y) - 1
            j = len(x) - 1
            while i > 0 and j > 0:
                if i == 0:
                    j -= 1
                elif j==0:
                    i -= 1
                else:
                    if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        i -= 1
                    elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        j -= 1
                    else:
                        i -= 1
                        j -= 1
                path.append([j, i])
            path.append([0,0])

            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]

            length_x = len(path_x)
            length_y = len(path_y)

            assert length_x == length_y # just to be sure :)

            distance = []
            for i in range(length_x):
                distance.append(abs(path_x[i] - path_y[i]))

            # calculate a value for similarity
            min_distance = min(distance)
            max_distance = max(distance)

            norm = []
            for i in range(len(distance)):
                z = float(distance[i] - min_distance) / float(max_distance - min_distance)
                norm.append(z)

            similarity = 100 * statistics.mean(norm)
            print "Similarity of {}: {}".format(features_names[feat], similarity)

            # now we need to estimate the percentage of difference based on the distance
            # print "*** Plot distance cost ***"
            # self.distance_cost_plot(accumulated_cost)
            # plt.plot(path_x, path_y)

        #plt.show()

    def DTW(self):
        # compare each feature
        a_piece_of_cake_train = self.DTW_X_train[0]
        a_piece_of_cake_test = self.DTW_X_test[7]

        self.dynamicTimeWarp(a_piece_of_cake_train, a_piece_of_cake_test)

    # model and trainer for phonemes prediction
    def load_PHONEMES_set(self, isTest=False):
        if isTest:
            for key, value in self.dictionary_testset.items():
                label = key.replace('.TextGrid', '')  # TODO check if necessary

                phonemes_values = value

                # set same length to each array - 30 should be enough
                initial_arr_length = len(phonemes_values)
                for i in range(30):
                    if i < initial_arr_length:
                        continue
                    phonemes_values.append(-1)

                self.PHONEMES_X_test.append(phonemes_values)
                self.PHONEMES_Y_test.append(self.test_labels_to_int[label])
        else:
            for key, value in self.dictionary_trainset.items():
                label = key.replace('.TextGrid', '')  # TODO check if necessary

                phonemes_values = value

                # set same length to each array - 30 should be enough
                initial_arr_length = len(phonemes_values)
                for i in range(30):
                    if i < initial_arr_length:
                        continue
                    phonemes_values.append(-1)

                self.PHONEMES_X_train.append(phonemes_values)
                self.PHONEMES_Y_train.append(self.train_labels_to_int[label])

    def labels_mapping(self):
        sentences = {}
        with open(self.sentences_directory) as sentences_file:
            lines = sentences_file.readlines()
            label_val = 1  # don't like 0 as label :)
            for s in lines:
                s = s.replace('\n', '')
                sentences[s] = label_val
                label_val += 1

        for label in self.dictionary_trainset.keys():
            label = label.replace('.TextGrid', '')  # TODO check if necessary
            for s in sentences:
                if s in label:
                    self.train_labels_to_int[label] = sentences[s]

        for label in self.dictionary_testset.keys():
            label = label.replace('.TextGrid', '')  # TODO check if necessary
            for s in sentences:
                if s in label:
                    self.test_labels_to_int[label] = sentences[s]

    def train_model(self):
        try:
            # Make a linear-chain CRF:
            print "*** Creating Linear Chain CRF ***\n"
            mycrf = ChainCrfLinear(30,
                                   11)  # 30 - number of dimensions (phonemes + -1s), 11 (number of labels 1 to 10 + 1)

            # Alternatively, we could have used one of these, for example:
            # mytrainer = trainers.OnlinegradientNocost(mycrf,0.95,0.01)
            # mytrainer = trainers.Bolddriver(mycrf,0.01)
            mytrainer = train.GradientdescentMomentum(mycrf, 0.95, 0.01)

            # Produce some stupid toy data for training:
            inputs = np.array(self.PHONEMES_X_train)
            outputs = np.array(self.PHONEMES_Y_train)

            # Train the model. Since we have registered our model with this trainer,
            # calling the trainers step-method trains our model (for a number of steps):
            print "*** Training model ***"
            for i in range(20):
                mytrainer.step((inputs, outputs), 0.001)
                print "Cost: ", mycrf.cost((inputs, outputs), 0.001)

            # Apply to some stupid test data:
            testinputs = np.array(self.PHONEMES_X_test)
            predictions = mycrf.viterbi(testinputs)
            print "\nLabels predicted: ", predictions  # each element corrisponds to a sentences in the sentences.txt file - Index starts from 1!!

            # JUST FOR TESTING!!!!
            reference = [[5, 10, 36, 3, 5, 20, 2, 7, 2],  # a piece of cake
                         [18, 4, 13, 5, 34, 60, 43, 14],  # blow a fuse
                         [2, 26, 42, 3, 17, 6, 14, 39, 14],  # catch some zs
                         [16, 53, 9, 12, 5, 35, 5, 15, 24, 25],  # down to the wire
                         [36, 30, 25, 18, 36, 20, 25],  # eager beaver
                         [34, 19, 23, 5, 9, 16, 3, 2, 15, 19, 23],  # fair and square
                         [30, 19, 12, 2, 13, 4, 16, 34, 36, 12],  # get cold feet
                         [6, 19, 4, 49, 53, 12],  # mellow out
                         [10, 38, 4, 31, 29, 60, 38, 23, 4, 19, 30, 14],  # pulling your legs
                         [32, 21, 29, 2, 5, 29, 53, 12, 4, 53, 16]]  # thinking out loud

            print "*** Applying WER ***"
            counter = 0
            for val in self.PHONEMES_X_test:
                test_phonemes = val  # hyphothesis
                wer_result, numCor, numSub, numIns, numDel = self.wer(reference[counter], test_phonemes)
                print "WER: {}, OK: {}, SUB: {}, INS: {}, DEL: {}".format(wer_result, numCor, numSub, numIns, numDel)
                # print "WER distance: ", self.wer(reference[counter], test_phonemes)

        except:
            print "Error: ", sys.exc_info()
            raise

    def wer(self, ref, hyp, debug=False):
        DEL_PENALTY = 2
        SUB_PENALTY = 1
        INS_PENALTY = 3

        r = ref  # .split()
        h = hyp  # .split()
        # costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r) + 1):
            costs[i][0] = DEL_PENALTY * i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    costs[i][j] = costs[i - 1][j - 1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                    insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                    deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i] + "\t" + "****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(numSub))
            print("#del " + str(numDel))
            print("#ins " + str(numIns))
            return (numSub + numDel + numIns) / (float)(len(r))

        wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
        return wer_result, numCor, numSub, numIns, numDel

    def run(self):
        # modeling
        print "*** Loading dictionaries ***"
        self.load_train_phonemes_dictionary()
        self.load_test_phonemes_dictionary()

        print "*** Phonemes ***"
        self.labels_mapping()
        self.load_PHONEMES_set()
        self.load_PHONEMES_set(True)
        self.train_model()

        print "\n*** DTW ***"
        self.load_DTW_set()
        self.load_DTW_set(True)
        self.DTW()

        print "\n*** END ***"
