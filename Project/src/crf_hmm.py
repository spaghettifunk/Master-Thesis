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

import itertools
import sys
import csv
from scipy import linalg
from sklearn import mixture
from sklearn.utils import shuffle
from copy import deepcopy

import yaml
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
from fastdtw import fastdtw

from libraries.utility import *

# monte
from libraries.monte.models.crf import ChainCrfLinear
from libraries.monte import train


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


class CRF_DTW:
    # region Global variables
    sentences_directory = "output-data/sentences.txt"
    train_dictionary_phonemes_directory = "output-data/train_audio_phonemes_labels.txt"
    test_dictionary_phonemes_directory = "output-data/test_audio_phonemes_labels.txt"
    train_csv_directory = "output-data/train-smoothed-csv-files/"
    test_csv_directory = "output-data/test-smoothed-csv-files/"
    dtw_comparison_directory = "output-data/dtw_comparison.txt"
    dtw_comparison_native_directory = "output-data/dtw_comparison_native.txt"

    dictionary_trainset = {}
    dictionary_testset = {}

    # DTW stuff related
    DTW_X_train = {}  # sample for training
    DTW_Y_train = {}  # training labels
    DTW_X_test = {}  # samples for testing
    DTW_Y_test = {}  # testing labels

    # Phonemes prediciton
    PHONEMES_X_train = []  # sample for training
    PHONEMES_Y_train = []  # training labels
    PHONEMES_X_test = []  # samples for testing
    PHONEMES_Y_test = []  # testing labels

    train_labels_to_int = {}  # dictionary for mapping the sentence with an integer -> the integer will be used as label for the classifier
    test_labels_to_int = {}  # dictionary for mapping the sentence with an integer -> the integer will be used as label for the classifier
    # endregion

    # region Load dictionaries from file
    def load_test_phonemes_dictionary(self):
        with open(self.test_dictionary_phonemes_directory) as data_file:
            self.dictionary_testset = yaml.load(data_file)

    def load_train_phonemes_dictionary(self):
        with open(self.train_dictionary_phonemes_directory) as data_file:
            self.dictionary_trainset = yaml.load(data_file)

    # endregion

    # region DTW methods
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

                    self.DTW_X_test[phonemes_key] = deepcopy(feat.get_matrix())
                    self.DTW_Y_test[phonemes_key] = deepcopy(np.array(phonemes_values))
                else:
                    temp = filename
                    phonemes_key = filename.replace('.csv', '')
                    phonemes_key = phonemes_key + '_' + str(1) + '.TextGrid'
                    phonemes_values = self.dictionary_trainset[phonemes_key]

                    self.DTW_X_train[phonemes_key] = deepcopy(feat.get_matrix())
                    self.DTW_Y_train[phonemes_key] = deepcopy(np.array(phonemes_values))

                    filename = temp

    def distance_cost_plot(self, distances):
        im = plt.imshow(distances, interpolation='nearest', cmap='Reds')
        plt.gca().invert_yaxis()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.colorbar()

    def dynamicTimeWarp(self, train, test):

        features_names = ['In', 'F1', 'F2', 'F3']
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
                        j -= 1
                    else:
                        i -= 1
                        j -= 1
                path.append([j, i])
            path.append([0, 0])

            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]

            length_x = len(path_x)
            length_y = len(path_y)

            assert length_x == length_y  # just to be sure :)

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

            similarity = 100 - (100 * statistics.mean(norm))

            with open(self.dtw_comparison_native_directory, 'a') as the_file:
                the_file.write("Similarity of {0}: {1:.2f}%\n".format(features_names[feat], similarity))

                # print "Similarity of {0}: {1:.2f}%".format(features_names[feat], similarity)

                # now we need to estimate the percentage of difference based on the distance
                # print "*** Plot distance cost ***"
                # self.distance_cost_plot(accumulated_cost)
                # plt.plot(path_x, path_y)

                # plt.show()

    def DTW_test(self):
        try:
            features_names = ['In', 'F1', 'F2', 'F3']
            already_used = []
            sentences = []
            with open(self.sentences_directory) as sentences_file:
                lines = sentences_file.readlines()
                for s in lines:
                    s = s.replace('\n', '')
                    sentences.append(s)

            for i in range(len(self.DTW_X_test)):

                # retrieve the sentence from the test set
                non_native = self.DTW_X_test[i]
                non_native_phonemes = self.DTW_Y_test[i]
                non_native_sentence = ""
                for key, val in self.dictionary_testset.items():
                    arr = np.array(val)
                    if np.array_equal(arr, non_native_phonemes):
                        non_native_sentence = key
                        break

                for sen in sentences:
                    # compare the non-native sentence with the classification set
                    if sen in non_native_sentence:

                        # retrieve the "same" sentence from the training set
                        for j in range(len(self.DTW_X_train)):
                            native = self.DTW_X_train[j]
                            native_phonemes = self.DTW_Y_train[j]
                            native_sentence = ""
                            for key, val in self.dictionary_trainset.items():

                                # if the sentence is the same
                                if sen in key:
                                    if np.array_equal(val, native_phonemes):

                                        # check if I already used this sentence
                                        if key in already_used:
                                            continue

                                        # save it and apply DTW
                                        native_sentence = key
                                        already_used.append(key)

                                        # debug
                                        print "Comparing: {} and {}".format(non_native_sentence, native_sentence)

                                        with open(self.dtw_comparison_directory, 'a') as the_file:
                                            the_file.write("Non native: {} - Native: {}\n".format(non_native_sentence,
                                                                                                  native_sentence))

                                            for feat in range(4):
                                                dist, path = fastdtw(non_native[:, feat], native[:, feat])

                                                path_x = [point[0] for point in path]
                                                path_y = [point[1] for point in path]

                                                length_x = len(path_x)
                                                length_y = len(path_y)

                                                assert length_x == length_y  # just to be sure :)

                                                distance = []
                                                for i in range(length_x):
                                                    distance.append(abs(path_x[i] - path_y[i]))

                                                # calculate a value for similarity
                                                min_distance = min(distance)
                                                max_distance = max(distance)

                                                norm = []
                                                for i in range(len(distance)):
                                                    z = float(distance[i] - min_distance) / float(
                                                        max_distance - min_distance)
                                                    norm.append(z)

                                                similarity = 100 - (100 * statistics.mean(norm))
                                                the_file.write(
                                                    "Similarity of {0}: {1:.2f}%\n".format(features_names[feat],
                                                                                           similarity))

                                                # self.distance_cost_plot(path)
                                                # plt.plot([int(i[0]) for i in path], [int(i[1]) for i in path])
                                                # plt.show()
        except:
            print "Error: ", sys.exc_info()
            raise

    def DTW_train(self):
        try:
            features_names = ['In', 'F1', 'F2', 'F3']

            non_native_sentence = []
            native_sentence = []

            for key, val in self.DTW_Y_train.items():
                non_native_sentence.append(key)
                native_sentence.append(key)

            already_used = []
            for j in range(len(non_native_sentence)):
                val = non_native_sentence[j]
                val = clean_filename_TextGrid(val)
                val = clean_filename_numbers(val)

                for k in range(len(native_sentence)):
                    if native_sentence[k] == non_native_sentence[j]:
                        continue

                    sec_val = native_sentence[k]
                    sec_val = clean_filename_TextGrid(sec_val)
                    sec_val = clean_filename_numbers(sec_val)

                    if sec_val != val:
                        continue

                    if native_sentence[k] in already_used:
                        continue

                    already_used.append(native_sentence[k])

                    non_native = self.DTW_X_train[non_native_sentence[j]]
                    native = self.DTW_X_train[native_sentence[k]]

                    # DTW operation
                    print "Comparing: {} and {}".format(non_native_sentence[j], native_sentence[k])

                    # not DTW between the same person
                    if np.array_equal(non_native, native):
                        continue

                    with open(self.dtw_comparison_native_directory, 'a') as the_file:
                        the_file.write(
                            "Non native: {} - Native: {}\n".format(non_native_sentence[j], native_sentence[k]))

                        for feat in range(4):
                            dist, path = fastdtw(non_native[:, feat], native[:, feat])

                            path_x = [point[0] for point in path]
                            path_y = [point[1] for point in path]

                            length_x = len(path_x)
                            length_y = len(path_y)

                            assert length_x == length_y  # just to be sure :)

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

                            similarity = 100 - (100 * statistics.mean(norm))
                            the_file.write("Similarity of {0}: {1:.2f}%\n".format(features_names[feat], similarity))

                            self.distance_cost_plot(path)
                            plt.plot(path_x, path_y)
                    plt.show()
                    x = 0

        except:
            print "Error: ", sys.exc_info()
            raise

    # endregion

    # region Model and trainer for phonemes prediction
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
            print "*** Creating Linear Chain CRF ***"
            mycrf = ChainCrfLinear(30,
                                   11)  # 30 - number of dimensions (phonemes + -1s), 11 (number of labels 1 to 10 + 1)

            # Alternatively, we could have used one of these, for example:
            # mytrainer = train.OnlinegradientNocost(mycrf, 0.95, 0.01)
            mytrainer = train.Bolddriver(mycrf, 0.01)
            # mytrainer = train.GradientdescentMomentum(mycrf, 0.95, 0.01)

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

            print "\n*** Applying WER ***"
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

    # endregion

    def run(self, train_model_on=True, dtw_on=True):
        # modeling
        print "*** Loading dictionaries ***"
        self.load_train_phonemes_dictionary()
        self.load_test_phonemes_dictionary()

        if train_model_on:
            print "*** Phonemes ***"
            self.labels_mapping()
            self.load_PHONEMES_set()
            self.load_PHONEMES_set(True)
            self.train_model()

        if dtw_on:
            print "\n*** DTW ***"
            self.load_DTW_set()
            self.load_DTW_set(True)

            # self.DTW_test()
            self.DTW_train()

        print "\n*** END ***"


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


class GMM_prototype:
    #region Global variables
    output_files_directory = 'output-data/formants_results/male/'
    #endregion

    # Classify Total Pronunciation (Call from CRF_DTW)
    def train_CRF(self):
        crf = CRF_DTW()
        # modeling
        print "*** Loading dictionaries for CRF ***"
        crf.load_train_phonemes_dictionary()
        crf.load_test_phonemes_dictionary()

        print "*** Phonemes for CRF ***"
        crf.labels_mapping()
        crf.load_PHONEMES_set()
        crf.load_PHONEMES_set(True)
        crf.train_model()

    # Train model with GMM
    def create_structure(self):
        all_data = []
        for root, dirs, files in os.walk(self.output_files_directory):
            for file in files:
                filename = os.path.join(root, file)

                if ".DS_Store" in filename or "_norm" not in filename:
                    continue

                with open(filename, 'r') as tabbed_file:
                    reader = csv.reader(tabbed_file, delimiter="\t")
                    all_lines = list(reader)

                    data = GMM_structure(file)

                    not_included = 0
                    for l in all_lines:
                        if not_included <= 2:
                            not_included += 1
                            continue

                        data.set_object(0, l[0])
                        data.set_object(1, l[1])
                        data.set_object(2, l[2])
                        try:
                            if l[3] == '':
                                f1_val = 0.0
                            else:
                                f1_val = float(l[3])

                            if l[4] == '':
                                f2_val = 0.0
                            else:
                                f2_val = float(l[4])

                            data.set_object(3, f1_val)
                            data.set_object(4, f2_val)
                        except:
                            print "Error: ", sys.exc_info()

                    all_data.append(deepcopy(data))
        return all_data

    def make_ellipses(self, gmm, ax):
        for n, color in enumerate('rgb'):
            v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[0], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v *= 9
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[0], 180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    def train_GMM(self):
        data = self.create_structure()

        all_vowels = []
        all_norm_f1 = []
        all_norm_f2 = []

        for str in data:
            temp_vowels = np.array(str.get_object(0))
            temp_norm_f1 = np.array(str.get_object(3))
            temp_norm_f2 = np.array(str.get_object(4))

            all_vowels.extend(temp_vowels)
            all_norm_f1.extend(temp_norm_f1)
            all_norm_f2.extend(temp_norm_f2)

        try:
            res_f1 = np.vstack(all_norm_f1)
            res_f2 = np.vstack(all_norm_f2)

            X_train = []
            for f, b in zip(all_norm_f1, all_norm_f2):
                X_train.append([f,b])

            X_train = np.array(X_train)
            Y_train = np.vstack(all_vowels)
            X_test = X_train[: int(0.25 * len(X_train))]
            Y_test = Y_train[: int(0.25 * len(Y_train))]

            labels = np.unique(Y_train)
            int_labels = np.arange(len(labels))

            map_int_label = dict(zip(labels, int_labels))

            Y_train_int = []
            for y in Y_train:
                val = map_int_label[y[0]]
                Y_train_int.append(val)

            Y_test_int = []
            for y in Y_test:
                val = map_int_label[y[0]]
                Y_test_int.append(val)

            n_classes = len(np.unique(Y_train))
            gmm_classifier = mixture.GMM(n_components=n_classes, covariance_type='full')

            # TODO: Check if i need to treat the means manually
            #gmm_classifier.means_ = np.array([X_train[Y_train == i].mean(axis=0) for i in xrange(n_classes)])
            gmm_classifier.fit(X_train)
            gmm_logprob, gmm_resp = gmm_classifier.score_samples(X_test)

            # dpgmm_classifier = mixture.DPGMM(n_components=n_classes, covariance_type='full')
            # dpgmm_classifier.fit(X_train)
            # dpgmm_logprob, dpgmm_resp = dpgmm_classifier.score_samples(X_test)

            #------------------------------------------------------------
            # Learn the best-fit GMM models
            #  Here we'll use GMM in the standard way: the fit() method
            #  uses an Expectation-Maximization approach to find the best
            #  mixture of Gaussians for the data

            # fit models with 1-10 components
            # N = np.arange(1, n_classes)
            # models = [None for i in range(len(N))]
            #
            # for i in range(len(N)):
            #     models[i] = mixture.GMM(N[i]).fit(X_train)
            #
            # # compute the AIC and the BIC
            # AIC = [m.aic(X_train) for m in models]
            # BIC = [m.bic(X_train) for m in models]
            #
            # #------------------------------------------------------------
            # # Plot the results
            # #  We'll use three panels:
            # #   1) data + best-fit mixture
            # #   2) AIC and BIC vs number of components
            # #   3) probability that a point came from each component
            #
            # # plot 1: data + best-fit mixture
            # M_best = models[np.argmin(AIC)]
            #
            # # plot 2: AIC and BIC
            # plt.plot(N, AIC, '-k', label='AIC')
            # plt.plot(N, BIC, '--k', label='BIC')
            # plt.xlabel('n. components')
            # plt.ylabel('information criterion')
            # plt.legend(loc=2)
            #
            # plt.show()

            x = 0
        except:
            print sys.exc_info()
            raise

    # Get Score!
    def calculate_score(self):
        x = 0

    # test
    def run(self):
        #self.train_CRF()
        self.train_GMM()
        self.calculate_score()





















