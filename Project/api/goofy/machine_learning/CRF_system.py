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

import sys
import yaml

from utilities.logger import Logger
from libraries.utility import *

# monte
from libraries.monte.models.crf import ChainCrfLinear
from libraries.monte import train

class CRF:
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

    # region Load dictionary from file
    def load_train_phonemes_dictionary(self):
        Logger.LogInfo("*** Load Train phonemes dictionary ***")
        with open(self.train_dictionary_phonemes_directory) as data_file:
            self.dictionary_trainset = yaml.load(data_file)
    # endregion

    # region Model and trainer for phonemes prediction
    def load_PHONEMES_set(self, isTest=False):
        Logger.LogInfo("*** Load Phonemes set ***")
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
        Logger.LogInfo("*** Mapping labels ***")
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
            Logger.LogInfo("*** Creating Linear Chain CRF ***")

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
            Logger.LogInfo("*** Training model ***")
            for i in range(20):
                mytrainer.step((inputs, outputs), 0.001)
                print "Cost: ", mycrf.cost((inputs, outputs), 0.001)

            # Apply to some stupid test data:
            testinputs = np.array(self.PHONEMES_X_test)
            predictions = mycrf.viterbi(testinputs)
            Logger.LogInfo("Labels predicted: ", predictions)  # each element corrisponds to a sentences in the sentences.txt file - Index starts from 1!!

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

            Logger.LogInfo("\n*** Applying WER ***")
            counter = 0
            for val in self.PHONEMES_X_test:
                test_phonemes = val  # hyphothesis
                wer_result, numCor, numSub, numIns, numDel = self.wer(reference[counter], test_phonemes)
                Logger.LogInfo("WER: {}, OK: {}, SUB: {}, INS: {}, DEL: {}".format(wer_result, numCor, numSub, numIns, numDel))

        except:
            message = "Error in " + self.train_model.__name__ + " - Reason: " + sys.exc_info()
            Logger.LogError(message)
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

    def run(self):
        Logger.LogInfo("*** Loading dictionaries ***")
        self.load_train_phonemes_dictionary()

        self.labels_mapping()
        self.load_PHONEMES_set()
        self.load_PHONEMES_set(True)
        self.train_model()

        Logger.LogInfo("*** END CRF Training phase ***")