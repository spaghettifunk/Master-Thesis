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
import os
import csv

import numpy as np
from fann2 import libfann

from libraries.MFCC import melScaling
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
# from keras.datasets import cifar10


class Features:
    intensity = []
    f1 = []
    f2 = []
    f3 = []

    def __init__(self):
        self.intensity = []
        self.f1 = []
        self.f2 = []
        self.f3 = []

    def getObject(self, n):
        if n == 0:
            return self.intensity
        if n == 1:
            return self.f1
        if n == 2:
            return self.f2
        if n == 3:
            return self.f3

    def setObject(self, n, val):
        if n == 0:
            self.intensity.append(val)
        if n == 1:
            self.f1.append(val)
        if n == 2:
            self.f2.append(val)
        if n == 3:
            self.f3.append(val)

class TestingFANN:
    train_audio_files_directory = "train-audio-data/smoothed_CSV_files/male/"
    test_audio_files_directory = "test-audio-data/"
    trainset_filename = "dataset_fann_features.txt"
    testset_filename = "testset_fann.txt"
    ann_filename = "set.net"
    dataset_file = os.path.join(train_audio_files_directory, trainset_filename)
    testset_file = os.path.join(test_audio_files_directory, testset_filename)
    ann_file = os.path.join(train_audio_files_directory, ann_filename)

    num_of_ceps = 25  # number of features extracted from the audio file - NN inputs must be the same number

    filename_title = ["a_piece_of_cake","blow_a_fuse","catch_some_zs","down_to_the_wire","eager_beaver","fair_and_square","get_cold_feet","mellow_out","pulling_your_legs","thinking_out_loud"]

    def extract_features(self, train_audio_signals, isTestSet=False):
        # TODO Filtering Step - (Hamming FIR)
        framelen = 1024

        features = []
        for key, value in train_audio_signals.items():
            # extract features
            signal = value[0]
            stream_rate = value[1]
            chunks = value[2]

            if isTestSet == False:
                classification_label = value[3]

            try:
                # MFCC library
                mfccMaker = melScaling(int(stream_rate), framelen / 2, 40)
                mfccMaker.update()

                # TODO Feature Extraction Step - FFT - Vector of initial features
                framespectrum = np.fft.fft(chunks)
                magspec = abs(framespectrum[:framelen / 2])

                # do the frequency warping and MFCC computation
                melSpectrum = mfccMaker.warpSpectrum(magspec)
                melCepstrum = mfccMaker.getMFCCs(melSpectrum, cn=True)
                melCepstrum = melCepstrum[1:]  # exclude zeroth coefficient
                melCepstrum = melCepstrum[:self.num_of_ceps]  # limit to lower MFCCs

                framefeatures = melCepstrum

                # Features Library
                # mfcc_feat = mfcc(signal, stream_rate)
                # framefeatures = mfcc_feat
                # fbank_feat = logfbank(signal, stream_rate)
                # print mfcc_feat.shape

                if isTestSet == False:
                    elem = (framefeatures, classification_label)
                else:
                    elem = framefeatures
                features.append(elem)

                # TODO Feature Selection Step - PCA - Principal Features -> is it necessary ?

            except:
                print "Error: ", sys.exc_info()[0]
                raise
        return features

    def crate_dataset_file(self, set, labels_ref):
        try:
            file = open(self.dataset_file, 'w')  # 'w' -> create a new file, 'a' append to file
            # first line:
            # 1) number of samples -> number of files (40)
            # 2) how many input values per sample -> number of features (25)
            # 3) the output value -> either a value or the filename (1)
            content = "" +  str(len(set)) + " " + "25 " + "1\n"

            # second/forth/sixth/etc. line:
            # - values of the sample
            for framefeatures, label in set:
                conversion = ["{:.5f}".format(x) for x in framefeatures]
                content += " ".join(conversion)
                content += "\n"

                # third/fifth/seventh/etc. line:
                # output value:
                content += str(labels_ref[label])
                content += "\n"

            file.write(content)
            file.close()
        except:
            print("Error: ", sys.exc_info())
            raise

    def create_testset_file(self, set):
        try:
            file = open(self.testset_file, 'w')  # 'w' -> create a new file, 'a' append to file
            content = "" +  str(len(set)) + " " + "25 " + "1\n"
            i = 0
            for framefeatures in set:
                conversion = ['{:.5f}'.format(x) for x in framefeatures]
                content += " ".join(conversion)

                if i != len(set) - 1:
                    content += "\n"
                i += 1

            file.write(content)
            file.close()
        except:
            print("Error: ", sys.exc_info())
            raise

    def create_training_set_csv(self, n_samples, n_inputs):
        try:
            file = open(self.dataset_file, 'w')  # 'w' -> create a new file, 'a' append to file
            # first line:
            # 1) number of samples -> number of files (40)
            # 2) how many input values per sample -> number of features (25)
            # 3) the output value -> either a value or the filename (1)
            content = "" +  str(n_samples) + " " + str(n_inputs) + " 1\n"

            # second/forth/sixth/etc. line:
            # - values of the sample
            for root, dirs, files in os.walk(self.train_audio_files_directory):
                for csv_file in files:
                    file_name = os.path.join(root, csv_file)

                    if "dataset_fann_features" in file_name or "set.net" in file_name:
                        continue

                    with open(file_name, 'rU') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')

                        file_ref = self.clean_filename(csv_file)
                        assert file_ref != ""

                        feat = Features()
                        i = 0
                        for row in reader:
                            if i == 0:
                                i += 1
                                continue
                            elif i == n_samples:   # need to have the same length! I loose information here
                                break

                            # row[0] = time column
                            feat.setObject(0, float(row[1]))
                            feat.setObject(1, float(row[2]))
                            feat.setObject(2, float(row[3]))
                            feat.setObject(3, float(row[4]))

                            i += 1

                        content += self.get_values(feat, 0, file_ref)
                        content += self.get_values(feat, 1, file_ref)
                        content += self.get_values(feat, 2, file_ref)
                        content += self.get_values(feat, 3, file_ref)

                file.write(content)
                file.close()
        except:
            print("Error: ", sys.exc_info())
            raise

    def clean_filename(self, filename):
        cleaned_name = filename.replace(".csv", "")
        if "Jeremy_" in cleaned_name:
            return cleaned_name.replace("Jeremy_", "")
        elif "Lenny_" in cleaned_name:
            return cleaned_name.replace("Lenny_", "")
        elif "Philip_" in cleaned_name:
            return cleaned_name.replace("Philip_", "")
        elif "Marty_" in cleaned_name:
            return cleaned_name.replace("Marty_", "")
        elif "Joyce_" in cleaned_name:
            return cleaned_name.replace("Joyce_", "")
        elif "Niki_" in cleaned_name:
            return cleaned_name.replace("Niki_", "")
        else:
            return ""

    def get_values(self, feat_obj, feat_ref, file_ref):
        string_to_write = ""

        conversion = ["{:.5f}".format(x) for x in feat_obj.getObject(feat_ref)]
        string_to_write += " ".join(conversion)
        string_to_write += "\n"

        # third/fifth/seventh/etc. line:
        # output value:

        # create a label based on the filename index and the feature index
        int_label = str(self.filename_title.index(file_ref)) + str(feat_ref)
        string_to_write += int_label
        string_to_write += "\n"
        return string_to_write


    def print_callback(epochs, error):
        print "Epochs     %8d. Current MSE-Error: %.10f\n" % (epochs, error)
        return 0

    def train_ann(self): #, train_audio_signals, labels_reference):
        #set = self.extract_features(train_audio_signals)
        #self.crate_dataset_file(set, labels_reference)

        n_samples = 40 # number of files times number of columns
        n_inputs = 90 # number of elements per column
        n_outputs = 1

        self.create_training_set_csv(n_samples, n_inputs)

        # initialize network parameters
        connection_rate = 1
        learning_rate = 0.07
        num_neurons_hidden = 32
        desired_error = 0.000001
        max_iterations = 1000
        iterations_between_reports = 50

        # create training data, and ann object
        print "Creating network."
        train_data = libfann.training_data()
        train_data.read_train_from_file(self.dataset_file)

        try:
            ann = libfann.neural_net()

            ann.create_sparse_array(connection_rate, (n_inputs, 20, 20, 20, 20, 20, 20, 20, n_outputs))
            ann.set_learning_rate(learning_rate)

            # start training the network
            print "Training network"
            ann.set_activation_function_hidden(libfann.SIGMOID_STEPWISE)
            ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)
            ann.set_training_algorithm(libfann.TRAIN_BATCH)

            ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error)

            # save network to disk
            print "Saving network"
            ann.save(self.ann_file)
        except:
            print("Error: ", sys.exc_info())

    def test_ann(self): #, test_set):

        try:
            #set = self.extract_features(test_set, isTestSet=True)
            #self.create_testset_file(set)

            # load ANN
            ann = libfann.neural_net()
            ann.create_from_file(self.ann_file)

            # test outcome
            print "Testing network"
            test_data = libfann.training_data()
            test_data.read_train_from_file(self.testset_file)

            ann.reset_MSE()
            ann.test_data(test_data)
            print "MSE error on test data: %f" % ann.get_MSE()

            print "Testing network again"
            ann.reset_MSE()
            input=test_data.get_input()
            output=test_data.get_output()

            for i in range(len(input)):
                ann.test(input[i], output[i])

            print "MSE error on test data: %f" % ann.get_MSE()
        except:
            print "Error: ", sys.exc_info()
            raise
