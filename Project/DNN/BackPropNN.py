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
import sys
import os

from fann2 import libfann
from MFCC import melScaling

class TestingFANN:
    train_audio_files_directory = "train-audio-data/"
    test_audio_files_directory = "test-audio-data/"
    trainset_filename = "dataset_fann.txt"
    testset_filename = "testset_fann.txt"
    ann_filename = "set.net"
    dataset_file = os.path.join(train_audio_files_directory, trainset_filename)
    testset_file = os.path.join(test_audio_files_directory, testset_filename)
    ann_file = os.path.join(train_audio_files_directory, ann_filename)

    num_of_ceps = 25  # number of features extracted from the audio file - NN inputs must be the same number

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

    def print_callback(epochs, error):
        print "Epochs     %8d. Current MSE-Error: %.10f\n" % (epochs, error)
        return 0

    def train_ann(self, train_audio_signals, labels_reference):
        #set = self.extract_features(train_audio_signals)
        #self.crate_dataset_file(set, labels_reference)

        # initialize network parameters
        connection_rate = 1
        learning_rate = 0.7
        num_neurons_hidden = 32
        desired_error = 0.000001
        max_iterations = 300
        iterations_between_reports = 1

        # create training data, and ann object
        print "Creating network."
        train_data = libfann.training_data()
        train_data.read_train_from_file(self.dataset_file)

        ann = libfann.neural_net()
        ann.create_sparse_array(connection_rate, (len(train_data.get_input()[0]), num_neurons_hidden, len(train_data.get_output()[0])))
        ann.set_learning_rate(learning_rate)


        # start training the network
        print "Training network"
        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        ann.set_activation_function_output(libfann.SIGMOID_STEPWISE)
        ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)

        ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error)

        # save network to disk
        print "Saving network"
        ann.save(self.ann_file)

    def test_ann(self, test_set):

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
