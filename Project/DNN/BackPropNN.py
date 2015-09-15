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
    trainset_filename = "dataset_fann.txt"
    ann_filename = "set.net"
    dataset_file = os.path.join(train_audio_files_directory, trainset_filename)
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

    def crate_dataset_file(self, set, labels_ref, isTestSet=False):

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
                conversion = ['{:.5f}'.format(x) for x in framefeatures]
                content += ' '.join(conversion)
                content += '\n'

                # third/fifth/seventh/etc. line:
                # output value:
                content += str(labels_ref[label])
                content += '\n'

            file.write(content)
            file.close()
        except:
            print("Error: ", sys.exc_info())
            raise

    def train_ann(self, train_audio_signals, labels_reference):
        set = self.extract_features(train_audio_signals)
        self.crate_dataset_file(set, labels_reference)

        connection_rate = 1
        learning_rate = 0.7
        num_input = 25
        num_hidden = 10
        num_output = 1

        desired_error = 0.0001
        max_iterations = 100000
        iterations_between_reports = 10

        ann = libfann.neural_net()
        ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
        ann.set_learning_rate(learning_rate)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

        ann.train_on_file(self.dataset_file, max_iterations, iterations_between_reports, desired_error)

        ann.save(self.ann_file)