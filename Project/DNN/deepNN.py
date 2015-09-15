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
import pylab as pl
import matplotlib.pyplot as plt

# For function test(...)
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer, BackpropTrainer
from sklearn.metrics import mean_squared_error
from pybrain.tools.validation import testOnSequenceData, ModuleValidator
from pybrain.structure.modules import SoftmaxLayer, TanhLayer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

from features import mfcc
from features import logfbank

from utility import Utility

# For function test2(...)
import sys
import sklearn.decomposition as deco
from numpy.fft import fft, fftshift
from MFCC import melScaling


class Test:
    num_of_ceps = 25  # number of features extracted from the audio file - NN inputs must be the same number

    def generate_dataset(self, audio_signals, isTestset):
        utility = Utility()
        set = None

        if set is None:
            set = []
            for key, value in audio_signals.items():
                # extract features
                signal = value[0]
                stream_rate = value[1]

                mfcc_feat = mfcc(signal, stream_rate, numcep=self.num_of_ceps)  # [TODO] investigate about this feature
                fbank_feat = logfbank(signal, stream_rate)  # [TODO] investigate about this feature

                # create the trainset
                num_ceps = len(mfcc_feat)
                set.append(np.mean(mfcc_feat[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))  # need to figure it out what is this

            utility.save_set(set, isTestset=isTestset)  # save the set
        return set

    def prepare_set_for_NN(self, set, dimensions, n_classes):
        all_train_data = ClassificationDataSet(dimensions, 1, nb_classes=n_classes)
        i = 0
        for array in set:
            input = array
            all_train_data.addSample(input, [i])
            i += 1
        return all_train_data

    def encode_classes(self, temp_set, dimensions, n_classes):
        encoded_data = ClassificationDataSet(dimensions, 1, nb_classes=n_classes)
        for n in xrange(0, temp_set.getLength()):
            encoded_data.addSample(temp_set.getSample(n)[0], temp_set.getSample(n)[1])

        encoded_data._convertToOneOfMany()
        return encoded_data

    def test(self, train_audio_signals, test_audio_signals, num_of_classes):
        utility = Utility()
        trainset = utility.load_set(True)  # it contains the mfcc features of all TRAIN audio files
        testset = utility.load_set(False)  # it contains the mfcc features of all TEST audio files

        if trainset is None:
            trainset = self.generate_dataset(train_audio_signals, False)

        if testset is None:
            testset = self.generate_dataset(test_audio_signals, True)

        # TRAIN set
        trndata_temp = self.prepare_set_for_NN(trainset, self.num_of_ceps, num_of_classes)
        trndata = self.encode_classes(trndata_temp, self.num_of_ceps, num_of_classes)

        # TEST SET
        tstdata_temp = self.prepare_set_for_NN(testset, self.num_of_ceps, num_of_classes)
        tstdata = self.encode_classes(tstdata_temp, self.num_of_ceps, num_of_classes)

        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim

        fnn = buildNetwork(trndata.indim, 10, trndata.outdim, hiddenclass=SigmoidLayer, bias=True,
                           outclass=SoftmaxLayer)
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.01, verbose=True, weightdecay=0.3)
        # trainer = RPropMinusTrainer(fnn, dataset=trndata, verbose=True)

        # carry out the training
        test_err = []
        train_err = []
        for i in xrange(50):
            trainer.trainEpochs(1)

            resTrain = 100 - percentError(trainer.testOnClassData(), trndata['class'])
            train_err.append(resTrain)

            resTest = 100 - percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            test_err.append(resTest)

            test_err[i] = ModuleValidator.MSE(fnn, trndata)
            print "MSE: %5.2f%% " % test_err[i]

        print "epoch: %4d " % trainer.totalepochs, "\ttrain acc: %5.2f%% " % resTrain, "\ttest acc: %5.2f%%" % resTest

        # Plot training and test error as a function of the training size
        pl.figure()
        pl.plot(test_err)
        pl.show()

        print "End!"

    def test2(self, train_audio_signals, test_audio_signals, num_of_classes):

        train_classified_features = self.extract_features(train_audio_signals)
        test_features = self.extract_features(test_audio_signals, isTestSet=True)

        # split data
        features = [x[0] for x in train_classified_features]
        labels = [x[1] for x in train_classified_features]

        try:
            # TODO Classification Step - ANN Classifier - Minimum Error Rate
            trainingSet = SupervisedDataSet(25, 1)
            for ri in range(len(features)):
                tuple = train_classified_features[ri]
                trainingSet.addSample(tuple[0], tuple[1])

            net = buildNetwork(self.num_of_ceps, 100, len(labels), bias=True)
            trainer = BackpropTrainer(net, trainingSet, learningrate=0.001, momentum=0.99)

            # carry out the training
            test_err = []
            train_err = []
            for i in xrange(50):
                trainer.trainEpochs(1)

                resTrain = 100 - percentError(trainer.testOnClassData(), trainingSet['class'])
                train_err.append(resTrain)

                resTest = 100 - percentError(trainer.testOnClassData(dataset=test_features), test_features['class'])
                test_err.append(resTest)

                test_err[i] = ModuleValidator.MSE(net, trainingSet)
                print "MSE: %5.2f%% " % test_err[i]

            print "1) ", net.active(test_features[0])

        except:
            print "Error: ", sys.exc_info()
            raise

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

    def DTWDistance(self, s1, s2, w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW = {}

        if w:
            w = max(w, abs(len(s1) - len(s2)))

            for i in range(-1, len(s1)):
                for j in range(-1, len(s2)):
                    DTW[(i, j)] = float('inf')

        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            if w:
                for j in range(max(0, i - w), min(len(s2), i + w)):
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
            else:
                for j in range(len(s2)):
                    dist = (s1[i] - s2[j]) ** 2
                    DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

        return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

    def LB_Keogh(self, s1, s2, r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum = 0
        for ind, i in enumerate(s1):

            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2

        return np.sqrt(LB_sum)
