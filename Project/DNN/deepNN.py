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
from pylab import plot, hold, show

# For function test(...)
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer, BackpropTrainer
from sklearn.metrics import mean_squared_error
from pybrain.tools.validation    import testOnSequenceData, ModuleValidator
from pybrain.structure.modules import SoftmaxLayer

# For function test2(...)
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

from features import mfcc
from features import logfbank

from utility import Utility

class Test:
    num_of_ceps = 13    # number of features extracted from the audio file - NN inputs must be the same number

    def generate_dataset(self, audio_signals, isTestset):
        utility = Utility()
        set = None

        if set is None:
            set = []
            for key, value in audio_signals.items():
                # extract features
                signal = value[0]
                stream_rate = value[1]

                mfcc_feat = mfcc(signal, stream_rate, numcep=self.num_of_ceps)           # [TODO] investigate about this feature
                fbank_feat = logfbank(signal, stream_rate)                 # [TODO] investigate about this feature

                # create the trainset
                num_ceps = len(mfcc_feat)
                set.append(np.mean(mfcc_feat[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))  # need to figure it out what is this

            utility.save_set(set, isTestset=isTestset)  # save the set
        return set

    def prepare_set_for_NN(self, set, dimensions, n_classes):
        all_train_data = ClassificationDataSet(dimensions, 1, nb_classes=n_classes)
        for n in range(400):
            i = 0
            for array in set:
                input = array
                all_train_data.addSample(input, [i])
                i += 1
        return all_train_data

    def encode_classes(self, temp_set, dimensions, n_classes):
        encoded_data = ClassificationDataSet(dimensions, 1, nb_classes=n_classes)
        for n in range(0, temp_set.getLength()):
            encoded_data.addSample( temp_set.getSample(n)[0], temp_set.getSample(n)[1])
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

        print("Number of training patterns: ", len(trndata))
        print("Input and output dimensions: ", trndata.indim, trndata.outdim)

        fnn = buildNetwork(trndata.indim, 2, trndata.outdim)

        #trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
        trainer = RPropMinusTrainer(fnn, dataset=trndata, verbose=True)

        # carry out the training
        test_err = []
        train_err = []
        for i in range(20):
            trainer.trainEpochs( 1 )
            train_err.append(percentError(trainer.testOnClassData(), trndata['class']))
            test_err.append(percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class']))

            test_err[i] = ModuleValidator.MSE(fnn, trndata)

        # just for reference, plot the first 5 timeseries
        plot(test_err)
        hold(True)
        show()

        print("End!")

    def test2(self, train_audio_signals, test_audio_signals):
        utility = Utility()
        trainset = utility.load_set(True)  # it contains the mfcc features of all TRAIN audio files
        testset = utility.load_set(False)  # it contains the mfcc features of all TEST audio files

        if trainset is None:
            trainset = self.generate_dataset(train_audio_signals, False)

        if testset is None:
            testset = self.generate_dataset(test_audio_signals, True)

        net = FeedForwardNetwork()
        inputLayer = LinearLayer(13)
        hiddenLayer = SigmoidLayer(3)
        outputLayer = LinearLayer(4)

        net.addInputModule(inputLayer)
        net.addModule(hiddenLayer)
        net.addOutputModule(outputLayer)

        in_to_hidden = FullConnection(inputLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outputLayer)

        net.addConnection(in_to_hidden)
        net.addConnection(hidden_to_out)

        # This call does some internal initialization
        # which is necessary before the net can finally
        # be used: for example, the modules are sorted topologically.
        net.sortModules()

        print(net.params)
        print("End!")

