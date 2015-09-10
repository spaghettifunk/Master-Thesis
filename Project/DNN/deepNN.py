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

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from features import mfcc
from features import logfbank

from utility import Utility


class Test:

    def generate_dataset(self, audio_signals, isTestset):
        utility = Utility()
        set = None

        if set is None:
            set = []
            for key, value in audio_signals.items():
                # extract features
                signal = value[0]
                stream_rate = value[1]

                mfcc_feat = mfcc(signal, stream_rate)
                fbank_feat = logfbank(signal, stream_rate)

                # create the trainset
                num_ceps = len(mfcc_feat)
                set.append(np.mean(mfcc_feat[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))  # need to figure it out what is this

            utility.save_set(set, isTestset=isTestset)  # save the set
        return set

    def test(self, train_audio_signals, test_audio_signals):
        utility = Utility()
        trainset = utility.load_set(True)  # it contains the mfcc features of all TRAIN audio files
        testset = utility.load_set(False)  # it contains the mfcc features of all TEST audio files

        if trainset is None:
            trainset = self.generate_dataset(train_audio_signals, False)

        if testset is None:
            testset = self.generate_dataset(test_audio_signals, True)

        # TRAIN set
        all_train_data = ClassificationDataSet(13, 1, nb_classes=4)
        for n in range(400):
            i = 0
            for array in trainset:
                input = array
                all_train_data.addSample(input, [i])
                i += 1

        trndata_temp = all_train_data  # TRAIN set

        # TEST SET
        all_test_data = ClassificationDataSet(13, 1, nb_classes=4)
        for n in range(400):
            i = 0
            for array in testset:
                input = array
                all_test_data.addSample(input, [i])
                i += 1

        tstdata_temp = all_test_data  # TEST set

        # test set
        tstdata = ClassificationDataSet(13, 1, nb_classes=4)
        for n in range(0, tstdata_temp.getLength()):
            tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

        trndata = ClassificationDataSet(13, 1, nb_classes=4)
        for n in range(0, trndata_temp.getLength()):
            trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

        print("Number of training patterns: ", len(trndata))
        print("Input and output dimensions: ", trndata.indim, trndata.outdim)
        print("First sample (input, target, class):")
        print(trndata['input'][0], trndata['target'][0]) #, trndata['class'][0])

        fnn = buildNetwork(trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

        for i in range(20):
            trainer.trainEpochs(1)
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

        print("epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult)
        print("End!")


