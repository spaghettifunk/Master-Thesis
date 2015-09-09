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
import numpy as np
import librosa

class DeepNN:
    # save locally the signals
    audio_data = {}

    # def __init__(self, audio_signals):
    #     self.audio_data = audio_signals
    #
    #     for key, value in self.audio_data.items():
    #         print(numpy.asarray(value).shape)

    # ref: http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_prime = self.sigmoid_prime
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_prime = self.tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            if k % 10000 == 0: print
            'epochs:', k

            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1.0 - x ** 2


class Test:
    def test(self, audio_signals):
        nn = DeepNN([2, 2, 1])

        #X = np.array([[0, 0],
        #              [0, 1],
        #              [1, 0],
        #              [1, 1]])

        #y = np.array([0, 1, 1, 0])

        signal, stream_rate = next(iter(audio_signals.values()))  # train set
        filename = next(iter(audio_signals.keys()))

        mfcc = librosa.feature.mfcc(y=signal, sr=stream_rate, n_mfcc=13)
        np.save('saved-trainset/cached_train_set', mfcc) # cache results so that ML becomes fast

        X_train = []
        #ceps = np.load("saved-trainset/cached_train_set")
        num_ceps = len(mfcc)
        X_train.append(np.mean(mfcc[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
        Vx = np.array(X_train) # use Vx as input values vector for neural net, k-means, etc

        print(Vx.shape)

        y = np.ndarray(shape=(190,1))   # label set

        try:
            nn.fit(X_train, y)
        except:
            print("Unexpected error:", sys.exc_info())
            raise

        for e in X_train:
            print(e, nn.predict(e))
