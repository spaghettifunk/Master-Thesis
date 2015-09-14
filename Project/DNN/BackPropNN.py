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

from scipy.special import expit
import numpy as np
import math
import random
import itertools
import collections
import sys

from MFCC import melScaling

def sigmoid_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )

    # Calculate activation signal
    signal = expit( signal )

    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1-signal)
    else:
        # Return the activation signal
        return signal
#end activation function

def elliot_function( signal, derivative=False ):
    """ A fast approximation of sigmoid """
    s = 1 # steepness

    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal * s) / abs_signal + 0.5
#end activation function


def symmetric_elliot_function( signal, derivative=False ):
    """ A fast approximation of tanh """
    s = 1 # steepness

    abs_signal = (1 + np.abs(signal * s))
    if derivative:
        return s / abs_signal**2
    else:
        # Return the activation signal
        return (signal * s) / abs_signal
#end activation function

def ReLU_function( signal, derivative=False ):
    if derivative:
        # Prevent overflow.
        signal = np.clip( signal, -500, 500 )
        # Return the partial derivation of the activation function
        return expit( signal )
    else:
        # Return the activation signal
        output = np.copy( signal )
        output[ output < 0 ] = 0
        return output
#end activation function

def LReLU_function( signal, derivative=False ):
    """
    Leaky Rectified Linear Unit
    """
    if derivative:
        derivate = np.copy( signal )
        derivate[ derivate < 0 ] *= 0.01
        # Return the partial derivation of the activation function
        return derivate
    else:
        # Return the activation signal
        output = np.copy( signal )
        output[ output < 0 ] = 0
        return output
#end activation function

def tanh_function( signal, derivative=False ):
    # Calculate activation signal
    signal = np.tanh( signal )

    if derivative:
        # Return the partial derivation of the activation function
        return 1-np.power(signal,2)
    else:
        # Return the activation signal
        return signal
#end activation function

def linear_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return 1
    else:
        # Return the activation signal
        return signal
#end activation function


class NeuralNet:
    def __init__(self, n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions ):
        self.n_inputs = n_inputs                # Number of network input signals
        self.n_outputs = n_outputs              # Number of desired outputs from the network
        self.n_hiddens = n_hiddens              # Number of nodes in each hidden layer
        self.n_hidden_layers = n_hidden_layers  # Number of hidden layers in the network
        self.activation_functions = activation_functions

        assert len(activation_functions)==(n_hidden_layers+1), "Requires "+(n_hidden_layers+1)+" activation functions, got: "+len(activation_functions)+"."

        if n_hidden_layers == 0:
            # Count the necessary number of weights for the input->output connection.
            # input -> [] -> output
            self.n_weights = ((n_inputs+1)*n_outputs)
        else:
            # Count the necessary number of weights summed over all the layers.
            # input -> [n_hiddens -> n_hiddens] -> output
            self.n_weights = (n_inputs+1)*n_hiddens+\
                             (n_hiddens**2+n_hiddens)*(n_hidden_layers-1)+\
                             n_hiddens*n_outputs+n_outputs

        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights() )
    #end

    def generate_weights(self, low=-0.1, high=0.1):
        # Generate new random weights for all the connections in the network
        if not False:
            # Support NumPy
            return [random.uniform(low,high) for _ in range(self.n_weights)]
        else:
            return np.random.uniform(low, high, size=(1,self.n_weights)).tolist()[0]
    #end

    def unpack(self, weight_list ):
        # This method will create a list of weight matrices. Each list element
        # corresponds to the connection between two layers.
        if self.n_hidden_layers == 0:
            return [ np.array(weight_list).reshape(self.n_inputs+1,self.n_outputs) ]
        else:
            weight_layers = [ np.array(weight_list[:(self.n_inputs+1)*self.n_hiddens]).reshape(self.n_inputs+1,self.n_hiddens) ]
            weight_layers += [ np.array(weight_list[(self.n_inputs+1)*self.n_hiddens+(i*(self.n_hiddens**2+self.n_hiddens)):(self.n_inputs+1)*self.n_hiddens+((i+1)*(self.n_hiddens**2+self.n_hiddens))]).reshape(self.n_hiddens+1,self.n_hiddens) for i in range(self.n_hidden_layers-1) ]
            weight_layers += [ np.array(weight_list[(self.n_inputs+1)*self.n_hiddens+((self.n_hidden_layers-1)*(self.n_hiddens**2+self.n_hiddens)):]).reshape(self.n_hiddens+1,self.n_outputs) ]

        return weight_layers
    #end

    def set_weights(self, weight_list ):
        # This is a helper method for setting the network weights to a previously defined list.
        # This is useful for utilizing a previously optimized neural network weight set.
        self.weights = self.unpack( weight_list )
    #end

    def get_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]
    #end

    def backpropagation(self, trainingset, ERROR_LIMIT=1e-3, learning_rate=0.3, momentum_factor=0.9  ):
        def addBias(A):
            # Add 1 as bias.
            return np.hstack(( np.ones((A.shape[0],1)), A ))
        #end addBias

        assert trainingset[0].features.shape[0] == self.n_inputs, "ERROR: input size varies from the defined input setting"
        assert trainingset[0].targets.shape[0] == self.n_outputs, "ERROR: output size varies from the defined output setting"

        training_data = np.array( [instance.features for instance in trainingset ] )
        training_targets = np.array( [instance.targets for instance in trainingset ] )

        MSE      = ( ) # inf
        neterror = None
        momentum = collections.defaultdict( int )

        epoch = 0
        while MSE > ERROR_LIMIT:
            epoch += 1

            input_layers      = self.update( training_data, trace=True )
            out               = input_layers[-1]

            error             = training_targets - out
            delta             = error
            MSE               = np.mean( np.power(error,2) )


            loop  = itertools.izip(
                            range(len(self.weights)-1, -1, -1),
                            reversed(self.weights),
                            reversed(input_layers[:-1]), )


            for i, weight_layer, input_signals in loop:
                # Loop over the weight layers in reversed order to calculate the deltas

                # Calculate weight change
                dW = learning_rate * np.dot( addBias(input_signals).T, delta ) + momentum_factor * momentum[i]

                if i!= 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skipping the bias weight during calculation.
                    weight_delta = np.dot( delta, weight_layer[1:,:].T )

                    # Calculate the delta for the subsequent layer
                    delta = np.multiply(  weight_delta, self.activation_functions[i-1]( input_signals, derivative=True) )

                # Store the momentum
                momentum[i] = dW

                # Update the weights
                self.weights[ i ] += dW

            if epoch%1000==0:
                # Show the current training status
                print ("* current network error (MSE):", MSE)

        print ("* Converged to error bound (%.4g) with MSE = %.4g." % ( ERROR_LIMIT, MSE ))
        print ("* Trained for %d epochs." % epoch)
    # end backprop

    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we calculate the network output
        # from a set of input signals.

        output = input_values
        if trace: tracelist = [ output ]

        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            output = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            output = self.activation_functions[i]( output )
            if trace: tracelist.append( output )

        if trace: return tracelist

        return output
    #end

    def save_to_file(self, filename = "network.pkl" ):
        """
        This save method pickles the parameters of the current network into a
        binary file for persistant storage.
        """
        with open( filename , 'wb') as file:
            store_dict = {
                "n_inputs"             : self.n_inputs,
                "n_outputs"            : self.n_outputs,
                "n_hiddens"            : self.n_hiddens,
                "n_hidden_layers"      : self.n_hidden_layers,
                "activation_functions" : self.activation_functions,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights

            }
            #cPickle.dump( store_dict, file, 2 )
    #end

    @staticmethod
    def load_from_file( filename = "network.pkl" ):
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( 0, 0, 0, 0, [0] )

        with open( filename , 'rb') as file:
            store_dict = []#cPickle.load(file)

            network.n_inputs             = store_dict["n_inputs"]
            network.n_outputs            = store_dict["n_outputs"]
            network.n_hiddens            = store_dict["n_hiddens"]
            network.n_hidden_layers      = store_dict["n_hidden_layers"]
            network.n_weights            = store_dict["n_weights"]
            network.weights              = store_dict["weights"]
            network.activation_functions = store_dict["activation_functions"]

        return network
    #end
#end class

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in out training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets = np.array(target)
#endclass Instance

class WillItWork:
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
                    elem = Instance(framefeatures, classification_label)
                else:
                    elem = framefeatures
                features.append(elem)

                # TODO Feature Selection Step - PCA - Principal Features -> is it necessary ?

            except:
                print("Error: ", sys.exc_info()[0])
                raise
        return features

    def test(self, train_audio_signals):
        # two training sets
        training_one =  self.extract_features(train_audio_signals)
        # training_two =  [ Instance( [0,0], [0,0] ), Instance( [0,1], [1,1] ), Instance( [1,0], [1,1] ), Instance( [1,1], [0,0] ) ]

        n_inputs = 2
        n_outputs = 1
        n_hiddens = 2
        n_hidden_layers = 1

        # specify activation functions per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
        activation_functions = [ tanh_function ]*n_hidden_layers + [ sigmoid_function ]

        # initialize the neural network
        network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions)

        # start training on test set one
        network.backpropagation(training_one, ERROR_LIMIT=1e-4, learning_rate=0.3, momentum_factor=0.9  )

        # save the trained network
        # network.save_to_file( "trained_configuration.pkl" )

        # load a stored network configuration
        # network = NeuralNet.load_from_file( "trained_configuration.pkl" )

        # print out the result
        for instance in training_one:
            print(instance.features, network.update( np.array([instance.features]) ), "\ttarget:", instance.targets)