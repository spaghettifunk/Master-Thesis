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

import os
import numpy as np

class Utility:

    # constants for trainset
    saved_sets_dir = 'saved-sets/'
    save_trainset_filename = 'cached_train_set.npy'
    save_testset_filename = 'cached_test_set.npy'

    def __init__(self):
        # check if path exist
        if not os.path.exists(self.saved_sets_dir):
            # create the folder
            os.makedirs(self.saved_sets_dir)

    # cache results so that ML becomes fast
    def save_set(self, obj, isTestset):
        if isTestset is True:
            np.save(os.path.join(self.saved_sets_dir, self.save_trainset_filename) , obj)
        else:
            np.save(os.path.join(self.saved_sets_dir, self.save_testset_filename) , obj)

    def load_set(self, isTestset):
        if isTestset is True:
            if not os.path.exists(os.path.join(self.saved_sets_dir, self.save_trainset_filename)):
                return None
            return np.load(os.path.join(self.saved_sets_dir, self.save_trainset_filename))
        else:
            if not os.path.exists(os.path.join(self.saved_sets_dir, self.save_testset_filename)):
                return None
            return np.load(os.path.join(self.saved_sets_dir, self.save_testset_filename))
