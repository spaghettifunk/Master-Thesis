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

import os
import sys
import csv
import numpy as np

from sklearn import mixture
from copy import deepcopy


class GMM_structure:
    filename = ""
    vowels = []
    stress = []
    words = []
    norm_F1 = []
    norm_F2 = []

    def __init__(self, fn):
        self.filename = fn
        self.vowels = []
        self.stress = []
        self.words = []
        self.norm_F1 = []
        self.norm_F2 = []

    def set_object(self, n, val):
        if n == 0:
            self.vowels.append(val)
        if n == 1:
            self.stress.append(val)
        if n == 2:
            self.words.append(val)
        if n == 3:
            self.norm_F1.append(val)
        if n == 4:
            self.norm_F2.append(val)

    def get_object(self, n):
        if n == 0:
            return self.vowels
        if n == 1:
            return self.stress
        if n == 2:
            return self.words
        if n == 3:
            return self.norm_F1
        if n == 4:
            return self.norm_F2


class GMM_prototype:
    #region Global variables
    output_files_directory = 'output-data/formants_results/male/'
    #endregion

    # Train model with GMM
    def create_structure(self):
        all_data = []
        for root, dirs, files in os.walk(self.output_files_directory):
            for file in files:
                filename = os.path.join(root, file)

                if ".DS_Store" in filename or "_norm" not in filename:
                    continue

                with open(filename, 'r') as tabbed_file:
                    reader = csv.reader(tabbed_file, delimiter="\t")
                    all_lines = list(reader)

                    data = GMM_structure(file)

                    not_included = 0
                    for l in all_lines:
                        if not_included <= 2:
                            not_included += 1
                            continue

                        data.set_object(0, l[0])
                        data.set_object(1, l[1])
                        data.set_object(2, l[2])
                        try:
                            if l[3] == '':
                                f1_val = 0.0
                            else:
                                f1_val = float(l[3])

                            if l[4] == '':
                                f2_val = 0.0
                            else:
                                f2_val = float(l[4])

                            data.set_object(3, f1_val)
                            data.set_object(4, f2_val)
                        except:
                            print "Error: ", sys.exc_info()

                    all_data.append(deepcopy(data))
        return all_data

    def train_GMM(self):
        data = self.create_structure()

        all_vowels = []
        all_norm_f1 = []
        all_norm_f2 = []

        for str in data:
            temp_vowels = np.array(str.get_object(0))
            temp_norm_f1 = np.array(str.get_object(3))
            temp_norm_f2 = np.array(str.get_object(4))

            all_vowels.extend(temp_vowels)
            all_norm_f1.extend(temp_norm_f1)
            all_norm_f2.extend(temp_norm_f2)

        try:
            X_train = []
            for f, b in zip(all_norm_f1, all_norm_f2):
                X_train.append([f,b])

            X_train = np.array(X_train)
            Y_train = np.vstack(all_vowels)

            n_classes = len(np.unique(Y_train))
            gmm_classifier = mixture.GMM(n_components=n_classes, covariance_type='full')
            gmm_classifier.fit(X_train)

        except:
            print sys.exc_info()
            raise