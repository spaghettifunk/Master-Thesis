__author__ = 'davideberdin'

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
import itertools
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture


class GmmStructure:
    stress = []
    words = []
    norm_F1 = []
    norm_F2 = []

    def __init__(self):
        self.stress = []
        self.words = []
        self.norm_F1 = []
        self.norm_F2 = []

    def set_object(self, n, val):
        if n == 0:
            self.stress.append(val)
        if n == 1:
            self.words.append(val)
        if n == 2:
            self.norm_F1.append(val)
        if n == 3:
            self.norm_F2.append(val)

    def concat_object(self, n, val):
        if n == 0:
            self.norm_F1 += val
        if n == 1:
            self.norm_F2 += val

    def get_object(self, n):
        if n == 0:
            return self.stress
        if n == 1:
            return self.words
        if n == 2:
            return self.norm_F1
        if n == 3:
            return self.norm_F2


def bic_model(X, classes, name_i):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, classes)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, init_params='wmc', min_covar=0.001,
                              n_init=1, n_iter=100, params='wmc', random_state=None, thresh=None, tol=0.001)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    fig = plt.figure()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)],width=.2, color=color))

    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))

    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    path = os.path.dirname(os.path.abspath(__file__))
    pics_file = os.path.join(path, 'pics/')
    os.chdir(pics_file)

    name = 'bic_' + str(name_i) + '.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    return best_gmm


def aic_model(X, classes, name_i):
    lowest_aic = np.infty
    aic = []
    n_components_range = range(1, classes)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, init_params='wmc', min_covar=0.001,
                              n_init=1, n_iter=100, params='wmc', random_state=None, thresh=None, tol=0.001)
            gmm.fit(X)
            aic.append(gmm.aic(X))
            if aic[-1] < lowest_aic:
                lowest_aic = aic[-1]
                best_gmm = gmm

    aic = np.array(aic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    fig = plt.figure()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, aic[i * len(n_components_range):(i + 1) * len(n_components_range)],width=.2, color=color))

    plt.xticks(n_components_range)
    plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
    plt.title('AIC score per model')
    xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(aic.argmin() / len(n_components_range))

    plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    path = os.path.dirname(os.path.abspath(__file__))
    pics_file = os.path.join(path, 'pics/')
    os.chdir(pics_file)

    name = 'aic_' + str(name_i) + '.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    return best_gmm


def create_structure():
    try:
        formants_files_directory = 'data/formants_results/files/'
        all_data = dict()

        path = os.path.dirname(os.path.abspath(__file__))
        formants_files = os.path.join(path, formants_files_directory)
        os.chdir(formants_files)

        for filename in os.listdir("."):

            if ".DS_Store" in filename or "_norm" not in filename:
                continue

            cleaned_filename = filename.replace(".txt", "")
            cleaned_filename = cleaned_filename.replace('_norm', '')
            last_index = cleaned_filename.rfind("_")
            cleaned_filename = cleaned_filename[:last_index]

            training_data = dict()

            with open(filename, 'r') as tabbed_file:
                reader = csv.reader(tabbed_file, delimiter="\n")
                all_lines = list(reader)

                not_included = 0
                for line in all_lines:
                    if not_included <= 2:
                        not_included += 1
                        continue

                    l = line[0].split('\t')
                    data = GmmStructure()

                    data.set_object(0, l[1])
                    data.set_object(1, l[2])
                    try:
                        if l[3] == '':
                            f1_val = 0.0
                        else:
                            f1_val = float(l[3])

                        if l[4] == '':
                            f2_val = 0.0
                        else:
                            f2_val = float(l[4])

                        data.set_object(2, f1_val)
                        data.set_object(3, f2_val)
                    except:
                        print "Error: ", sys.exc_info()

                    if l[0] in training_data:
                        # append the new number to the existing array at this slot
                        obj = training_data.get(l[0])

                        # we use it only for phoneme prediction
                        obj.concat_object(0, data.norm_F1)
                        obj.concat_object(1, data.norm_F2)

                        training_data[l[0]] = obj
                    else:
                        # create a new array in this slot
                        training_data[l[0]] = data

            if cleaned_filename in all_data:
                curr = all_data.get(cleaned_filename)
                vowels = curr.keys()

                for key, value in training_data.items():
                    if key in vowels:  # the vowel is present - otherwise mistake
                        old_gmm_struct = curr.get(key)

                        old_gmm_struct.concat_object(0, value.norm_F1)
                        old_gmm_struct.concat_object(1, value.norm_F2)

                        curr[key] = old_gmm_struct
                    else:
                        curr[key] = value
            else:
                all_data[cleaned_filename] = training_data

        return all_data
    except Exception as e:
        pass


if __name__ == "__main__":
    all_data = create_structure()

    keys = all_data.keys()
    n_classes = len(np.unique(keys))

    i = 0
    for data in all_data.values():
        for val in data.values():
            f1 = val.get_object(2)
            f2 = val.get_object(3)
            data = zip(f1, f2)
            if len(data) >= n_classes:
                try:
                    X = np.array(data)
                    bic_m = bic_model(X, n_classes, i)
                    aic_m = aic_model(X, n_classes, i)
                    i += 1

                    print "Done"
                except Exception, e:
                    print str(e)

    print "Check graphs"
