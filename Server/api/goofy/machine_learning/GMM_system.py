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
import cPickle
import base64

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn import mixture

from prepare_data import GMM_structure


class GMM_prototype:
    # region Global variables
    male_formants_files_directory = 'data/formants_results/male/'
    female_formants_files_directory = 'data/formants_results/female/'
    male_model_name = 'models/gmm_male_model.pkl'
    female_model_name = 'models/gmm_female_model.pkl'

    male_trainset_name = '/models/trainset_male.pkl'
    female_trainset_name = '/models/trainset_female.pkl'

    native_vowels = '/data/labels.txt'
    native_sentences = '/data/sentences.txt'
    # endregion

    # Train model with GMM
    def create_structure(self, isFemale=False):
        if isFemale:
            self.formants_files_directory = self.female_formants_files_directory
        else:
            self.formants_files_directory = self.female_formants_files_directory

        all_data = dict()

        path = os.path.dirname(os.path.abspath(__file__))
        formants_files = os.path.join(path, self.formants_files_directory)
        os.chdir(formants_files)

        for filename in os.listdir("."):

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

                    if l[0] in all_data:
                        # append the new number to the existing array at this slot
                        obj = all_data[l[0]]

                        # we use it only for phoneme prediction
                        obj.concat_object(0, data.norm_F1)
                        obj.concat_object(1, data.norm_F2)

                        all_data[l[0]] = obj
                    else:
                        # create a new array in this slot
                        all_data[l[0]] = data
        return all_data

    def get_native_vowels(self, sentence):

        path = os.path.dirname(os.path.abspath(__file__))
        label_path = path + self.native_vowels
        sentences_path = path + self.native_sentences

        s = sentence.lower()

        vowels = []
        with open(label_path, 'rb') as vowels_file:
            reader = csv.reader(vowels_file, delimiter='\n')
            all_lines = list(reader)

            for line in all_lines:
                l = line[0].split(' ')
                vowels.append(l)

        sentences = []
        with open(sentences_path, 'rb') as sentences_file:
            reader = csv.reader(sentences_file, delimiter='\n')
            all_lines = list(reader)

            for line in all_lines:
                sen = line[0]
                sentences.append(sen)

        result = dict(zip(sentences, vowels))
        return result[s]


    def train_GMM(self, isFemale=False):

        data = self.create_structure(isFemale)
        path = os.path.dirname(os.path.abspath(__file__))

        # all_vowels = []
        # all_norm_f1 = []
        # all_norm_f2 = []

        # for str in data:
        #     temp_vowels = np.array(str.get_object(0))
        #     temp_norm_f1 = np.array(str.get_object(3))
        #     temp_norm_f2 = np.array(str.get_object(4))
        #
        #     all_vowels.extend(temp_vowels)
        #     all_norm_f1.extend(temp_norm_f1)
        #     all_norm_f2.extend(temp_norm_f2)

        try:
            # X_train = []
            # for f, b in zip(all_norm_f1, all_norm_f2):
            #     X_train.append([f, b])
            #
            # X_train = np.array(X_train)
            # Y_train = np.vstack(all_vowels)

            X_train = data.values()
            Y_train = data.keys()

            if isFemale:
                path_trainset = path + self.female_trainset_name
            else:
                path_trainset = path + self.male_trainset_name

            with open(path_trainset, 'wb') as fid:
                cPickle.dump([X_train, Y_train], fid)

            n_classes = len(np.unique(Y_train))
            gmm_classifier = mixture.GMM(n_components=n_classes, covariance_type='tied', params='mc')

            for val in data.values():
                f1 = val.get_object(2)
                f2 = val.get_object(3)
                data = zip(f1, f2)
                if len(data) >= n_classes:
                    gmm_classifier.fit(data)

                # f1 = np.vstack(f1)
                # if len(f1) >= n_classes:
                #     gmm_classifier.fit(f1)
                #
                # f2 = val.get_object(3)
                # f2 = np.vstack(f2)
                # if len(f2) >= n_classes:
                #     gmm_classifier.fit(f2)

            # save the classifier
            if isFemale:
                model_name = self.female_model_name
            else:
                model_name = self.male_model_name

            model_directory = os.path.join(path, model_name)

            with open(model_directory, 'wb') as fid:
                cPickle.dump(gmm_classifier, fid)

        except:
            print "counter: {0} - Error: {1}", c, sys.exc_info()
            raise

    def test_GMM(self, X_test, Y_test, plot_filename, sentence, isFemale=False):

        path = os.path.dirname(os.path.abspath(__file__))
        path += '/'

        if isFemale:
            model_name = path + self.female_model_name
            trainset_name = path + self.female_trainset_name
        else:
            model_name = path + self.male_model_name
            trainset_name = path + self.male_trainset_name

        # load it again
        with open(model_name, 'rb') as model:
            gmm_classifier = cPickle.load(model)

        with open(trainset_name, 'rb') as traindata:
            X_train, Y_train = cPickle.load(traindata)

        # results
        labels = np.unique(Y_train)
        int_labels = np.arange(len(labels))

        map_int_label = dict(zip(int_labels, labels))
        map_label_int = dict(zip(labels, int_labels))

        try:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            markers = ['.',',','v','^','>','<','o']

            predicted_formants = []
            p = 0
            for val in X_test:
                f1 = val.norm_F1
                f2 = val.norm_F2
                data = zip(f1, f2)

                gmm_logprob, gmm_resp = gmm_classifier.score_samples(data)
                gmm_res = sum(gmm_classifier.score(data))
                gmm_predict = gmm_classifier.predict(data)
                gmm_predict_proba = gmm_classifier.predict_proba(data)

                predicted_formants.append(gmm_predict.tolist())

                # print the predicted-vowels based on the formants
                for l in gmm_predict.tolist():
                    ax1.scatter(f2, f1, s=40, c='r', marker='o', label=r"$ {} $".format(map_int_label[l]))
                    p += 1

            predicted_labels = []
            for list in predicted_formants:
                for l in list:
                    predicted_labels.append(map_int_label[l])

            native_vowels = self.get_native_vowels(sentence)
            test_accuracy = np.mean(predicted_labels == np.array(native_vowels)) * 100

            # predict the native set for that sentence
            i = 0
            native_data = dict(zip(Y_train, X_train))
            for n in native_vowels:
                struct = native_data[n]
                native_f1 = struct.get_object(2)
                native_f2 = struct.get_object(3)

                ax1.scatter(native_f2, native_f1, s=40, c=colors[i], marker=markers[i], label=r"$ {} $".format(n))
                i += 1

            plt.xlabel('F2')
            plt.ylabel('F1')

            fontP = FontProperties()
            fontP.set_size('x-small')

            plt.grid('on')
            lgd = plt.legend(loc='best', ncol=(i + p), prop=fontP)

            plt.title('Vowel Predicted - Test accuracy: %.3f' % test_accuracy)
            plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

            with open(plot_filename, "rb") as imageFile:
                return base64.b64encode(imageFile.read())
        except:
            print "Error: ", sys.exc_info()
            raise

    def models_if_exist(self):
        try:
            path = os.path.dirname(os.path.abspath(__file__))

            female_model = os.path.join(path, self.female_model_name)
            male_model = os.path.join(path, self.male_model_name)

            exist_female = os.path.exists(female_model)
            exist_male = os.path.exists(male_model)

            return (exist_female and exist_male)
        except:
            return False
