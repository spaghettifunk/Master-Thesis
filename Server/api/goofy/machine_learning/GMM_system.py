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
import matplotlib as mpl

from matplotlib.font_manager import FontProperties
from sklearn import mixture
from prepare_data import GMM_structure
from libraries.utility import clean_filename, clean_filename_numbers


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

            cleaned_filename = clean_filename(filename)
            cleaned_filename = clean_filename_numbers(cleaned_filename)
            cleaned_filename = cleaned_filename.replace('_norm', '')

            training_data = dict()

            with open(filename, 'r') as tabbed_file:
                reader = csv.reader(tabbed_file, delimiter="\t")
                all_lines = list(reader)

                not_included = 0
                for l in all_lines:
                    if not_included <= 2:
                        not_included += 1
                        continue

                    data = GMM_structure()

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
                    if key in vowels: # the vowel is present - otherwise mistake
                        old_gmm_struc = curr.get(key)

                        old_gmm_struc.concat_object(0, value.norm_F1)
                        old_gmm_struc.concat_object(1, value.norm_F2)

                        curr[key] = old_gmm_struc
            else:
                all_data[cleaned_filename] = training_data

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

        all_data = self.create_structure(isFemale)
        path = os.path.dirname(os.path.abspath(__file__))

        try:

            keys = all_data.keys()
            n_classes = len(np.unique(keys))
            gmm_classifier = mixture.GMM(n_components=n_classes, covariance_type='tied', params='mc')

            for data in all_data.values():
                for val in data.values():
                    f1 = val.get_object(2)
                    f2 = val.get_object(3)
                    data = zip(f1, f2)
                    if len(data) >= n_classes:
                        gmm_classifier.fit(data)

            # save data
            if isFemale:
                path_trainset = path + self.female_trainset_name
            else:
                path_trainset = path + self.male_trainset_name

            with open(path_trainset, 'wb') as fid:
                cPickle.dump(all_data, fid)

            # save the classifier
            if isFemale:
                model_name = self.female_model_name
            else:
                model_name = self.male_model_name

            model_directory = os.path.join(path, model_name)

            with open(model_directory, 'wb') as fid:
                cPickle.dump(gmm_classifier, fid)

        except:
            print "Error: ", sys.exc_info()
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
            all_data = cPickle.load(traindata)


        all_vowels = []
        for key, val in all_data.items():
            for v in val.keys():
                all_vowels.append(v)

        labels = np.unique(all_vowels)
        int_labels = np.arange(len(labels))

        map_int_label = dict(zip(int_labels, labels))

        # results
        key_sentence = sentence.lower()
        key_sentence = key_sentence.replace(' ', '_')

        train_dict = all_data.get(key_sentence)
        X_train = train_dict.values()
        Y_train = train_dict.keys()

        try:
            plt.figure(1)
            plt.subplots_adjust(wspace=0.4, hspace=0.5)

            colors = ['b', 'g', 'c', 'm', 'y', 'k']

            # markers_predicted = ['s', 'p', '*', '+', 'd', 'D']
            # markers_native = ['.',',','+','x','_','|']

            predicted_formants = []
            p = 0
            index = 1
            for val in X_test:
                f1 = val.norm_F1
                f2 = val.norm_F2
                data = zip(f1, f2)

                gmm_predict = gmm_classifier.predict(data)
                predicted_formants.append(gmm_predict.tolist())

                # print the predicted-vowels based on the formants
                for l in gmm_predict.tolist():
                    plt.subplot(2, 2, index)
                    plt.scatter(f1, f2, s=80, c='r', marker='+', label=r"$ {} $".format(map_int_label[l]))
                    p += 1
                index += 1

            predicted_labels = []
            for list in predicted_formants:
                for l in list:
                    predicted_labels.append(map_int_label[l])

            native_vowels = self.get_native_vowels(sentence)
            uniq_predicted_labels = np.unique(predicted_labels)

            if uniq_predicted_labels.shape != np.array(native_vowels).shape:
                test_accuracy = 0.0
            else:
                test_accuracy = np.mean(uniq_predicted_labels == np.array(native_vowels)) * 100

            # predict the native set for that sentence
            i = 0
            duplicate = []
            native_data = dict(zip(Y_train, X_train))
            index = 1
            for n in native_vowels:

                if n in duplicate:
                    continue

                struct = native_data[n]

                native_f1 = struct.get_object(2)
                native_f2 = struct.get_object(3)

                ax = plt.subplot(2, 2, index)
                ax.scatter(native_f1, native_f2, s=40, c=colors[i], marker='.', label=r"$ {} $".format(n))
                axes = plt.gca()
                axes.set_xlim([min(native_f1) - 500, max(native_f1) + 500])
                axes.set_ylim([min(native_f2) - 500, max(native_f2) + 500])
                ax.set_xlabel('F1')
                ax.set_ylabel('F2')
                ax.set_title("Vowel: " + n)

                #fontP = FontProperties()
                #fontP.set_size('x-small')

                #plt.grid('on')
                #plt.legend(loc='upper center', ncol=2, prop=fontP)

                # ellipse inside graph
                self.make_ellipses(ax, native_f1, native_f2)
                duplicate.append(n)

                i += 1
                index += 1

            plt.savefig(plot_filename, bbox_inches='tight', transparent=True)

            with open(plot_filename, "rb") as imageFile:
                return base64.b64encode(imageFile.read())
        except:
            print "Error: ", sys.exc_info()
            raise

    def make_ellipses(self, ax, native_f1, native_f2):
        x1 = min(native_f1)
        x2 = max(native_f1)
        y1 = min(native_f2)
        y2 = max(native_f2)

        ellipse = mpl.patches.Ellipse(xy=((x2+x1)/2, (y2+y1)/2), width=(x2-x1)*1.4, height=(y2-y1)*1.2)
        ellipse.set_edgecolor('r')
        ellipse.set_facecolor('none')
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.5)
        ax.add_artist(ellipse)

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
