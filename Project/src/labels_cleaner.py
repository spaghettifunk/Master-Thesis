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
import json

class Labels_Cleaner:
    test_labels_directory = "p2fa-force_alignment/labels/test/"
    train_labels_directory = "p2fa-force_alignment/labels/train/"
    phonemes_dict = "p2fa-force_alignment/model/monophones"
    train_dict_phonemes_directory = "output-data/train_audio_phonemes_labels.txt"
    test_dict_phonemes_directory = "output-data/test_audio_phonemes_labels.txt"

    def create_phonemes_dictionary(self, isTestSet=False):
        phonemes_arr = {}
        phonemes_int_label = 0  # assign an integer value to  each phoneme (used for training the model)
        with open(self.phonemes_dict) as dict_file:
            data = dict_file.readlines()
            for line in data:
                cleaned = line.replace('\n', '')
                phonemes_arr[cleaned] = phonemes_int_label
                phonemes_int_label += 1

        phonemes = {}

        if isTestSet:
            labels_directory = self.test_labels_directory
        else:
            labels_directory = self.train_labels_directory

        for root, dirs, files in os.walk(labels_directory):
            for file in files:
                filename = os.path.join(root, file)

                if ".DS_Store" in filename:
                    continue

                temp_phonemes = []
                with open(filename) as label_file:
                    data = label_file.readlines()

                    for line in data:
                        cleaned = line.replace('\n', '')
                        cleaned = cleaned.replace('"', '')

                        if cleaned == 'sp':
                            continue

                        if cleaned in phonemes_arr:
                            temp_phonemes.append(cleaned)

                # create the mapping between the retrieved phonems and the
                # actual value as number
                phonemes_int_labels = []
                for tp in temp_phonemes:
                    int_val = phonemes_arr[tp]
                    phonemes_int_labels.append(int_val)

                # save it in the dictionary
                phonemes[file] = phonemes_int_labels

        if isTestSet:
            json.dump(phonemes, open(self.test_dict_phonemes_directory, 'w'))
        else:
            json.dump(phonemes, open(self.train_dict_phonemes_directory, 'w'))


if __name__ == "__main__":
    goofy = Labels_Cleaner()
    goofy.create_phonemes_dictionary()
    goofy.create_phonemes_dictionary(True)
