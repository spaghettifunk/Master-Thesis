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

labels_directory = "output-data/labels_p2fa/"
phonemes_dict = "../p2fa-force_alignment/model/monophones"
dict_phonemes_directory = "output-data/audio_phonemes_labels.txt"

phonemes_arr = []
with open(phonemes_dict) as dict_file:
    data = dict_file.readlines()
    for line in data:
        cleaned = line.replace('\n', '')
        phonemes_arr.append(cleaned)

phonemes = {}
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

        phonemes[file] = temp_phonemes

json.dump(phonemes, open(dict_phonemes_directory, 'w'))
