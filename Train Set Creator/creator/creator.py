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
from FAVE_extract.extract_data import FormantsExtractor

train_audio_files_directory = 'sounds/'
train_results_directory = 'p2fa/labels/train/'
sentences_directory = 'p2fa/labels/sentences/'

fave_output = 'FAVE_extract/train/out/'
fave_textgrid = 'FAVE_extract/train/train/'


def creator():
    audio_files_directory = train_audio_files_directory
    results_directory = train_results_directory

    for root, dirs, files in os.walk(audio_files_directory):
        for audio_file in files:
            filename = os.path.join(root, audio_file)

            if ".DS_Store" in filename:
                continue

            last_index = audio_file.rfind("_")
            phonemes_filename = audio_file[:last_index]

            # directory containing the txt files with each sentence
            get_sentences_directory = os.path.join(sentences_directory, phonemes_filename + '.txt')

            # result of p2fa
            output_filename = os.path.join(results_directory, audio_file.replace('.wav', '.TextGrid'))

            # call the file
            command = "python p2fa/align.py " + filename + " " + get_sentences_directory + " " + output_filename
            os.system(command)


def fave():

    audio_files_directory = train_audio_files_directory
    results_directory = fave_output

    for root, dirs, files in os.walk(audio_files_directory):
        for audio_file in files:
            filename = os.path.join(root, audio_file)

            if ".DS_Store" in filename:
                continue

            # last_index = audio_file.rfind("_")
            # phonemes_filename = audio_file[:last_index]

            # result of p2fa
            textgrid_file = os.path.join(fave_textgrid, audio_file.replace('.wav', '.TextGrid'))
            output_file = os.path.join(results_directory, audio_file.replace('.wav', '.txt'))

            f = FormantsExtractor()
            f.extract_data(filename, textgrid_file, output_file)

if __name__ == "__main__":
    # creator()
    # fave() # -> see FAVE_extract/extract_data.py
    x = 0