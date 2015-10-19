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
from src.libraries import utility

class Labels_Creator:
    train_audio_files_directory = '../train-audio-data/'
    test_audio_files_directory = '../test-audio-data/'
    train_results_directory = 'labels/train/'
    test_results_directory = 'labels/test/'
    sentences_directory = 'labels/sentences/'

    def creator(self, isTest=False):
        if isTest:
            audio_files_directory = self.test_audio_files_directory
            results_directory = self.test_results_directory
        else:
            audio_files_directory = self.train_audio_files_directory
            results_directory = self.train_results_directory

        for root, dirs, files in os.walk(audio_files_directory):
            for audio_file in files:
                filename = os.path.join(root, audio_file)

                if ".DS_Store" in filename:
                    continue

                phonemes_filename = utility.clean_filename(audio_file)

                # TODO Check if necessary the cleaning number method
                if isTest == False:
                    phonemes_filename = utility.clean_filename_numbers(phonemes_filename)

                # directory containing the txt files with each sentence
                get_sentences_directory = os.path.join(self.sentences_directory, phonemes_filename + '.txt')

                # result of p2fa
                output_filename = os.path.join(results_directory, audio_file.replace('.wav', '.TextGrid'))

                # call the file
                command = "python align.py " + filename + " " + get_sentences_directory + " " + output_filename
                os.system(command)

if __name__ == "__main__":
    goofy = Labels_Creator()
    goofy.creator()
    goofy.creator(True)
