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

import sys
import os
import librosa

# Class for reading files from a folder and save them in a specific structure
class AudioHandler:
    # Constants
    train_audio_files_directory = "train-audio-data/"
    test_audio_files_directory = "test-audio-data/"

    # Constructor
    def __init__(self, isTestSet):
        # structure containing all the audio signals
        self.audio_data = self.read_data_from_folder(isTestSet)

    def read_data_from_folder(self, isTestSet):
        all_signals = {}
        dir = ""

        if isTestSet is True:
            dir = self.train_audio_files_directory
        else:
            dir = self.test_audio_files_directory

        for root, dirs, files in os.walk(dir):
            for audio in files:
                file_name = os.path.join(root, audio)

                if '.DS_Store' in file_name:
                    continue

                try:
                    signal_data, stream_rate = librosa.load(file_name)
                    all_signals[file_name] = signal_data, stream_rate
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

        return all_signals
