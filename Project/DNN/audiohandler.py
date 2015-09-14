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
import numpy as np
import wave
import librosa
import struct

# Class for reading files from a folder and save them in a specific structure
class AudioHandler:
    # Constants
    train_labels_directory = "train-audio-data/wavToLabel.txt"
    train_audio_files_directory = "train-audio-data/"
    test_audio_files_directory = "test-audio-data/"
    num_files_audio = 0

    # Constructor
    def __init__(self, isTestSet):
        # structure containing all the audio signals
        self.audio_data = self.read_data_from_folder(isTestSet)
        self.num_files_audio = 0

    def read_data_from_folder(self, isTestSet):
        all_signals = {}
        dir = ""

        if isTestSet == False:
            dir = self.train_audio_files_directory
        else:
            dir = self.test_audio_files_directory

        labels = [line.rstrip('\n') for line in open(self.train_labels_directory )]

        for root, dirs, files in os.walk(dir):
            label_counter = 0
            for audio in files:
                file_name = os.path.join(root, audio)

                if '.DS_Store' in file_name or 'wavToLabel.txt' in file_name:
                    continue

                try:
                    signal_data, stream_rate = librosa.load(file_name)

                    stream = wave.open(file_name,"rb")
                    num_channels = stream.getnchannels()
                    sample_width = stream.getsampwidth()
                    num_frames = 1024

                    raw_data = stream.readframes( num_frames ) # Returns byte data
                    stream.close()

                    total_samples = num_frames * num_channels

                    if sample_width == 1:
                        fmt = "%iB" % total_samples # read unsigned chars
                    elif sample_width == 2:
                        fmt = "%ih" % total_samples # read signed 2 byte shorts
                    else:
                        raise ValueError("Only supports 8 and 16 bit audio formats.")

                    chunks = struct.unpack(fmt, raw_data)
                    del raw_data # Keep memory tidy (who knows how big it might be)

                    if isTestSet == True:
                        all_signals[file_name] = signal_data, stream_rate, chunks
                    else:
                        all_signals[file_name] = signal_data, stream_rate, chunks, labels[label_counter]
                        label_counter += 1
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
                self.num_files_audio += 1

        return all_signals
