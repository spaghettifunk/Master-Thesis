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
from subprocess import call, Popen

class FormantsExtractor:
    audio_files_directory = '../../train-audio-data/'
    textgrid_file_directory = 'train/'
    output_file_directory = 'results/'

    male_name = ['Jeremy', 'Lenny', 'Philip']
    female_name = ['Marty', 'Joyce', 'Niki']

    # Extract phonemes, stress, normalized formants with FAVE-extract
    def extract_data(self, female=False):
        # need to change speakerfile for the female gender
        if female:
            self.textgrid_file_directory += 'female/'
            self.audio_files_directory += 'female/'
        else:
            self.textgrid_file_directory += 'male/'
            self.audio_files_directory += 'male/'

        for root, dirs, files in os.walk(self.audio_files_directory):
            for audio_file in files:
                if ".DS_Store" in audio_file:
                    continue

                wav_file = os.path.join(root, audio_file)

                wav_file_cleaned = audio_file.replace('.wav', '.TextGrid')
                textgrid_file = os.path.join(self.textgrid_file_directory, wav_file_cleaned)
                output_file = os.path.join(self.output_file_directory, audio_file.replace('.wav', '.txt'))

                # debug print
                command = "python bin/extractFormants.py +config.txt " + wav_file + " " + textgrid_file + " " + output_file + " >> output.txt"
                print command

                # run command
                Popen(['/usr/local/bin/zsh', '-c', command])


if __name__ == "__main__":
    goofy = FormantsExtractor()
    goofy.extract_data()
    goofy.extract_data(True)
