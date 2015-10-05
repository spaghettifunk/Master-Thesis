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

audio_files_directory = 'female/'
results_directory = 'labels/results/'
sentences_directory = 'labels/sentences/'

def clean_filename(filename):
    cleaned_name = filename.replace(".wav", "")
    if "Jeremy_" in cleaned_name:
        return cleaned_name.replace("Jeremy_", "")
    elif "Lenny_" in cleaned_name:
        return cleaned_name.replace("Lenny_", "")
    elif "Philip_" in cleaned_name:
        return cleaned_name.replace("Philip_", "")
    elif "Marty_" in cleaned_name:
        return cleaned_name.replace("Marty_", "")
    elif "Joyce_" in cleaned_name:
        return cleaned_name.replace("Joyce_", "")
    elif "Niki_" in cleaned_name:
        return cleaned_name.replace("Niki_", "")
    else:
        return ""

def clean_filename_numbers(filename):
    if "_1" in filename:
        return filename.replace("_1", "")
    elif "_2" in filename:
        return filename.replace("_2", "")
    elif "_3" in filename:
        return filename.replace("_3", "")
    elif "_4" in filename:
        return filename.replace("_4", "")
    else:
        return ""

for root, dirs, files in os.walk(audio_files_directory):
    for audio_file in files:
        filename = os.path.join(root, audio_file)

        if ".DS_Store" in filename:
            continue

        phonemes_filename = clean_filename(audio_file)
        phonemes_filename = clean_filename_numbers(phonemes_filename)

        # directory containing the txt files with each sentence
        get_sentences_directory = os.path.join(sentences_directory, phonemes_filename + '.txt')

        # result of p2fa
        output_filename = os.path.join(results_directory, audio_file.replace('.wav', '.TextGrid'))

        # call the file
        command = "python align.py " + filename + " " + get_sentences_directory + " " + output_filename;
        os.system(command)
