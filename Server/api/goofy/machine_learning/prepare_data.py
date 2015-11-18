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
import csv
import sys
import base64
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from subprocess import Popen

native_phonemes = ["AH PIYS AHV KEYK", "BLOW AH FYUWZ", "KAECH SAHM ZIYZ", "DAWN TAH DHAH WAYER", "IYGER BIYVER",
                   "FEHR AHND SKWEHR", "GEHT KOWLD FIYT", "MEHLOW AWT", "PUHLIHNG YUHR LEHGZ", "THIHNGKAHNG AWT LAWD"]
native_sentences = ["A piece of cake", "Blow a fuse", "Catch some zs", "Down to the wire", "Eager beaver",
                    "Fair and square", "Get cold feet", "Mellow out", "Pulling your legs", "Thinking out loud"]


class GMM_structure:
    stress = []
    words = []
    norm_F1 = []
    norm_F2 = []

    def __init__(self):
        self.stress = []
        self.words = []
        self.norm_F1 = []
        self.norm_F2 = []

    def set_object(self, n, val):
        if n == 0:
            self.stress.append(val)
        if n == 1:
            self.words.append(val)
        if n == 2:
            self.norm_F1.append(val)
        if n == 3:
            self.norm_F2.append(val)

    def concat_object(self, n, val):
        if n == 0:
            self.norm_F1 += val
        if n == 1:
            self.norm_F2 += val

    def get_object(self, n):
        if n == 0:
            return self.stress
        if n == 1:
            return self.words
        if n == 2:
            return self.norm_F1
        if n == 3:
            return self.norm_F2


def force_alignment(audio_file, sentence):
    path = os.path.dirname(os.path.abspath(__file__))
    results_directory = path + "/data"
    path_fa = path + "/libraries/force_alignment/"
    path_fa_sentences = path_fa + "sentences/"

    # sentence: A piece of cake -> a_piece_of_cake
    tmp_sentence = sentence.lower()
    phonemes_filename = tmp_sentence.replace(' ', '_')

    # directory containing the txt files with each sentence
    get_sentences_directory = os.path.join(path_fa_sentences, phonemes_filename + '.txt')

    # result of p2fa
    try:
        (dirName, fileName) = os.path.split(audio_file)
        output_filename = os.path.join(results_directory, fileName.replace('.wav', '.TextGrid'))

        # call the file
        command = "python " + path_fa + "align.py " + audio_file + " " + get_sentences_directory + " " + output_filename

        # run command
        proc = Popen(['/usr/local/bin/zsh', '-c', command])
        proc.wait()
    except:
        print "Error: ", sys.exc_info()
        raise


def extract_phonemes(audio_file, sentence):
    path = os.path.dirname(os.path.abspath(__file__))
    textgrid_directory = path + "/data"

    (dirName, fileName) = os.path.split(audio_file)
    output_filename = os.path.join(textgrid_directory, fileName.replace('.wav', '.txt'))

    vowel_stress = []
    phonemes = []
    with open(output_filename, 'r') as texgrid_file:
        reader = csv.reader(texgrid_file, delimiter='\t')
        all_lines = list(reader)

        i = 0
        for line in all_lines:
            if i == 0:
                i += 1
                continue

            # vowel, stress
            vowel = line[12]
            stress = line[13]
            vowel_stress.append((vowel, stress))

            # phonemes
            pre_word_trans = line[39]
            word_trans = line[40]
            fol_word_trans = line[41]

            pre_word_trans = pre_word_trans.replace(' ', '')
            if pre_word_trans != "SP" and pre_word_trans not in phonemes:
                phonemes.append(pre_word_trans)

            word_trans = word_trans.replace(' ', '')
            if word_trans != "SP" and word_trans not in phonemes:
                phonemes.append(word_trans)

            fol_word_trans = fol_word_trans.replace(' ', '')
            if fol_word_trans != "SP" and fol_word_trans not in phonemes:
                phonemes.append(fol_word_trans)

    index = native_sentences.index(sentence)
    test_phonemes = (' '.join(x for x in phonemes))
    test_phonemes = ''.join([i for i in test_phonemes if not i.isdigit()])
    current_native_phonemes = native_phonemes[index]
    wer_result, numCor, numSub, numIns, numDel = wer(current_native_phonemes, test_phonemes)
    result_wer = "Word Error Rate: {}%".format(wer_result * 100)

    return phonemes, vowel_stress, result_wer


def wer(ref, hyp, debug=False):
    DEL_PENALTY = 2
    SUB_PENALTY = 1
    INS_PENALTY = 3

    r = ref  # .split()
    h = hyp  # .split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
        return (numSub + numDel + numIns) / (float)(len(r))

    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return wer_result, numCor, numSub, numIns, numDel


def extract_data(audio_file, female=False):
    # need to change speakerfile for the female gender
    path = os.path.dirname(os.path.abspath(__file__))
    path_fave = path + "/libraries/FAVE_extract/"

    if female:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker " + path_fave + "/speakerinfo_female.speakerfile"
    else:
        config_file = "--outputFormat txt --candidates --speechSoftware Praat --formantPredictionMethod default --measurementPointMethod faav --nFormants 3 --minVowelDuration 0.001 --nSmoothing 12 --remeasure --vowelSystem phila --speaker " + path_fave + "/speakerinfo_male.speakerfile"

    textgrid_file_directory = path + "/data/"
    output_file_directory = path + "/data/"

    wav_file = audio_file
    wav_file_cleaned = wav_file.replace('.wav', '.TextGrid')

    (dirName, fileName) = os.path.split(wav_file_cleaned)

    textgrid_file = os.path.join(textgrid_file_directory, fileName)
    output_file = os.path.join(output_file_directory, fileName.replace('.TextGrid', '.txt'))

    # debug print
    command = "python " + path_fave + "bin/extractFormants.py " + config_file + " " + audio_file + " " + textgrid_file + " " + output_file
    print command

    try:
        # run command
        proc = Popen(['/usr/local/bin/zsh', '-c', command])
        proc.wait()
    except:
        print "Error: ", sys.exc_info()
        raise


def get_pitch_contour(audio_file, sentence, isFemale=False):
    try:
        path = os.path.dirname(os.path.abspath(__file__))
        path_script = path + "/libraries/pitch_contour/pitch_intensity_formants.praat"

        (dirName, fileName) = os.path.split(audio_file)
        output_name = fileName.replace(".wav", ".csv")
        output_folder = path + "/data/" + output_name

        sentence = sentence.lower()
        sentence = sentence.replace(' ', '_')

        if isFemale:
            min_pitch = '75'
            native_csv = path + "/data/native/female/" + sentence + ".csv"
        else:
            min_pitch = '50'
            native_csv = path + "/data/native/male/" + sentence + ".csv"

        # see script file for the usage
        proc = Popen(
            ['/Applications/Praat.app/Contents/MacOS/Praat', path_script, audio_file, output_folder, 'wav', '10',
             min_pitch, '500', '11025'])
        proc.wait()

        # TODO: Read user and native csv files
        # native
        native_pitch = []
        with open(native_csv, 'r') as native_file:
            reader = csv.reader(native_file, delimiter=',')
            all_lines = list(reader)

            for line in all_lines:
                if line[1] == 'pitch':
                    continue
                native_pitch.append(line[1])

        # user
        user_pitch = []
        with open(output_folder, 'r') as user_file:
            reader = csv.reader(user_file, delimiter=',')
            all_lines = list(reader)

            for line in all_lines:
                if line[1] == 'pitch':
                    continue
                user_pitch.append(line[1])

        # Padding with 0s on the end
        if len(native_pitch) != len(user_pitch):
            copy_native_pitch = native_pitch
            index = 0
            for val in copy_native_pitch:
                if val == 0 or val == '0':
                    del native_pitch[index]
                    index += 1
                else:
                    break

            copy_user_pitch = user_pitch
            index = 0
            for val in copy_user_pitch:
                if val == 0 or val == '0':
                    del user_pitch[index]
                    index += 1
                else:
                    break

            length_native = len(native_pitch)
            length_user = len(user_pitch)
            if length_native > length_user:
                diff = length_native - length_user
                temp = ['0'] * (diff)
                user_pitch += temp

            elif length_user > length_native:
                diff = length_user - length_native
                temp = ['0'] * (diff)
                native_pitch += temp

        # Create scatter image
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        time = []
        val = 0
        for i in range(len(native_pitch)):
            val += 0.1
            time.append(val)

        # Normalized Data
        normalized_native = []
        normalized_native_floats = [float(x) for x in native_pitch]
        for val in normalized_native_floats:
            dd = (val - min(normalized_native_floats)) / (max(normalized_native_floats) - min(normalized_native_floats))
            normalized_native.append(dd)

        normalized_user = []
        normalized_user_floats = [float(x) for x in user_pitch]
        for val in normalized_user_floats:
            dd = (val - min(normalized_user_floats)) / (max(normalized_user_floats) - min(normalized_user_floats))
            normalized_user.append(dd)

        # plot pitch
        ax1.scatter(time, normalized_native, s=100, c='r', marker='.', label='Native Pitch')
        ax1.scatter(time, normalized_user, s=100, c='b', marker='+', label='User Pitch')

        plt.xlabel('Time (sec)')
        plt.ylabel('Normalized Frequency (Hz)')

        fontP = FontProperties()
        fontP.set_size('x-small')
        plt.grid('on')

        lgd = plt.legend(loc='lower center', ncol=2, prop=fontP)
        plt.title('Stress trend')

        # Save as bynary file
        plot_filename = audio_file.replace('.wav', '_pitch.png')
        plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight', transparent=True)

        with open(plot_filename, "rb") as imageFile:
            return base64.b64encode(imageFile.read())
    except:
        print "Error: ", sys.exc_info()
        raise


def create_test_data(filename):
    path = os.path.dirname(os.path.abspath(__file__))
    path_data = path + "/data/"

    txt_file = path_data + filename.replace('.wav', '_norm.txt')
    csv_file = path_data + filename.replace('.wav', '.csv')

    # use 'with' if the program isn't going to immediately terminate
    # so you don't leave files open
    # the 'b' is necessary on Windows
    # it prevents \x1a, Ctrl-z, from ending the stream prematurely
    # and also stops Python converting to / from different line terminators
    # On other platforms, it has no effect

    with open(txt_file, "rb") as opened_txt:
        in_txt = csv.reader(opened_txt, delimiter='\t')

        with open(csv_file, 'wb') as opened_csv:
            out_csv = csv.writer(opened_csv)
            out_csv.writerows(in_txt)

    all_data = dict()
    with open(csv_file, 'r') as tabbed_file:
        reader = csv.reader(tabbed_file, delimiter="\t")
        all_lines = list(reader)

        not_included = 0
        for line in all_lines:
            if not_included <= 2:
                not_included += 1
                continue

            l = line[0].split(',')

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


def create_test_set(test_data):
    try:
        X_test = test_data.values()
        Y_test = test_data.keys()

        return X_test, Y_test
    except:
        print "Error: ", sys.exc_info()
        raise
