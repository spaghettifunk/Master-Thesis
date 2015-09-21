#!/usr/bin/env python3
# encoding: utf-8
'''
prepare_analysis -- Prepare the annotations and recordings for statistical use

prepare_analysis is a description

It defines classes_and_methods

Tested on a Mac. Should work on Linux (adjust Praat location). On Windows YMMV

@author:     Sheean Spoel
        
@copyright:  2013 Universiteit van Amsterdam. All rights reserved.
        
@license:    MIT

@contact:    sheean@sheean.nl
@deffield    updated: Updated
'''

import sys
import os
import tempfile
import subprocess

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from Lib import stimuli

praat_location = "/Applications/Praat.app/Contents/MacOS/Praat" 
csv_delimiter = ';'

__version__ = 0.1
__date__ = '2013-06-15'
__updated__ = '2013-07-21'

class Interval:
    start = None
    end = None
    word = None
    silence_before = 0.0
    silence_after = 0.0
    offset = 0.0
    
    @property
    def duration(self):
        return self.end - self.start
    
def main(argv=None):
    global praat_location
    
    '''Command line options.'''
    
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by Sheean Spoel on %s.
  Copyright © 2013 Universiteit van Amsterdam. All rights reserved.
  
  Permission is hereby granted, free of charge, to any person obtaining a 
  copy of this software and associated documentation files (the "Software"), 
  to deal in the Software without restriction, including without limitation the 
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
  sell copies of the Software, and to permit persons to whom the Software is 
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in 
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
  SOFTWARE.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-sid", "--subject_id", dest="subject_id",
                            help="Subject identifier e.g. S001. Default: %(default)s")
        parser.add_argument("-sex", "--sex", dest="sex",
                            help="Subject sex 'male' or 'female'. Default: %(default)s")
        parser.add_argument("-st", "--stimuli", dest="stimuli",
                            type=int,
                            help="Number of stimuli. Default: %(default)s")
        parser.add_argument("-e", "--exclude", dest="exclude",
                            help="Exclude sentence from analysis, separated by comma")
        
        parser.set_defaults(subject_id='S000', sex='female',stimuli=80,exclude=None)
        
        # Process arguments
        args = parser.parse_args()
        
        prepare_analysis(args.subject_id, args.sex, args.stimuli, 
                         [] if args.exclude == None 
                         else [int(x) for x in args.exclude.split(',')])
        
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help\n\n")
        raise(e)


def generate_praat_script(subject_id, sex, stimuli, base_dirname, 
                          temp_directory_name, script_location, exclude):
    try:
        praat_script = open(script_location, 'w')
        
        # Same values as Gubian 2010        
        if sex == 'female':
            min_pitch = 150
            max_pitch = 500
        else:
            min_pitch = 70 
            max_pitch = 300
            
        mapping = {
            'base_dirname':base_dirname, 
            'subject_id':subject_id, 
            'temp_directory':temp_directory_name, 
            'min_pitch':min_pitch, 
            'max_pitch':max_pitch }
        
        praat_script.write(str.format_map("""
do ("Read from file...", "{base_dirname}/Annotation/{subject_id}.Collection")
""", mapping))
        
        for i in range(0, stimuli):
            if i in exclude:
                continue
            
            mapping['i'] = i
            praat_script.write(str.format_map("""
# open sound file
do ("Read from file...", "{base_dirname}/Output/Recordings/{subject_id}/{i}.wav")

# convert sound to pitch
select Sound {i}
do ("To Pitch (ac)...", 0, {min_pitch}, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, {max_pitch})
do ("Down to PitchTier")
do ("Save as headerless spreadsheet file...", "{temp_directory}/PitchTier {i}.txt")

# convert sound to intensity
select Sound {i}
do ("To Intensity...", 100, 0, "yes")
do ("Down to IntensityTier")
do ("To AmplitudeTier")
do ("Down to TableOfReal")
do ("Save as headerless spreadsheet file...", "{temp_directory}/AmplitudeTier {i}.txt")

# convert 
select TextGrid {i}
do ("Save as short text file...", "{temp_directory}/{i} short.TextGrid")
""", mapping))
        
        praat_script.write("""
do ("Quit")
""")
    finally:
        praat_script.close()


def intervals_from_praat(temp_directory_name, stimulus):
    # convert the text grids
    intervals = []
    text_grid_fname = temp_directory_name + '/' + str(stimulus) + ' short.TextGrid'
    print("Converting: " + text_grid_fname)
    with open(text_grid_fname) as file:
        interval = None
        previous_interval = None
        # number of header lines parsed, starting from "word" = 1
        # 0 = 2
        # length = 3
        # number of tiers = 4
        header_lines = 0
        for line in [line.replace('\n', '') for 
            line in file.readlines()]:
            # the line contains "" for the first empty space
            if header_lines == 0:
                if line == '"word"':
                    header_lines += 1
            elif header_lines < 4:
                header_lines += 1
            elif interval == None:
                interval = Interval()
                interval.start = float(line)
            elif interval.end == None:
                interval.end = float(line)
            else:
                interval.word = line[1:-1] # remove surrounding "
                
                if previous_interval != None:
                    if previous_interval.word == '':
                        # previous interval is a pause!
                        interval.silence_before = previous_interval.duration
                    elif interval.word == '':
                        # this is a pause
                        previous_interval.silence_after = interval.duration
                
                intervals.append(interval)
                previous_interval = interval
                interval = None
    
    return intervals


def intervals_to_csv(csv_delimiter, interval_dirname, stimulus, intervals):
    # write the intervals to disk
    with open(interval_dirname + '/' + str(stimulus) + '.csv', 'w') as file:
        file.write(csv_delimiter.join(['interval', 'word', 'start', 'end']) 
                   + '\n')
        i = 0
        
        for interval in intervals:
            file.write(csv_delimiter.join([str(i), 
                                           interval.word, 
                                           str(interval.start), 
                                           str(interval.end)]) + '\n')
            i += 1


def write_interval_line(current_interval, interval_file, timepoint, value):
    global csv_delimiter
    
    return interval_file.write(
        csv_delimiter.join([
            str(timepoint - current_interval.start + current_interval.offset), 
            str(value)]) + '\n')


def convert_amplitudes(temp_directory_name, amplitude_dirname, stimulus, intervals):
    amplitude_fname = temp_directory_name + '/AmplitudeTier ' + str(stimulus) + '.txt'
    print("Converting: " + amplitude_fname)
    
    first = True
    i = 0
    interval = intervals[i]
    nonempty = 0
    interval_filename = None
    try:
        interval_file = None
        with open(amplitude_fname) as file:
            try:
                for line in file.readlines():
                    if first:
                        # skip the header
                        first = False
                        continue
                
                    timepoint, pressure = [float(x) for x in line.split('\t')[1:3]]
                
                    # add the line
                    if interval_file != None:
                        write_interval_line(interval, interval_file, timepoint, 
                                            pressure)
                    
                    if timepoint < interval.end:
                        continue
                
                    # new interval! close the current interval
                    if interval_file != None:
                        interval_file.close()
                        interval_file = None                    
                        
                    i += 1
                
                    if i == len(intervals):
                        # No need to parse the rest of this file
                        return

                    # go to next interval
                    interval = intervals[i]
                    interval.offset = interval.start - timepoint 
                
                    if interval.word != '':
                        # non empty interval, make a new file
                        interval_filename = str.format(
                            '{0}/{1}_{2}_{3}.csv', 
                            amplitude_dirname, 
                            str(stimulus), 
                            str(nonempty), 
                            interval.word)
                    
                        interval_file = open(interval_filename, 'w')
                        nonempty += 1
                    
                        if timepoint >= interval.start:
                            write_interval_line(interval, interval_file, timepoint, 
                                                pressure)
            except Exception as e:
                file.seek(0)
                print("Dump of file: ")
                print("".join(file.readlines()))
                raise(e) 
    finally:
        if interval_file != None:
            interval_file.close()
            
def convert_pitches(temp_directory_name, pitch_dirname, stimulus, intervals):
    pitch_fname = temp_directory_name + '/PitchTier ' + str(stimulus) + '.txt'
    print("Converting: " + pitch_fname)
    
    i = 0
    interval = intervals[i]
    nonempty = 0
    
    previous_timepoint = None
    previous_pitch = None
    interval_filename = None
    
    try:
        interval_file = None
        with open(pitch_fname) as file:
            for line in file.readlines():
                timepoint, pitch = [float(x) for x in line.split('\t')]
                
                # add the line
                if interval_file != None:
                    write_interval_line(interval, interval_file, timepoint, 
                                        pitch)
                    
                if timepoint >= interval.end:
                    # new interval! close the current interval
                    if interval_file != None:
                        interval_file.close()
                        interval_file = None
                        
                    i += 1
                    
                    if i == len(intervals):
                        # No need to parse the rest of this file
                        return
    
                    # go to next interval
                    interval = intervals[i]
                    
                    if interval.word != '':
                        # non empty interval, make a new file
                        interval_filename = str.format(
                            '{0}/{1}_{2}_{3}.csv', 
                            pitch_dirname, 
                            str(stimulus), 
                            str(nonempty), 
                            interval.word)
                        
                        interval_file = open(interval_filename, 'w')
                        nonempty += 1
                        
                        if timepoint > interval.start:
                            # write the previous timepoint
                            write_interval_line(interval, interval_file, 
                                                previous_timepoint, 
                                                previous_pitch)
                          
                        if timepoint <= interval.end:  
                            write_interval_line(interval, interval_file, 
                                                timepoint, pitch)
                            
                previous_timepoint = timepoint
                previous_pitch = pitch    
    finally:
        if interval_file != None:
            interval_file.close()
            
def prepare_analysis(subject_id, sex, stimuli_count, exclude):
    global csv_delimiter, praat_text_encoding
    
    # base directory name of the script and experiment files
    base_dirname = os.path.split(os.path.abspath(__file__))[0]
    
    with tempfile.TemporaryDirectory() as temp_directory:
        # generate the Praat script
        script_location = temp_directory + '/script.praat'
        
        print('Generating script: ' + script_location)
        generate_praat_script(subject_id, sex, stimuli_count, base_dirname, 
                              temp_directory, script_location, exclude)
            
        # run Praat with this script
        print('Executing Praat script. This will take a while!')
        subprocess.check_output([praat_location, script_location])
        
        # prepare the interval directory
        interval_dirname = base_dirname + '/Output/Intervals/' + subject_id
        amplitude_dirname = base_dirname + '/Output/Amplitude/' + subject_id
        pitch_dirname = base_dirname + '/Output/Pitch/' + subject_id
            
        for dirname in [interval_dirname, amplitude_dirname, pitch_dirname]:
            # make sure the existing output is removed
            if os.path.exists(dirname):
                clear_directory(dirname)
            else:
                os.makedirs(dirname)
        
        sentences = stimuli.readfile(base_dirname + '/Output/Stimuli/' + 
                                     subject_id + '.txt', stimuli_count)
        invalid_sentences = {}
        
        for i in range(0, stimuli_count):
            if i in exclude:
                continue
            
            intervals = intervals_from_praat(temp_directory, i)
            intervals_to_csv(csv_delimiter, interval_dirname, i, intervals)            
            
            valid, message = check_intervals_with_sentence(intervals, 
                                                           sentences[i])
            
            if not valid:
                invalid_sentences[i] = message
            
            sentences[i].intervals = intervals
            
            # this will also update the offsets in the interval objects
            convert_amplitudes(temp_directory, amplitude_dirname, i, 
                               intervals)
            
            convert_pitches(temp_directory, pitch_dirname, i, intervals)
            
        if len(invalid_sentences) == 0:
            print("Stimuli validated against annotation. No problems found!")
            write_timings(base_dirname, subject_id, sentences, exclude)
        else:    
            for pos in invalid_sentences:
                print(str.format("ERROR IN {0}: {1}", pos, invalid_sentences[pos]))
                
            print("Skipped writing timings due to these errors.")

def write_timings(base_dirname, sid, sentences, exclude):    
    timingsdir = base_dirname + '/Output/Timings/'
    if not os.path.exists(timingsdir):
        os.makedirs(timingsdir)
    
    filename = timingsdir + sid + '.csv'
    
    if os.path.exists(filename):
        os.unlink(filename)
        
    with open(filename, 'w') as f:
        f.write(csv_delimiter.join(['participant', 'sentence_pos', 'carrier', 
                                    'sequence_length', 'word_pos', 'word_name', 
                                    'stress', 'syllables', 'silence_before', 
                                    'silence_after', 'duration']) + '\n')
        
        sentence_pos = 0
        for s in sentences:
            sentence_pos += 1
            
            if sentence_pos - 1 in exclude:
                continue
            
            word_intervals = [x for x in s.intervals if x.word != '']
        
            sequence_length = len(s.words)
            
            for word_i in range(0, len(s.words)):
                word = s.words[word_i]
                
                if (word_i == len(s.words) - 1):
                    # last word, skip the conjunction
                    interval = word_intervals[-1]
                    
                    # this silence duration is unusable
                    interval.silence_after = -99
                else:
                    # skip the last word of the carrier sentence
                    interval = word_intervals[word_i + 1]
                
                f.write(csv_delimiter.join(str(x) for x in 
                                           [sid, sentence_pos, s.carrier, 
                                            sequence_length, 
                                            word_i + 1, word.word,
                                            word.stress, word.syllables, 
                                            interval.silence_before,
                                            interval.silence_after,
                                            interval.duration]) + '\n')
            

def clear_directory(dirname):
    files = os.listdir(dirname)
    for f in files:
        os.unlink(dirname + '/' + f)
    
def check_intervals_with_sentence(intervals, sentence):
    word_i = 0
    
    word_count = len(sentence.words)
    
    for interval in intervals:
        if interval.word == '':
            continue
        
        if word_i == 0:
            check_word = sentence.carrier.split(' ')[-1]
        elif word_i == word_count:
            check_word = 'en'
        elif word_i == word_count+1:
            check_word = sentence.words[-1].word
        else:
            check_word = sentence.words[word_i-1].word
            
        if interval.word != check_word:
            return False, str.format("Interval[{0}]:{1} does not match {2}",
                                     word_i, 
                                     interval.word,
                                     check_word)
        
        word_i += 1
    
    if intervals[-1].word != '':
        return False, "Last interval is not empty"
    
    if word_i != len(sentence.words) + 2:
        return False, str.format("Number of intervals ({0}) do not match number in sentence ({1})",
                                 word_i - 2,
                                 len(sentence.words))
        
    return True, None  

main(sys.argv[1:])
