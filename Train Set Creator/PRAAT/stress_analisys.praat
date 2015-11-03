########################################################
# NAME: Stress-analysis.praat                               
# Tae-Jin Yoon (UVic)
# March 3, 2008
#
# Davide: when you want to run this script, TextGrid with phonemes
#         and .wav files, must be in the same folder!
#
########################################################

form Measure duration, intensity and pitch (interval tier 1)
   word inputdir /Users/dado/Documents/University/Courses/Master-Thesis/Project/src/test-audio-data/
   comment The segments are located in tier number:
   integer tier_number 1
   comment output gets written to (path and filename incl extension):
   word outdir /Users/dado/Desktop/
   word output test_data
endform


# Read the list of files
;Read Strings from raw text file... 'inputdir$'/'listfile$'
Create Strings as file list... list 'inputdir$'/*.TextGrid
end = Get number of strings

for filecounter from 1 to end

   select Strings list
   file$ = Get string... filecounter
   fileName$ = file$ - ".TextGrid"
   waveFile$ = file$-".TextGrid"+".wav"


   Read from file... 'inputdir$'/'waveFile$'
   To Pitch... 0.01 75 500
   globalMeanF0 = Get mean... 0 0 Hertz
   globalstdevF0 = Get standard deviation... 0 0 Hertz


   select Sound 'fileName$'
   To Intensity... 100 0.0 Yes
    # getting overall statistic may not be necessary
   globalMeandB = Get mean... 0 0 dB
   globalstdevdB = Get standard deviation... 0 0


   Read from file... 'inputdir$'/'file$'

      
   select TextGrid 'fileName$'
   nlabels = Get number of intervals... 'tier_number'
   index = 1
   for label from 1 to nlabels
       labelx$ = Get label of interval... tier_number label

       if labelx$ <> ""
           w_begin = Get starting point... tier_number label
           w_end = Get end point... tier_number label
           w_duration = (w_end - w_begin)*1000

	   select Pitch 'fileName$'
	   meanF0 = Get mean... w_begin w_end Hertz
	   stdevF0 = Get standard deviation... w_begin w_end Hertz

	   select Intensity 'fileName$'
	   meandB = Get mean... w_begin w_end dB
   	   stdevdB = Get standard deviation... w_begin w_end
                                 
	   printline 'file$' 'labelx$' 'index' 'w_duration:2' 'globalMeanF0:2' 'globalstdevF0:2'
		... 'meanF0:2' 'stdevF0:2' 'meandB:2' 'stdevdB:2'

	   fileappend 'outdir$'/'output$' 'fileName$''tab$''labelx$''tab$''index''tab$'
		...'w_duration:2''tab$''globalMeanF0:2''tab$''globalstdevF0:2''tab$'
		...'meanF0:2''tab$''stdevF0:2''tab$''0.0''tab$''stdevdB:2'
		...'newline$'

	   index = index + 1


	   select TextGrid 'fileName$'
       endif

   endfor

endfor

select all
Remove