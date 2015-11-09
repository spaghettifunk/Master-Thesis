# Usage: praat .wav /directory/out/ wav 10 75 500 11025

form Get_arguments
  text infile
  text outfile
  sentence type_file
  positive prediction_order
  positive minimum_pitch
  positive maximum_pitch
  positive new_sample_rate
endform

Read from file... 'infile$'
numSelected = numberOfSelected ("Sound")

# change the name of each file - for batch processing
for i to numSelected
	select all
	currSoundID = selected ("Sound", i)
	select 'currSoundID'
	currName$ = "word_'i'"
	Rename... 'currName$'
endfor

for i to numSelected
	select Sound word_'i'

	# get the finishing time of the Sound file
	fTime = Get finishing time

	# Use numTimes in the loop
	numTimes = fTime / 0.01
	newName$ = "word_'i'"
	select Sound word_'i'

	# 1st argument: New sample rate 2nd argument: Precision (samples)
	Resample... 'new_sample_rate' 50

	# 1st argument: Time step (s), 2nd argument: Minimum pitch for Analysis,
	# 3rd argument: Maximum pitch for Analysis
	To Pitch... 0.01 'minimum_pitch' 'maximum_pitch'
	Rename... 'newName$'

	Create Table... table_word_'i' numTimes 1
	Set column label (index)... 1 pitch

	for itime to numTimes
		select Pitch word_'i'
		curtime = 0.01 * itime
		f0 = 0
		f0 = Get value at time... 'curtime' Hertz Linear
		f0$ = fixed$ (f0, 2)

		if f0$ = "--undefined--"
			f0$ = "0"
		endif

		select Table table_word_'i'
		Set numeric value... itime pitch 'f0$'
	endfor

	select Table table_word_1
	Save as comma-separated file... 'outfile$'
endfor
