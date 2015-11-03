form Change duration
	positive New_duration_(s) 3.0
	choice Method 1
		button Stretch
		button Cut/add time
endform
include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	if method = 1
		dur = Get total duration
include minmaxf0.praat
		result = Lengthen (overlap-add)... 'minF0' 'maxF0' 'new_duration'/'dur'
	elsif method = 2
		result = Extract part... 0 new_duration rectangular 1 no
	endif
	Rename... 's$'_changeduration_'method$'_'new_duration:2'
	execute fixdc.praat
	select wrk
	Remove
	select result
endproc
