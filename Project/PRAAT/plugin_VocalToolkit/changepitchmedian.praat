form Change pitch median
	positive Pitch_median_(Hz) 130.81
endform
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
include minmaxf0.praat
	tmp = Change gender... minF0 maxF0 1 pitch_median 1 1
	execute workpost.praat
	result = Rename... 's$'_changepitchmedian_'pitch_median:2'
	select wrk
	plus tmp
	Remove
	select result
endproc
