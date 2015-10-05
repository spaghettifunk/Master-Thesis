form Pitch Smoothing
	natural Pitch_smoothing_(1-100_%) 50
endform
if pitch_smoothing > 100
	pitch_smoothing = 100
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
include minmaxf0.praat
	pitch = To Pitch... 0.01 minF0 maxF0
	plus wrk
	manipulation = To Manipulation
	select pitch
	smoothedpitch = Smooth... -6*ln('pitch_smoothing'/10)+15
	pitchtier = Down to PitchTier
	plus manipulation
	Replace pitch tier
	select manipulation
	res = Get resynthesis (overlap-add)
	execute workpost.praat
	result = Rename... 's$'_pitchsmoothing_'pitch_smoothing'
	select wrk
	plus pitch
	plus manipulation
	plus smoothedpitch
	plus pitchtier
	plus res
	Remove
	select result
endproc
