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
	tmp = Kill octave jumps
	pitchtier = Down to PitchTier
	plus manipulation
	Replace pitch tier
	select manipulation
	res = Get resynthesis (overlap-add)
	execute workpost.praat
	result = Rename... 's$'_kill_octave_jumps
	select wrk
	plus pitch
	plus manipulation
	plus tmp
	plus pitchtier
	plus res
	Remove
	select result
endproc
