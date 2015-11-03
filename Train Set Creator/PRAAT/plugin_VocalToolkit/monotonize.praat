include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
include minmaxf0.praat
	tmp = Change gender... minF0 maxF0 1 0 0 1
	execute workpost.praat
	result = Rename... 's$'_monotone
	select wrk
	plus tmp
	Remove
	select result
endproc
