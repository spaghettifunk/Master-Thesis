include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
include minmaxf0.praat
	tmp = Change speaker... minF0 maxF0 1 1 -1 1
	execute workpost.praat
	result = Rename... 's$'_invertpitch
	select wrk
	plus tmp
	Remove
	select result
endproc
