include batch.praat
procedure action
	s$ = selected$("Sound")
	Copy... 's$'_normalized
	Scale... 0.9999
endproc
