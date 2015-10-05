include batch.praat
procedure action
	clip = Get absolute extremum... 0 0 None
	if clip = undefined
		clip = 0
	endif
	clip = 'clip:4'
	if clip >= 1
		Scale... 0.9999
	endif
endproc
