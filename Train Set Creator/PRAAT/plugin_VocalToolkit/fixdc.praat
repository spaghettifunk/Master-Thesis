include batch.praat
procedure action
	s = selected("Sound")
	Subtract mean
	tmp = Filter (pass Hann band)... 60 0 20
	execute declip.praat
	select s
	Formula... Object_'tmp'[]
	select tmp
	Remove
	select s
	stt = Get start time
	if stt <> 0
		dur = Get total duration
		Scale times to... 0 dur
	endif
endproc
