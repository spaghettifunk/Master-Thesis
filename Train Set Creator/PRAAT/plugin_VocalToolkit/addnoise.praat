form Add noise
	natural Volume_(1-100_dB) 40
endform
if volume > 100
	volume = 100
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	dur = Get total duration
	sr = Get sample rate
	ch = Get number of channels
	result = Create Sound from formula... 's$'_addnoise_'volume' 'ch' 0.0 dur sr randomUniform(-1, 1)
	Pre-emphasize (in-line)... 4000
	De-emphasize (in-line)... 400
	Scale intensity... volume
	Formula...  self + Object_'wrk'[]
	execute declip.praat
	select wrk
	Remove
	select result
endproc
