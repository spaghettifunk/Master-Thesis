form Rumble filter (high-pass)
	real Frequency_(0-1000_Hz) 120
endform
if frequency < 0
	frequency = 0
endif
if frequency > 1000
	frequency = 1000
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	result = Filter (pass Hann band)... 'frequency' 0 100
	Rename... 's$'_rumblefilter_'frequency'
	execute fixdc.praat
	select wrk
	Remove
	select result
endproc
