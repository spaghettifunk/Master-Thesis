form Hiss filter (low-pass)
	positive Frequency_(1000-20000_Hz) 7500
endform
if frequency < 1000
	frequency = 1000
endif
if frequency > 20000
	frequency = 20000
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	result = Filter (pass Hann band)... 0 'frequency' 100
	Rename... 's$'_hissfilter_'frequency'
	execute fixdc.praat
	select wrk
	Remove
	select result
endproc
