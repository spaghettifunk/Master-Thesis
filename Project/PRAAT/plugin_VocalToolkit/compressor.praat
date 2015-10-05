form Compressor
	natural Compression_(1-100_%) 25
	boolean Normalize_output_signal 1
endform
if compression > 100
	compression = 100
endif
comp = compression/100
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	pow1 = Get power... 0 0
	intensity = To Intensity... 400 0 0
	Formula... -self*'comp'
	intensitytier = Down to IntensityTier
	plus wrk
	tmp = Multiply
	pow2 = Get power... 0 0
	max = Get maximum... 0 0 None
	nocheck Scale...  max*sqrt(pow1/pow2)
	execute workpost.praat
	result = Rename... 's$'_compressor_'compression'
	select wrk
	plus intensity
	plus intensitytier
	plus tmp
	Remove
	select result
	if normalize_output_signal
		Scale... 0.9999
	endif
endproc
