form Limiter
	real Threshold_(0-100_dB) 75
	boolean Normalize_output_signal 1
endform
if threshold < 0
	threshold = 0
endif
if threshold > 100
	threshold = 100
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	dur = Get total duration
	pow1 = Get power... 0 0
	intensity = To Intensity... 400 0 0
	intensitytier = Down to IntensityTier
	tableofreal = Down to TableOfReal
	numberOfRows = Get number of rows
	for i to numberOfRows
		time'i' = Get value... i 1
		n =  Get value... i 2
		dB'i' =  round(n*100)/100
		if dB'i' = undefined
			dB'i' = -300
		endif
	endfor
	limiter = Create IntensityTier... Limiter 0 dur
	for i to numberOfRows
		if dB'i' = 0
			Add point... time'i' 0
		else
			if dB'i' > threshold
				Add point... time'i' 1/dB'i'-(dB'i'-threshold)
			else
				Add point... time'i' 1/dB'i'
			endif
		endif
	endfor
	plus wrk
	tmp = Multiply
	pow2 = Get power... 0 0
	max = Get maximum... 0 0 None
	nocheck Scale...  max*sqrt(pow1/pow2)
	execute workpost.praat
	result = Rename... 's$'_limiter_'threshold'
	select wrk
	plus intensity
	plus intensitytier
	plus limiter
	plus tableofreal
	plus tmp
	Remove
	select result
	if normalize_output_signal
		Scale... 0.9999
	endif
endproc
