form Gate
	natural Threshold_(1-100_dB) 42
endform
if threshold > 100
	threshold = 100
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	result = Copy... 's$'_gate_'threshold'
	dur = Get total duration
	intensity = To Intensity... 100 0.01 0
	i = 0
	zerostart = 0
	zeroend = 0
	zeroselection = 0
	select result
	repeat
		dB = Object_'intensity' (i)
		if dB = undefined
			dB = 0
		endif
		if dB < threshold
			if zeroselection = 0
				zerostart = i
				zeroselection = 1
			endif
		else
			if zeroselection = 1
				zeroend = i
				if zerostart <> zeroend
					Set part to zero... zerostart zeroend at nearest zero crossing
				endif
				zeroselection = 0
			endif
		endif
		i = i + 0.01
	until i > dur
	if zeroselection = 1
		zeroend = i
		if zerostart <> zeroend
			Set part to zero... zerostart zeroend at nearest zero crossing
		endif
		zeroselection = 0
	endif
	select intensity
	Remove
	select result
endproc
