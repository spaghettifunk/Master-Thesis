form Change semitones
	real Semitones_(-24_+24) 12
endform
if semitones < -24
	semitones = -24
endif
if semitones > 24
	semitones = 24
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	if semitones = 0
		Copy... 's$'_changesemitones_0
	else
		execute workpre.praat
		wrk = selected("Sound")
include minmaxf0.praat
		pitch = To Pitch... 0.01 minF0 maxF0
		f0 = Get quantile... 0 0 0.50 Hertz
		f0 = 'f0:3'
		hz = f0*exp(semitones*ln(2)/12)
		hz = 'hz:3'
		plus wrk
		tmp = Change gender... 1 hz 1 1
		execute workpost.praat
		result = Rename... 's$'_changesemitones_'semitones'
		select wrk
		plus pitch
		plus tmp
		Remove
		select result
	endif
endproc
