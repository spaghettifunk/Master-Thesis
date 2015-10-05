form Vibrato and Tremolo
	positive Semitones_(average)_(0.5-12) 1.0
	positive Pulses_per_second_(1-10) 5.5
endform
pulses = pi*(pulses_per_second*2)/100
if semitones < 0.5
	semitones = 0.5
endif
if semitones > 12
	semitones = 12
endif
if pulses_per_second < 1
	pulses_per_second = 1
endif
if pulses_per_second > 10
	pulses_per_second = 10
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	dur = Get total duration
	if dur < 0.5
		exit Sound too short: 'dur:4' s. (Min.: 0.5 s.)
	endif
	pow1 = Get power... 0 0
include minmaxf0.praat
	manipulation = To Manipulation... 0.01 minF0 maxF0
	Extract pitch tier
	for i from 0 to dur*100
		val_'i' = Get value at time... i/100
	endfor
	Remove
	vibrato = Create PitchTier... Vibrato 0 dur
	tremolo = Create IntensityTier... Tremolo 0 dur
	call VibratoTremolo round(dur*100) semitones
	select manipulation
	plus vibrato
	Replace pitch tier
	durationtier = Create DurationTier... 's$' 0 dur
	Add point... 0 1
	plus manipulation
	Replace duration tier
	select manipulation
	res = Get resynthesis (overlap-add)
	plus tremolo
	tmp = Multiply... yes
	pow2 = Get power... 0 0
	max = Get maximum... 0 0 None
	nocheck Scale...  max*sqrt(pow1/pow2)
	execute workpost.praat
	result = Rename... 's$'_vibratotremolo
	select wrk
	plus manipulation
	plus vibrato
	plus res
	plus tremolo
	plus durationtier
	plus tmp
	Remove
	select result
endproc
procedure VibratoTremolo tim vib
	select vibrato
	ramp = 0
	for i from 0 to tim-1
		val = val_'i'
		if vib<>0
			if i <= tim - 25 and ramp <= 1
					ramp = ramp + 0.04
					if ramp > 1
						ramp = 1
					endif
			else
					ramp = ramp - 0.04
					if ramp < 0
						ramp = 0
					endif
			endif
			b = 12*ln(val/261.63)/ln(2)
			c = b + (vib/2) * sin(pulses * i) * ramp
			c = 261.63*exp(c*ln(2)/12)
		else
			c = val
		endif
		Add point... i/100 c
	endfor
	if vib<>0
		select tremolo
		db = 90
		ramp = 0
		for i from 0 to tim-1
			if i < 25 and i < tim - 25
				ramp = ramp - 0.02
				ramp = 'ramp:2'
			endif
			if i > tim - 25
				ramp = ramp + 0.02
				ramp = 'ramp:2'
			endif
			newdb = db + ramp - (ramp * sin(pulses * i) + ramp)
			Add point... i/100 newdb
		endfor
	endif
endproc
