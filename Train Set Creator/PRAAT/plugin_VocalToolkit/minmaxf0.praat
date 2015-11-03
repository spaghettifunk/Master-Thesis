selsnd_m = selected("Sound")
nocheck To Pitch... 0 40 600
fullName$ = selected$()
type$ = extractWord$(fullName$, "")
if type$ = "Pitch"
	voicedframes = Count voiced frames
	if voicedframes > 0
		q25 = Get quantile... 0 0 0.25 Hertz
		q75 = Get quantile... 0 0 0.75 Hertz
		minF0 = round(q25 * 0.75)
		maxF0 = round(q75 * 1.5)
	else
		minF0 = 40
		maxF0 = 600
	endif
	Remove
else
	minF0 = 40
	maxF0 = 600
endif
select selsnd_m
