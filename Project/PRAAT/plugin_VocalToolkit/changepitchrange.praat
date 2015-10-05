form Change pitch range
	integer Pitch_range_(-200_to_200_%) 200
endform
if pitch_range < -200
	pitch_range = -200
endif
if pitch_range > 200
	pitch_range = 200
endif
pitch_range_factor = pitch_range/100
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	dur = Get total duration
include minmaxf0.praat
	pitch = To Pitch... 0.01 minF0 maxF0
	f0 = Get quantile... 0 0 0.50 Hertz
	f0 = 'f0:3'
	plus wrk
	manipulation = To Manipulation
	pitchtier = Extract pitch tier
	npoints = Get number of points
	for i to npoints
		tim'i' = Get time from index... i
		val'i' = Get value at index... i
	endfor
	Remove points between... 0 dur
	fref_st = 12*ln(f0/100)/ln(2)
	for i to npoints
		tim = tim'i'
		val = val'i'
		if val <> undefined
			f_st= fref_st+12*ln(val/f0)/ln(2)*pitch_range_factor
			newval = 100*exp(f_st*ln(2)/12)
			Add point... tim newval
		endif
	endfor
	plus manipulation
	Replace pitch tier
	select manipulation
	res = Get resynthesis (overlap-add)
	execute workpost.praat
	result = Rename... 's$'_changepitchrange_'pitch_range'
	select pitch
	plus wrk
	plus manipulation
	plus pitchtier
	plus res
	Remove
	select result
endproc
