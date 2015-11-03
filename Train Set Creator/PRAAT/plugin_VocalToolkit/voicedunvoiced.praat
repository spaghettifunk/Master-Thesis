include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	sr = Get sample rate
include minmaxf0.praat
	pointprocess = To PointProcess (periodic, cc)... minF0 maxF0
	tg = To TextGrid (vuv)... 0.02 0.01
	call zerocrossing wrk
	select wrk
	plus tg
	Extract all intervals... 1 yes
	n = numberOfSelected ("Sound")
	partsV = 0
	for i to n
		ids'i' = selected("Sound", i)
		ids$ = selected$("Sound", i)
		if ids$ = "V"
			partsV += 1
			v_'partsV' = selected("Sound", i)
		endif
	endfor
	select v_1
	for i from 2 to partsV
		plus v_'i'
	endfor
	Set part to zero... 0 0 at exactly these times
	select ids1
	for i to n
		plus ids'i'
	endfor
	unvoiced = Concatenate
	Rename... 's$'_unvoiced
	select ids1
	for i to n
		plus ids'i'
	endfor
	Remove
	select wrk
	plus tg
	Extract all intervals... 1 yes
	n = numberOfSelected ("Sound")
	partsU = 0
	for i to n
		ids'i' = selected("Sound", i)
		ids$ = selected$("Sound", i)
		if ids$ = "U"
			partsU += 1
			u_'partsU' = selected("Sound", i)
		endif
	endfor
	select u_1
	for i from 2 to partsU
		plus u_'i'
	endfor
	Set part to zero... 0 0 at exactly these times
	select ids1
	for i to n
		plus ids'i'
	endfor
	voiced = Concatenate
	Rename... 's$'_voiced
	select ids1
	for i to n
		plus ids'i'
	endfor
	plus wrk
	plus pointprocess
	plus tg
	Remove
	select voiced
endproc
procedure zerocrossing sId
	numberOfTiers = Get number of intervals... 1
	select tg
	for i from 1 to numberOfTiers
		tiertext_'i'$ = Get label of interval... 1 i
	endfor
	select tg
	for i from 2 to numberOfTiers
		timeoriginal_'i' = Get starting point... 1 i
	endfor
	select sId
	for i from 2 to numberOfTiers
		timeoriginal = timeoriginal_'i'
		zerocrossing_'i' = Get nearest zero crossing... Left timeoriginal
	endfor
	select tg
	for i from 2 to numberOfTiers
		timeoriginal = timeoriginal_'i'
		zerocrossing = zerocrossing_'i'
		Remove boundary at time... 1 timeoriginal
		Insert boundary... 1 zerocrossing
	endfor
	select tg
	for i from 1 to numberOfTiers
		tiertext$ = tiertext_'i'$
		Set interval text... 1 i 'tiertext$'
	endfor
endproc
