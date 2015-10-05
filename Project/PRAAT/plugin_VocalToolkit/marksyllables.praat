include batch.praat
procedure action
	s$ = selected$("Sound")
	s = selected("Sound")
	result = Copy... 's$'_markbysyllables
	execute fixdc.praat
	Scale... 0.9999
	dur = Get total duration
	intensity = To Intensity... 40 0.05 0
	intensitytier = Down to IntensityTier
	tableofreal = Down to TableOfReal
	pnts = Get number of rows
	for i from 1 to pnts
		pnt'i' = Get value... 'i' 1
		db'i' = Get value... 'i' 2
	endfor
	tg = Create TextGrid... 0.0 'dur' 's$'_markbysyllables 1
	thld = 40
	mrgn = 3
	for i from 1 to pnts
		tPnt = pnt'i'
		iPrev = i - 1
		iNext = i + 1
		db = db'i'
		if i<>1
			dbPrev = db'iPrev'
		else
			dbPrev = db
		endif
		if i<>pnts
			dbNext = db'iNext'
		else
			dbNext = db
		endif
		if ((db<thld and dbPrev>thld) or (db<thld and dbNext>thld)) or (db<dbPrev+mrgn and db<dbNext-mrgn and db>thld)
			Insert boundary... 1 tPnt
		endif
	endfor
	select intensity
	plus intensitytier
	plus tableofreal
	Remove
	select tg
	call zerocrossing
	plus result
	Edit
endproc
procedure zerocrossing
	numberOfTiers = Get number of intervals... 1
	select tg
	for i from 2 to numberOfTiers
		timeoriginal_'i' = Get starting point... 1 i
	endfor
	select s
	for i from 2 to numberOfTiers
		timeoriginal = timeoriginal_'i'
		zerocrossing_'i' = Get nearest zero crossing... Left timeoriginal
	endfor
	select tg
	for i from 2 to numberOfTiers
		timeoriginal = timeoriginal_'i'
		zerocrossing = zerocrossing_'i'
		Remove boundary at time... 1 timeoriginal
		nocheck Insert boundary... 1 zerocrossing
	endfor
endproc
