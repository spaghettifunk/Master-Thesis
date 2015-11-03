form Raspiness
	natural Raspiness_(1-100_%) 20
endform
if raspiness > 100
	raspiness = 100
endif
value = raspiness/100000
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	dur = Get total duration
	pitch = To Pitch... 0 40 600
	select wrk
	plus pitch
	manipulation = To Manipulation
	durationtier = Create DurationTier... 's$' 0 dur
	Add point... 0 1
	plus manipulation
	Replace duration tier
	select pitch
	pointprocess = To PointProcess
	matrix = To Matrix
	Formula...  self + randomGauss(0, 'value')
	pointprocess2 = To PointProcess
	select manipulation
	plus pointprocess2
	Replace pulses
	select manipulation
	resynthesis = Get resynthesis (overlap-add)
	execute workpost.praat
	result = Rename... 's$'_raspiness_'raspiness'
	select wrk
	plus pitch
	plus manipulation
	plus pointprocess
	plus matrix
	plus durationtier
	plus pointprocess2
	plus resynthesis
	Remove
	select result
endproc
